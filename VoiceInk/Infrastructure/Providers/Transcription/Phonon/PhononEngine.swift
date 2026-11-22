import Foundation
import PhononCoreML
import os

/// `Transcriber` carries no Sendable annotation, yet sharing it across threads is safe here: every decode runs on
/// `PhononEngine.queue` (serial), and the package guards its background function loads with its own locks
/// (`modelLock`, `bgLoads`), which is what `waitForBackgroundLoads()` and `loadedFunctions` touch.
private final class TranscriberHandle: @unchecked Sendable {
    let transcriber: Transcriber

    init(_ transcriber: Transcriber) {
        self.transcriber = transcriber
    }
}

/// Phonon-2 through Fermion's phonon-coreml package, run the way Deta runs it: the model bundle is opened once per
/// process (first run compiles the Core ML package and caches it under ~/Library/Caches/phonon-coreml), followed by
/// a probe decode of one second of low noise, with the remaining encoder functions loaded in the background. One
/// instance serves prewarm, batch and realtime so the open is paid once, at launch, and never during a dictation.
final class PhononEngine: @unchecked Sendable {
    static let shared = PhononEngine()

    enum State: Equatable {
        case idle
        case opening
        case ready
    }

    static let sampleRate = 16_000
    /// Deta pads every decode with 0.25 s of silence at both ends.
    static let paddingSamples = Int(0.25 * Double(sampleRate))

    private let logger = Logger(subsystem: "com.prakashjoshipax.voiceink", category: "PhononEngine")
    private let queue = DispatchQueue(label: "com.prakashjoshipax.voiceink.phonon", qos: .userInitiated)
    private let batchQueue = DispatchQueue(label: "com.prakashjoshipax.voiceink.phonon.batch", qos: .utility)
    private let stateLock = NSLock()
    private var handle: TranscriberHandle?
    private var bundleURL: URL?
    private var state: State = .idle
    private var openTask: Task<Void, Error>?
    private var openGeneration: UInt64 = 0
    private var activeSessions = 0
    private var activeSessionsHandler: (@MainActor (Bool) -> Void)?

    private init() {}

    var currentState: State {
        stateLock.withLock { state }
    }

    var hasActiveSessions: Bool {
        stateLock.withLock { activeSessions > 0 }
    }

    func setActiveSessionsHandler(_ handler: (@MainActor (Bool) -> Void)?) {
        let inUse = stateLock.withLock { () -> Bool in
            activeSessionsHandler = handler
            return activeSessions > 0
        }
        if let handler {
            Task { @MainActor in handler(inUse) }
        }
    }

    func beginSession() {
        let (inUse, handler) = stateLock.withLock { () -> (Bool, (@MainActor (Bool) -> Void)?) in
            activeSessions += 1
            return (activeSessions > 0, activeSessionsHandler)
        }
        if let handler {
            Task { @MainActor in handler(inUse) }
        }
    }

    func endSession() {
        let (inUse, handler) = stateLock.withLock { () -> (Bool, (@MainActor (Bool) -> Void)?) in
            activeSessions = max(0, activeSessions - 1)
            return (activeSessions > 0, activeSessionsHandler)
        }
        if let handler {
            Task { @MainActor in handler(inUse) }
        }
    }

    /// Opens `bundle` if it is not already the open bundle. Concurrent callers share one open.
    /// `onCompile` fires when the first-run Core ML compile is about to happen (about a minute, once per Mac).
    func open(bundle: URL, onCompile: (@Sendable () -> Void)? = nil) async throws {
        let existingTask: Task<Void, Error>? = stateLock.withLock {
            if handle != nil, bundleURL == bundle, state == .ready { return nil }
            if let openTask, bundleURL == bundle { return openTask }
            openGeneration &+= 1
            let generation = openGeneration
            let task = Task { [self] in
                try await self.performOpen(bundle: bundle, generation: generation, onCompile: onCompile)
            }
            openTask = task
            bundleURL = bundle
            state = .opening
            return task
        }
        guard let existingTask else { return }
        try await existingTask.value
    }

    private func performOpen(bundle: URL, generation: UInt64, onCompile: (@Sendable () -> Void)?) async throws {
        let start = Date()
        let logger = self.logger
        let needsCompile = !Self.compiledModelExists(for: bundle)
        if needsCompile { onCompile?() }

        let handle: TranscriberHandle
        do {
            handle = try await withCheckedThrowingContinuation { continuation in
                queue.async {
                    do {
                        var options = Transcriber.Options()
                        options.backgroundLoad = true
                        // As Deta's open: resident first is the function covering the largest one-shot window
                        // (35 s), which serves every hold. The package default (15 s) would load the 35 s
                        // function synchronously mid-decode.
                        let manifest = try Self.manifest(for: bundle)
                        let singleShot = manifest.single_shot_max_s ?? manifest.functions_s.max() ?? 15
                        if let covering = manifest.functions_s.sorted().first(where: { $0 >= singleShot }) {
                            options.eagerFunctions = [covering]
                        }
                        options.progress = { phase, seconds in
                            logger.notice("Phonon-2 \(phase, privacy: .public) at \(seconds, format: .fixed(precision: 2))s")
                        }
                        let transcriber = try Transcriber(bundle: bundle, options: options)
                        // Deta probes one second of low noise right after the open: an unsupported op surfaces at the first
                        // prediction, not at load.
                        var generator = SystemRandomNumberGenerator()
                        let probe = (0..<Self.sampleRate).map { _ in Float.random(in: -0.01...0.01, using: &generator) }
                        _ = try transcriber.transcribe(probe)
                        continuation.resume(returning: TranscriberHandle(transcriber))
                    } catch {
                        continuation.resume(throwing: error)
                    }
                }
            }
        } catch {
            stateLock.withLock {
                if self.openGeneration == generation {
                    self.openTask = nil
                    self.bundleURL = nil
                    self.state = .idle
                }
            }
            throw error
        }

        let committed = stateLock.withLock { () -> Bool in
            guard self.openGeneration == generation else { return false }
            self.handle = handle
            self.state = .ready
            self.openTask = nil
            return true
        }
        guard committed else {
            throw CancellationError()
        }

        let elapsed = Date().timeIntervalSince(start)
        let compile = handle.transcriber.compileSeconds
        let load = handle.transcriber.loadSeconds
        logger.notice(
            "Phonon-2 ready in \(elapsed, format: .fixed(precision: 2))s compile=\(compile, format: .fixed(precision: 2))s load=\(load, format: .fixed(precision: 2))s"
        )

        // The other encoder functions load on the package's background queue; until one is ready, the next
        // larger resident function serves its windows. Waiting for them on `queue` would block every decode,
        // so wait elsewhere.
        DispatchQueue.global(qos: .utility).async {
            handle.transcriber.waitForBackgroundLoads()
            logger.notice("Phonon-2 encoder functions loaded: \(handle.transcriber.loadedFunctions, privacy: .public)")
        }
    }

    /// Mono 16 kHz samples -> transcript, padded with silence at both ends as Deta does.
    func transcribe(_ samples: [Float]) async throws -> Transcript {
        try Task.checkCancellation()
        let handle = try readyHandle()
        beginSession()
        defer { endSession() }
        return try await withCheckedThrowingContinuation { continuation in
            queue.async {
                let padding = [Float](repeating: 0, count: Self.paddingSamples)
                continuation.resume(with: Result { try handle.transcriber.transcribe(padding + samples + padding) })
            }
        }
    }

    struct FileTranscript: Sendable {
        let text: String
        let audioSeconds: Double
        let decodeSeconds: Double
        let windows: Int
    }

    private final class FileSourceHandle: @unchecked Sendable {
        let source: FileSource
        init(_ source: FileSource) { self.source = source }
    }

    private struct PlannedWindow: Sendable {
        let start: Int
        let end: Int
        let overlapFromPrevious: Bool
        let splitSample: Int?
    }

    private struct StitchedWord {
        let text: String
        let start: Double
        let end: Double
    }

    /// A recorded file, planned off the realtime queue and decoded one window at a time so realtime
    /// dictations can interleave between windows and cancellation interrupts promptly.
    func transcribe(url: URL) async throws -> FileTranscript {
        try Task.checkCancellation()
        let handle = try readyHandle()
        beginSession()
        defer { endSession() }

        let tStart = Date()
        let (sourceHandle, windows): (FileSourceHandle, [PlannedWindow]) = try await withCheckedThrowingContinuation {
            continuation in
            batchQueue.async {
                continuation.resume(with: Result {
                    let source = try FileSource(url: url)
                    let planned = try handle.transcriber.plan(source: source).map {
                        PlannedWindow(
                            start: $0.start,
                            end: $0.end,
                            overlapFromPrevious: $0.overlapFromPrevious,
                            splitSample: $0.splitSample
                        )
                    }
                    return (FileSourceHandle(source), planned)
                })
            }
        }

        let sampleRate = Double(Self.sampleRate)
        var words: [StitchedWord] = []

        for window in windows {
            try Task.checkCancellation()
            let samples: [Float] = try await withCheckedThrowingContinuation { continuation in
                batchQueue.async {
                    continuation.resume(with: Result { try sourceHandle.source.read(window.start..<window.end) })
                }
            }
            try Task.checkCancellation()
            let windowWords: [StitchedWord] = try await withCheckedThrowingContinuation { continuation in
                queue.async {
                    continuation.resume(with: Result {
                        let transcript = try handle.transcriber.transcribe(samples)
                        let offset = Double(window.start) / sampleRate
                        let limit = Double(window.end - window.start) / sampleRate
                        return transcript.words.map { word in
                            let clampedEnd = min(word.end, limit)
                            let start = ((word.start + offset) * 1000).rounded() / 1000
                            let end = ((max(clampedEnd, word.start) + offset) * 1000).rounded() / 1000
                            return StitchedWord(text: word.text, start: start, end: end)
                        }
                    })
                }
            }
            if window.overlapFromPrevious, let splitSample = window.splitSample {
                Self.stitch(kept: &words, incoming: windowWords, split: Double(splitSample) / sampleRate)
            } else {
                words.append(contentsOf: windowWords)
            }
        }

        let wall = Date().timeIntervalSince(tStart)
        return FileTranscript(
            text: words.map(\.text).joined(separator: " "),
            audioSeconds: Double(sourceHandle.source.count) / sampleRate,
            decodeSeconds: wall,
            windows: windows.count
        )
    }

    private static func stitch(kept: inout [StitchedWord], incoming: [StitchedWord], split: Double, jitter: Double = 0.25) {
        while let last = kept.last, last.start >= split { kept.removeLast() }
        var seamA: [StitchedWord] = []
        var index = kept.count - 1
        while index >= 0 && kept[index].end > split - jitter - 0.5 {
            seamA.append(kept[index])
            index -= 1
        }
        for word in incoming where word.start >= split - jitter {
            var duplicate = false
            for existing in seamA {
                let overlapStart = max(existing.start, word.start)
                let overlapEnd = min(existing.end, word.end)
                if overlapStart >= overlapEnd + 1e-9 { continue }
                let sameText = existing.text.lowercased() == word.text.lowercased()
                let shorter = min(existing.end - existing.start, word.end - word.start)
                let overlap = overlapEnd - overlapStart
                if sameText || overlap > 0.5 * shorter {
                    duplicate = true
                    break
                }
            }
            if !duplicate { kept.append(word) }
        }
    }

    func unload() async {
        let openTask: Task<Void, Error>? = stateLock.withLock {
            openGeneration &+= 1
            return self.openTask
        }
        openTask?.cancel()
        _ = try? await openTask?.value
        await withCheckedContinuation { continuation in
            queue.async {
                self.stateLock.withLock {
                    self.handle = nil
                    self.bundleURL = nil
                    self.state = .idle
                    self.openTask = nil
                }
                continuation.resume()
            }
        }
    }

    private func readyHandle() throws -> TranscriberHandle {
        guard let handle = stateLock.withLock({ state == .ready ? handle : nil }) else {
            throw PhononEngineError.notOpen
        }
        return handle
    }

    // MARK: - Compile cache

    static func manifest(for bundle: URL) throws -> Manifest {
        try JSONDecoder().decode(Manifest.self, from: Data(contentsOf: bundle.appendingPathComponent("manifest.json")))
    }

    /// Where the package keeps the compiled model for `bundle` (the Neural Engine cache is keyed on this path).
    static func compiledModelCacheDirectory(for bundle: URL) -> URL? {
        guard let manifest = try? manifest(for: bundle) else { return nil }
        return FileManager.default.urls(for: .cachesDirectory, in: .userDomainMask)[0]
            .appendingPathComponent("phonon-coreml")
            .appendingPathComponent(Transcriber.cacheKey(bundle: bundle, manifest: manifest))
    }

    static func compiledModelExists(for bundle: URL) -> Bool {
        guard let manifest = try? manifest(for: bundle), let cacheDirectory = compiledModelCacheDirectory(for: bundle)
        else { return false }
        let compiledName = (manifest.multifunction as NSString).deletingPathExtension + ".mlmodelc"
        return FileManager.default.fileExists(atPath: bundle.appendingPathComponent(compiledName).path)
            || FileManager.default.fileExists(atPath: cacheDirectory.appendingPathComponent(compiledName).path)
    }
}

enum PhononEngineError: LocalizedError {
    case notOpen

    var errorDescription: String? {
        switch self {
        case .notOpen:
            return String(localized: "Phonon-2 is not loaded.")
        }
    }
}

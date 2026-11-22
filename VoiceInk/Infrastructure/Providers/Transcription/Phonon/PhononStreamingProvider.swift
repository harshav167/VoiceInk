import Foundation
import PhononCoreML
import os

/// Realtime Phonon-2 the way Deta's engine does it: every partial re-decodes the audio since the last commit (one
/// Neural Engine window), paced to the measured decode cost; once 36 s have piled up past the last commit the hold is
/// cut at a sentence end (or the quietest 300 ms) near 28 s and that piece is decoded as final text; on stop the
/// remainder is decoded once. No agreement engine: each partial is the latest full hypothesis for its span.
final class PhononStreamingProvider: StreamingTranscriptionProvider {
    private static let sampleRate = Double(PhononEngine.sampleRate)
    // Deta's shipped cadence (phonon_app/config.py): first decode at 350 ms of audio, then gaps of 350 and 450 ms
    // before the steady 500 ms. Deta measured key-down to first word falling from 1.46 s to 0.80 s with it.
    private static let firstPartialSeconds = 0.35
    private static let partialRampSeconds = [0.35, 0.45]
    private static let partialGapSeconds = 0.5
    /// The next decode waits at least this multiple of the last decode's wall time so decodes never stack up.
    private static let selfClockFactor = 1.5
    private static let commitAtSeconds = 36.0
    private static let commitTargetSeconds = 28.0
    private static let commitSearchSeconds = 6.0
    private static let quietWindowSeconds = 0.3

    private let logger = Logger(subsystem: "com.prakashjoshipax.voiceink", category: "PhononStreaming")
    private let engine = PhononEngine.shared
    private var eventsContinuation: AsyncStream<StreamingTranscriptionEvent>.Continuation?
    private(set) var transcriptionEvents: AsyncStream<StreamingTranscriptionEvent>
    private var finalizationContinuation: AsyncStream<String>.Continuation?
    private(set) var finalizationEvents: AsyncStream<String>?

    private let bufferLock = NSLock()
    /// Audio not yet committed; `committedSampleCount` samples before it have been decoded as final text.
    private var audioBuffer: [Float] = []
    private var committedSampleCount = 0
    private var committedSegments: [String] = []
    private var lastPartialWords: [Word] = []
    private var progressiveCommitFailed = false
    private var hasBegunEngineSession = false
    private var partialTask: Task<Void, Never>?
    /// Written only by the partial task; read in `commit()` once that task has finished.
    private var partialDecodes = PartialDecodeStats()

    private struct PartialDecodeStats {
        var count = 0
        var totalSeconds = 0.0
        var maxSeconds = 0.0

        mutating func record(_ seconds: TimeInterval) {
            count += 1
            totalSeconds += seconds
            maxSeconds = max(maxSeconds, seconds)
        }
    }

    init() {
        var continuation: AsyncStream<StreamingTranscriptionEvent>.Continuation!
        transcriptionEvents = AsyncStream { continuation = $0 }
        eventsContinuation = continuation

        var finalContinuation: AsyncStream<String>.Continuation!
        finalizationEvents = AsyncStream { finalContinuation = $0 }
        finalizationContinuation = finalContinuation
    }

    deinit {
        partialTask?.cancel()
        endEngineSessionIfNeeded()
        finalizationContinuation?.finish()
        eventsContinuation?.finish()
    }

    var stopDisposition: StreamingStopDisposition {
        bufferLock.withLock { progressiveCommitFailed ? .useBatchFallback : .finalizeStreaming }
    }

    private func beginEngineSessionIfNeeded() {
        let shouldBegin = bufferLock.withLock { () -> Bool in
            guard !hasBegunEngineSession else { return false }
            hasBegunEngineSession = true
            return true
        }
        if shouldBegin {
            engine.beginSession()
        }
    }

    private func endEngineSessionIfNeeded() {
        let shouldEnd = bufferLock.withLock { () -> Bool in
            guard hasBegunEngineSession else { return false }
            hasBegunEngineSession = false
            return true
        }
        if shouldEnd {
            engine.endSession()
        }
    }

    func connect(model: any TranscriptionModel, language: String?) async throws {
        let start = Date()
        let bundle = await PhononModelManager.shared.bundleDirectory
        beginEngineSessionIfNeeded()
        do {
            try await engine.open(bundle: bundle)
        } catch {
            endEngineSessionIfNeeded()
            throw error
        }
        bufferLock.withLock {
            audioBuffer = []
            committedSampleCount = 0
            committedSegments = []
            lastPartialWords = []
            progressiveCommitFailed = false
        }
        partialDecodes = PartialDecodeStats()
        startPartialLoop()
        eventsContinuation?.yield(.sessionStarted)
        logger.notice(
            "Phonon-2 streaming started connect=\(Date().timeIntervalSince(start), format: .fixed(precision: 3))s"
        )
    }

    func sendAudioChunk(_ data: Data) async throws {
        let samples = PCMAudioConverter.float32Samples(fromPCM16Data: data)
        bufferLock.withLock {
            guard !progressiveCommitFailed else { return }
            audioBuffer.append(contentsOf: samples)
        }
    }

    func commit() async throws {
        partialTask?.cancel()
        await partialTask?.value
        partialTask = nil
        let live = partialDecodes
        let average = live.count > 0 ? live.totalSeconds / Double(live.count) : 0
        logger.notice(
            """
            Phonon-2 live decodes count=\(live.count) \
            avg=\(average, format: .fixed(precision: 3))s max=\(live.maxSeconds, format: .fixed(precision: 3))s
            """
        )

        let (remainder, committed, failedProgressive) = bufferLock.withLock {
            (audioBuffer, committedSegments, progressiveCommitFailed)
        }
        if failedProgressive {
            finalizationContinuation?.finish()
            throw StreamingTranscriptionError.serverError("Phonon-2 progressive commit failed")
        }

        var tailText = ""
        if !remainder.isEmpty {
            do {
                let start = Date()
                tailText = try await engine.transcribe(remainder).text.trimmingCharacters(in: .whitespacesAndNewlines)
                let seconds = Double(remainder.count) / Self.sampleRate
                logger.notice(
                    "Phonon-2 final decode \(seconds, format: .fixed(precision: 1))s in \(Date().timeIntervalSince(start), format: .fixed(precision: 3))s"
                )
            } catch {
                logger.error("Phonon-2 final decode failed: \(error, privacy: .public)")
                eventsContinuation?.yield(.error(error))
                finalizationContinuation?.finish()
                throw error
            }
        }

        let allSegments = tailText.isEmpty ? committed : committed + [tailText]
        let fullText = allSegments.joined(separator: " ")
        finalizationContinuation?.yield(fullText)
        finalizationContinuation?.finish()
    }

    func disconnect() async {
        partialTask?.cancel()
        await partialTask?.value
        partialTask = nil
        bufferLock.withLock {
            audioBuffer = []
            committedSampleCount = 0
            committedSegments = []
            lastPartialWords = []
            progressiveCommitFailed = false
        }
        endEngineSessionIfNeeded()
        finalizationContinuation?.finish()
        eventsContinuation?.finish()
        logger.notice("Phonon-2 streaming disconnected")
    }

    // MARK: - Partials

    private func startPartialLoop() {
        partialTask = Task { [weak self] in
            var nextPartialAt = Int(Self.firstPartialSeconds * Self.sampleRate)
            var emitted = 0
            while !Task.isCancelled {
                try? await Task.sleep(for: .milliseconds(50))
                guard let self, !Task.isCancelled else { break }
                let (available, failed) = self.bufferLock.withLock {
                    (self.audioBuffer.count, self.progressiveCommitFailed)
                }
                if failed { break }
                guard available >= nextPartialAt else { continue }

                if Double(available) / Self.sampleRate >= Self.commitAtSeconds {
                    let succeeded = await self.commitPiece()
                    if !succeeded { break }
                    nextPartialAt = Int(Self.firstPartialSeconds * Self.sampleRate)
                    continue
                }

                let wall = await self.decodePartial()
                emitted += 1
                let ramp = Self.partialRampSeconds
                var gap = emitted <= ramp.count ? ramp[emitted - 1] : Self.partialGapSeconds
                if let wall { gap = max(gap, wall * Self.selfClockFactor) }
                let now = self.bufferLock.withLock { self.audioBuffer.count }
                nextPartialAt = now + Int(gap * Self.sampleRate)
            }
        }
    }

    /// Decodes everything since the last commit; returns the decode wall time (nil when nothing was decoded).
    private func decodePartial() async -> TimeInterval? {
        let samples = bufferLock.withLock { audioBuffer }
        guard !samples.isEmpty else { return nil }
        let start = Date()
        do {
            let transcript = try await engine.transcribe(samples)
            try Task.checkCancellation()
            bufferLock.withLock { lastPartialWords = transcript.words }
            let text = transcript.text.trimmingCharacters(in: .whitespacesAndNewlines)
            if !text.isEmpty {
                eventsContinuation?.yield(.partial(text: text))
            }
            let wall = Date().timeIntervalSince(start)
            partialDecodes.record(wall)
            return wall
        } catch is CancellationError {
            return nil
        } catch {
            logger.error("Phonon-2 partial decode failed: \(error, privacy: .public)")
            return nil
        }
    }

    /// Deta's progressive final: cut the hold near 28 s, decode that piece as committed text, keep the rest live.
    private func commitPiece() async -> Bool {
        let (piece, cutSample) = bufferLock.withLock { () -> ([Float], Int) in
            let cut = Self.cutSample(in: audioBuffer, words: lastPartialWords)
            return (Array(audioBuffer[..<cut]), cut)
        }
        do {
            let start = Date()
            let transcript = try await engine.transcribe(piece)
            try Task.checkCancellation()
            let text = transcript.text.trimmingCharacters(in: .whitespacesAndNewlines)
            bufferLock.withLock {
                audioBuffer.removeFirst(cutSample)
                committedSampleCount += cutSample
                if !text.isEmpty {
                    committedSegments.append(text)
                }
                lastPartialWords = []
            }
            eventsContinuation?.yield(.committed(text: text))
            let seconds = Double(cutSample) / Self.sampleRate
            logger.notice(
                "Phonon-2 committed \(seconds, format: .fixed(precision: 1))s in \(Date().timeIntervalSince(start), format: .fixed(precision: 3))s"
            )
            return true
        } catch is CancellationError {
            return false
        } catch {
            logger.error("Phonon-2 commit decode failed; switching to batch fallback: \(error, privacy: .public)")
            bufferLock.withLock {
                progressiveCommitFailed = true
                audioBuffer = []
                lastPartialWords = []
            }
            eventsContinuation?.yield(.error(error))
            return false
        }
    }

    /// The sample to cut at: the sentence end (. ? !) closest to 28 s with a pause after it, within 28 ± 6 s;
    /// otherwise the quietest 300 ms in that band. Never within 1 s of the live edge.
    private static func cutSample(in audio: [Float], words: [Word]) -> Int {
        let sr = sampleRate
        let lowSeconds = commitTargetSeconds - commitSearchSeconds
        let highSeconds = min(commitTargetSeconds + commitSearchSeconds, Double(audio.count) / sr - 1)
        var best: Double?
        // Word timings are on the padded clock; the engine pads 0.25 s before the audio.
        let padding = Double(PhononEngine.paddingSamples) / sr
        for index in words.indices.dropLast() {
            let word = words[index], next = words[index + 1]
            let end = word.end - padding
            guard let last = word.text.last(where: { $0.isLetter || $0.isNumber || ".!?".contains($0) }),
                ".!?".contains(last), end >= lowSeconds, end <= highSeconds, next.start - word.end >= 0.12
            else { continue }
            let candidate = (end + (next.start - padding)) / 2
            if best == nil || abs(candidate - commitTargetSeconds) < abs(best! - commitTargetSeconds) {
                best = candidate
            }
        }
        if let best { return Int(best * sr) }

        let low = Int(lowSeconds * sr), high = Int(highSeconds * sr)
        let window = Int(quietWindowSeconds * sr)
        guard high - low > window else { return Int(commitTargetSeconds * sr) }
        var energy: Float = 0
        for sample in audio[low..<(low + window)] { energy += sample * sample }
        var bestEnergy = energy
        var bestStart = low
        for start in (low + 1)...(high - window) {
            energy += audio[start + window - 1] * audio[start + window - 1] - audio[start - 1] * audio[start - 1]
            if energy < bestEnergy {
                bestEnergy = energy
                bestStart = start
            }
        }
        return bestStart + window / 2
    }
}

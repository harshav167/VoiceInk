import Foundation
import os

@MainActor
final class LFM3TranscriptionSession: TranscriptionSession {
    private let resolveContext: () async -> LFM3SessionContext
    private let onPartial: ((String) -> Void)?
    private let logger = Logger(subsystem: "com.prakashjoshipax.voiceink", category: "LFM3Session")
    private var continuation: AsyncStream<Data>.Continuation?
    private var worker: Task<String, Error>?
    private var cancelled = false
    private(set) var integratedEnhancement: IntegratedTranscriptionEnhancement?

    init(context: @escaping () async -> LFM3SessionContext, onPartial: ((String) -> Void)?) {
        resolveContext = context
        self.onPartial = onPartial
    }

    /// Backlog the decoder may fall behind by before the recording fails. Measured in audio, not
    /// chunks: the microphone delivers ~10 ms chunks, so a chunk count hides how far behind it is.
    private static let maximumBacklogSamples = 90 * 16_000

    func prepare(configuration: TranscriptionRuntimeConfiguration) async throws -> ((Data) -> Void)? {
        let channel = AsyncStream<Data>.makeStream(bufferingPolicy: .unbounded)
        let backlog = OSAllocatedUnfairLock(initialState: Backlog())
        continuation = channel.continuation
        worker = Task {
            try await consume(channel.stream, backlog: backlog, modelName: configuration.model.displayName)
        }
        let limit = Self.maximumBacklogSamples
        return { data in
            let samples = data.count / MemoryLayout<Int16>.size
            let overflowed = backlog.withLock { state -> Bool in
                state.queuedSamples += samples
                state.peakSamples = max(state.peakSamples, state.queuedSamples)
                if state.queuedSamples > limit { state.overflowed = true }
                return state.overflowed
            }
            if overflowed {
                channel.continuation.finish()
            } else {
                channel.continuation.yield(data)
            }
        }
    }

    private struct Backlog {
        var queuedSamples = 0
        var peakSamples = 0
        var overflowed = false
    }

    private func consume(
        _ stream: AsyncStream<Data>, backlog: OSAllocatedUnfairLock<Backlog>, modelName: String
    ) async throws -> String {
        let started = Date()
        var strategy = LFM3POCStrategy.baseline
        var audioSamples = 0
        var blockCount = 0
        var identifier: UUID?
        do {
            let context = await resolveContext()
            strategy = context.strategy
            try Task.checkCancellation()
            if context.strategy != .baseline {
                integratedEnhancement = .init(
                    modelName: modelName,
                    promptName: context.promptName ?? context.strategy.title,
                    instructions: context.instructions.isEmpty
                        ? context.systemMessage(live: context.strategy != .contextualReview)
                        : context.instructions)
            }
            logger.notice(
                """
                Native Omni session begin strategy=\(context.strategy.rawValue, privacy: .public) \
                screenshotPresent=\(context.screenshot != nil, privacy: .public) \
                contextChars=\(context.contextText.count, privacy: .public) \
                instructionChars=\(context.instructions.count, privacy: .public) \
                integrated=\(self.integratedEnhancement != nil, privacy: .public)
                """)
            let session = try await LFM3OmniEngine.shared.begin(context: context)
            identifier = session
            var text = ""
            var decodeSeconds: Double = 0
            var firstPartialSeconds: Double?
            var partialCount = 0
            for await data in stream {
                try Task.checkCancellation()
                if backlog.withLock({ $0.overflowed }) { throw LFM3SessionError.audioBufferOverflow }
                let samples = PCMAudioConverter.float32Samples(fromPCM16Data: data)
                audioSamples += samples.count
                let updates = try await LFM3OmniEngine.shared.append(samples, to: session)
                backlog.withLock { $0.queuedSamples -= samples.count }
                for update in updates {
                    text = update.text
                    blockCount += 1
                    if !update.text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
                        partialCount += 1
                        if firstPartialSeconds == nil {
                            firstPartialSeconds = Date().timeIntervalSince(started)
                            logger.notice(
                                """
                                Native Omni first partial seconds=\(firstPartialSeconds ?? 0, privacy: .public) \
                                textChars=\(update.text.count, privacy: .public) strategy=\(context.strategy.rawValue, privacy: .public)
                                """)
                        }
                    }
                    decodeSeconds += update.decodeSeconds
                    onPartial?(text)
                }
            }
            try Task.checkCancellation()
            if backlog.withLock({ $0.overflowed }) { throw LFM3SessionError.audioBufferOverflow }
            let stopped = Date()
            let livePreview = text
            let updates = try await LFM3OmniEngine.shared.append([], to: session, final: true)
            text = updates.last?.text ?? text
            let finalSource = updates.last?.fromReview == true ? "fullReview" : "live"
            let statistics = await LFM3OmniEngine.shared.liveStatistics(session)
            await LFM3OmniEngine.shared.end(session)
            let peakBacklog = Double(backlog.withLock { $0.peakSamples }) / 16_000
            logger.notice(
                """
                Native Omni session blocks=\(blockCount, privacy: .public) partials=\(partialCount, privacy: .public) \
                firstPartialSeconds=\(firstPartialSeconds ?? -1, privacy: .public) \
                livePreviewChars=\(livePreview.count, privacy: .public) finalChars=\(text.count, privacy: .public) \
                finalSource=\(finalSource, privacy: .public) strategy=\(context.strategy.rawValue, privacy: .public) \
                audioSeconds=\(Double(audioSamples) / 16_000, privacy: .public) \
                segments=\(statistics?.segments ?? 0, privacy: .public) \
                suppressedReplies=\(statistics?.suppressed ?? 0, privacy: .public) \
                peakBacklogSeconds=\(peakBacklog, privacy: .public) \
                liveDecodeSeconds=\(decodeSeconds, privacy: .public) \
                stopSeconds=\(Date().timeIntervalSince(stopped), privacy: .public) \
                totalSeconds=\(Date().timeIntervalSince(started), privacy: .public)
                """
            )
            return text
        } catch {
            if let identifier { await LFM3OmniEngine.shared.end(identifier) }
            if !(error is CancellationError) {
                let peakBacklog = Double(backlog.withLock { $0.peakSamples }) / 16_000
                logger.error(
                    """
                    Native Omni session failed strategy=\(strategy.rawValue, privacy: .public) \
                    error=\(error.localizedDescription, privacy: .public) \
                    audioSeconds=\(Double(audioSamples) / 16_000, privacy: .public) \
                    blocks=\(blockCount, privacy: .public) peakBacklogSeconds=\(peakBacklog, privacy: .public) \
                    totalSeconds=\(Date().timeIntervalSince(started), privacy: .public)
                    """)
            }
            throw error
        }
    }

    func transcribe(audioURL: URL) async throws -> String {
        guard !cancelled, let worker else { throw CancellationError() }
        // Finish the input, then drain every queued chunk and await the authoritative final decode.
        // Stopping during model/prefix preparation does NOT switch to a second batch model load.
        continuation?.finish()
        continuation = nil
        return try await worker.value
    }

    func cancel() {
        cancelled = true
        worker?.cancel()
        continuation?.finish()
        continuation = nil
    }
}

enum LFM3SessionError: LocalizedError {
    case audioBufferOverflow

    var errorDescription: String? {
        "LFM3 Omni could not keep up and its audio queue filled. The recording was not silently truncated."
    }
}

final class LFM3TranscriptionService: TranscriptionService {
    func transcribe(
        audioURL: URL, model: any TranscriptionModel,
        context: TranscriptionRequestContext
    ) async throws -> String {
        try await transcribe(audioURL: audioURL, model: model, context: context, lfm3Context: .baseline)
    }

    func transcribe(
        audioURL: URL, model: any TranscriptionModel,
        context: TranscriptionRequestContext,
        lfm3Context: LFM3SessionContext
    ) async throws -> String {
        let samples = try await AudioProcessor().processAudioToSamples(audioURL, normalize: false)
        return try await LFM3OmniEngine.shared.transcribe(samples: samples, context: lfm3Context)
    }
}

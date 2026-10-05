import FluidAudio
import Foundation
import TranscribeCpp
import os

/// True real-time streaming provider backed by the shared transcribe.cpp runtime.
///
/// Final transcript authority mirrors Handy's transcribe-cpp integration: after
/// `Stream.finalize()` the native stream's `text.full` is the source of truth
/// (committed is append-only; the finalized committed prefix holds the full
/// text), delivered via `finalizationEvents`. Live `.committed` / `.partial`
/// events only drive the preview. A finalize failure finishes the finalization
/// stream empty so the shared layer falls back to batch transcription.
final class TranscribeCppStreamingProvider: StreamingTranscriptionProvider, @unchecked Sendable {
    private let logger = Logger(subsystem: "com.prakashjoshipax.voiceink", category: "TranscribeCppStreaming")
    private let transcribeCppService: TranscribeCppTranscriptionService

    private var nativeModel: TranscribeCpp.Model?
    private var stream: TranscribeCpp.Stream?
    private let stateLock = NSLock()
    private var finalizationContinuation: AsyncStream<String>.Continuation?
    private var lastCommittedText: String = ""
    private var totalAudioDurationSeconds: Double = 0
    private var lastUpdateAudioSeconds: Double = 0
    private var speechDetected: Bool = false
    private var leadingSilenceSeconds: Double = 0

    private(set) var transcriptionEvents: AsyncStream<StreamingTranscriptionEvent>
    private var eventsContinuation: AsyncStream<StreamingTranscriptionEvent>.Continuation?

    /// Provider-confirmed authoritative full transcript, emitted once after
    /// `Stream.finalize()`. The shared layer uses it instead of accumulating
    /// committed-event diffs, and falls back to batch when it finishes empty.
    let finalizationEvents: AsyncStream<String>?

    init(transcribeCppService: TranscribeCppTranscriptionService) {
        self.transcribeCppService = transcribeCppService

        var continuation: AsyncStream<StreamingTranscriptionEvent>.Continuation!
        transcriptionEvents = AsyncStream { continuation = $0 }
        eventsContinuation = continuation
        var finalization: AsyncStream<String>.Continuation!
        finalizationEvents = AsyncStream { finalization = $0 }
        finalizationContinuation = finalization
    }

    deinit {
        eventsContinuation?.finish()
        finalizationContinuation?.finish()
    }

    func connect(model: any TranscriptionModel, language: String?) async throws {
        guard let transcribeCppModel = model as? TranscribeCppModel else {
            throw StreamingTranscriptionError.connectionFailed("Unsupported streaming model")
        }

        let loadedModel = try await transcribeCppService.getOrLoadModel(for: transcribeCppModel)
        let session = try loadedModel.session()

        let compatibleLanguage = TranscriptionLanguageSupport.validLanguageOrFallback(language, for: model)
        let selectedLanguage = compatibleLanguage == "auto" ? nil : compatibleLanguage
        let runOptions = RunOptions(timestamps: .none, language: selectedLanguage)
        let nativeStream = try session.stream(runOptions, StreamOptions(commitPolicy: .auto))

        stateLock.withLock {
            nativeModel = loadedModel
            stream = nativeStream
            lastCommittedText = ""
            totalAudioDurationSeconds = 0
            lastUpdateAudioSeconds = 0
            speechDetected = false
            leadingSilenceSeconds = 0
        }

        eventsContinuation?.yield(.sessionStarted)
        logger.notice("TranscribeCpp streaming started for \(model.displayName, privacy: .public)")
    }

    func sendAudioChunk(_ data: Data) async throws {
        guard let stream = stateLock.withLock({ self.stream }) else {
            throw StreamingTranscriptionError.notConnected
        }

        let samples = PCMAudioConverter.float32Samples(fromPCM16Data: data)
        guard !samples.isEmpty else { return }

        let chunkDuration = Double(samples.count) / 16_000.0

        // Prevent feeding excessive initial silence to Voxtral while waiting for speech.
        // If silence continues beyond 200ms before speech starts, skip excess silence
        // so Voxtral doesn't hallucinate web priors and emit an unrecoverable early EOS.
        let shouldFeed: Bool = stateLock.withLock {
            totalAudioDurationSeconds += chunkDuration
            if !speechDetected {
                var sumSquares: Float = 0
                for s in samples { sumSquares += s * s }
                let rms = sqrt(sumSquares / Float(samples.count))
                if rms > 0.015 {
                    speechDetected = true
                    return true
                } else {
                    leadingSilenceSeconds += chunkDuration
                    return leadingSilenceSeconds <= 0.20
                }
            }
            return true
        }

        guard shouldFeed else { return }

        let update = try stream.feed(samples)
        guard update.committedChanged || update.tentativeChanged else { return }

        stateLock.withLock {
            lastUpdateAudioSeconds = totalAudioDurationSeconds
        }

        emitEvents(for: stream.text)
    }

    func commit() async throws {
        guard let stream = stateLock.withLock({ self.stream }) else {
            // No active stream: finish empty so the shared layer batch-falls back.
            finalizationContinuation?.finish()
            return
        }

        do {
            _ = try stream.finalize()
            let text = stream.text
            emitEvents(for: text)

            let (totalDuration, lastUpdate, isStalled) = stateLock.withLock { () -> (Double, Double, Bool) in
                let duration = totalAudioDurationSeconds
                let last = lastUpdateAudioSeconds
                let chars = text.full.trimmingCharacters(in: .whitespacesAndNewlines).count
                // Early EOS detection: if audio was over 4s long, but produced suspiciously little text
                // or went completely dead for > 75% of the session after an early burst, the stream
                // got locked by an early EOS token in transcribe.cpp. Trigger batch fallback to
                // re-transcribe cleanly offline.
                let charsPerSec = duration > 0 ? Double(chars) / duration : 0
                let deadTimeRatio = duration > 0 ? (duration - last) / duration : 0
                let stalled = duration > 4.0 && (chars < 25 || charsPerSec < 1.0 || deadTimeRatio > 0.75)
                return (duration, last, stalled)
            }

            if isStalled {
                logger.warning(
                    "Voxtral stream stalled prematurely (chars=\(text.full.count, privacy: .public) duration=\(totalDuration, format: .fixed(precision: 1), privacy: .public)s lastUpdate=\(lastUpdate, format: .fixed(precision: 1), privacy: .public)s); triggering batch fallback"
                )
                // Finishing without yielding causes StreamingTranscriptionService to run batch fallback
            } else {
                finalizationContinuation?.yield(text.full)
            }
        } catch {
            // Finalize failed: finish empty so the shared layer batch-falls back
            // instead of surfacing a hard error with a truncated transcript.
            logger.error("Stream finalize failed, requesting batch fallback: \(error.localizedDescription, privacy: .public)")
        }
        finalizationContinuation?.finish()
    }

    func disconnect() async {
        stateLock.withLock {
            stream?.reset()
            stream = nil
            nativeModel = nil
        }
        eventsContinuation?.finish()
        finalizationContinuation?.finish()
        logger.notice("TranscribeCpp streaming disconnected")
    }

    // MARK: - Private

    /// Emits the incremental committed suffix and the cumulative display text.
    private func emitEvents(for text: StreamText) {
        let newlyCommitted = String(text.committed.dropFirst(lastCommittedText.count))
        if !newlyCommitted.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
            lastCommittedText = text.committed
            eventsContinuation?.yield(.committed(text: newlyCommitted))
        }

        let display = text.display.trimmingCharacters(in: .whitespacesAndNewlines)
        if !display.isEmpty {
            eventsContinuation?.yield(.partial(text: display))
        }
    }
}

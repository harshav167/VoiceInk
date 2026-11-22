import Foundation
import os

/// Batch transcription with Phonon-2 through the shared `PhononEngine`.
final class PhononTranscriptionService: TranscriptionService {
    private let logger = Logger(subsystem: "com.prakashjoshipax.voiceink", category: "PhononTranscriptionService")

    @MainActor
    private var bundleDirectory: URL { PhononModelManager.shared.bundleDirectory }

    /// Opens the engine (compiling on first run) so a later dictation never pays the load.
    func loadModel() async throws {
        let bundle = await bundleDirectory
        try await PhononEngine.shared.open(bundle: bundle)
    }

    func transcribe(audioURL: URL, model: any TranscriptionModel, context: TranscriptionRequestContext) async throws
        -> String
    {
        try await loadModel()
        let start = Date()
        let transcript = try await PhononEngine.shared.transcribe(url: audioURL)
        let elapsed = Date().timeIntervalSince(start)
        logger.notice(
            "Phonon-2 transcribed \(transcript.audioSeconds, format: .fixed(precision: 1))s in \(elapsed, format: .fixed(precision: 3))s windows=\(transcript.windows)"
        )
        return transcript.text.trimmingCharacters(in: .whitespacesAndNewlines)
    }
    // The engine stays resident between dictations (as Deta keeps it); only deleting the model unloads it.
}

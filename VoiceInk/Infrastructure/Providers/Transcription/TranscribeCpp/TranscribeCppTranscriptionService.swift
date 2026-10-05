import Foundation
import TranscribeCpp

/// Routes catalog-backed models through the shared transcribe.cpp runtime.
final class TranscribeCppTranscriptionService: TranscriptionService, @unchecked Sendable {
    private let offlineService = OfflineTranscribeCppService()

    func loadModel(for model: TranscribeCppModel) async throws {
        guard TranscribeCppModelCatalog.artifact(for: model.name) != nil else {
            throw unsupportedModelError(model.name)
        }
        try await offlineService.loadModel(for: model)
    }

    /// Loads or reuses the shared native model instance for streaming sessions.
    func getOrLoadModel(for model: TranscribeCppModel) async throws -> Model {
        guard let artifact = TranscribeCppModelCatalog.artifact(for: model.name) else {
            throw unsupportedModelError(model.name)
        }
        return try await offlineService.getOrLoadModel(for: model, artifact: artifact)
    }

    func transcribe(
        audioURL: URL,
        model: any TranscriptionModel,
        context: TranscriptionRequestContext
    ) async throws -> String {
        guard let transcribeCppModel = model as? TranscribeCppModel else {
            throw NSError(
                domain: "TranscribeCppTranscriptionService",
                code: -1,
                userInfo: [NSLocalizedDescriptionKey: "Unsupported transcription model"]
            )
        }

        guard TranscribeCppModelCatalog.artifact(for: transcribeCppModel.name) != nil else {
            throw unsupportedModelError(transcribeCppModel.name)
        }
        return try await offlineService.transcribe(
            audioURL: audioURL,
            model: transcribeCppModel,
            context: context
        )
    }

    func cleanup() {
        offlineService.cleanup()
    }

    private func unsupportedModelError(_ modelName: String) -> NSError {
        NSError(
            domain: "TranscribeCppTranscriptionService",
            code: -2,
            userInfo: [NSLocalizedDescriptionKey: "No transcribe.cpp service is registered for \(modelName)"]
        )
    }
}

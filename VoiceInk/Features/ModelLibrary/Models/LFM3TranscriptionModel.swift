import Foundation

struct LFM3TranscriptionModel: TranscriptionModel, Sendable {
    let id = UUID()
    let name = "lfm3-omni-2b-mlx-8bit"
    let displayName = "LFM3 Omni 2B"
    let description =
        "Liquid AI's on-device multimodal model for live English transcription with optional screen-aware cleanup."
    let provider = ModelProvider.lfm3
    let supportsStreaming = true
    let isMultilingualModel = false
    let supportedLanguages = ["en": "English"]
}

import Foundation

enum LFM3ModelError: LocalizedError {
    case unsupportedConfiguration
    case unavailableBuild
    case missingModel
    case invalidImage
    case contextFull
    case invalidStream
    case generationLimit

    var errorDescription: String? {
        switch self {
        case .unsupportedConfiguration:
            "This experimental adapter requires the selected LFM3 Omni 2B 8-bit architecture."
        case .unavailableBuild: "The native LFM3 Omni POC is available only in local development builds."
        case .missingModel: "Download LFM3 Omni 2B before starting transcription."
        case .invalidImage: "The initial screen image could not be prepared for LFM3 Omni."
        case .contextFull: "The LFM3 Omni context is full. Reduce the initial context."
        case .invalidStream: "The LFM3 Omni stream is no longer active."
        case .generationLimit: "LFM3 Omni did not complete its response within the decode budget."
        }
    }
}

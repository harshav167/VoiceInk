import CoreML
import Foundation

/// Where a FluidAudio Core ML model runs. The Parakeet preprocessor always stays on the CPU (FluidAudio pins it), and
/// Core ML may run layers an accelerator does not support on the CPU under every choice.
enum FluidAudioComputePipeline: Equatable {
    case cpuAndNeuralEngine
    case gpuAndNeuralEngine
    case gpu

    var computeUnits: MLComputeUnits {
        switch self {
        case .cpuAndNeuralEngine:
            return .cpuAndNeuralEngine
        case .gpuAndNeuralEngine:
            return .all
        case .gpu:
            return .cpuAndGPU
        }
    }

    var usesGPU: Bool { self != .cpuAndNeuralEngine }
}

/// The user's pipeline choice for Parakeet-family models and Cohere Transcribe.
/// Nemotron and Parakeet Unified load through their own FluidAudio managers and are not affected.
enum FluidAudioComputePipelineSetting: String, CaseIterable, Identifiable {
    case automatic
    case cpuAndNeuralEngine
    case gpuAndNeuralEngine
    case gpu

    static let userDefaultsKey = "FluidAudioComputePipeline"
    static let defaultValue: FluidAudioComputePipelineSetting = .automatic

    static var current: FluidAudioComputePipelineSetting {
        UserDefaults.standard.string(forKey: userDefaultsKey).flatMap(Self.init(rawValue:)) ?? defaultValue
    }

    var id: String { rawValue }

    var title: String {
        switch self {
        case .automatic:
            return String(localized: "Automatic")
        case .cpuAndNeuralEngine:
            return String(localized: "CPU + Neural Engine")
        case .gpuAndNeuralEngine:
            return String(localized: "GPU + Neural Engine")
        case .gpu:
            return String(localized: "GPU")
        }
    }

    /// Automatic picks what measured fastest on an M4 Max (2026-10-07, identical transcripts on every pipeline):
    /// Parakeet-family models on the CPU + Neural Engine (GPU-only was ~35% slower), Cohere Transcribe with the GPU
    /// allowed (~3x faster than CPU + Neural Engine once warm).
    func pipeline(isCohere: Bool) -> FluidAudioComputePipeline {
        switch self {
        case .automatic:
            return isCohere ? .gpuAndNeuralEngine : .cpuAndNeuralEngine
        case .cpuAndNeuralEngine:
            return .cpuAndNeuralEngine
        case .gpuAndNeuralEngine:
            return .gpuAndNeuralEngine
        case .gpu:
            return .gpu
        }
    }
}

import Foundation
import MLX
import MLXLMCommon
import MLXNN

struct LFM3OmniConfiguration: Decodable {
    let modelType: String
    let textConfig: LFM3TextConfiguration
    let quantization: BaseConfiguration.Quantization

    enum CodingKeys: String, CodingKey {
        case modelType = "model_type"
        case textConfig = "text_config"
        case quantization
    }

    static func load(from directory: URL) throws -> LFM3OmniConfiguration {
        let data = try Data(contentsOf: directory.appendingPathComponent("config.json"))
        let config = try JSONDecoder().decode(Self.self, from: data)
        let text = config.textConfig
        guard config.modelType == "lfm3_omni", text.hiddenSize == 2048, text.vocabSize == 128_000,
            text.layerTypes.count == 24, text.numAttentionHeads == 16, text.numKeyValueHeads == 4,
            text.headDim == 128, text.convLCache == 7, text.slidingWindow == 256,
            text.layerTypes.allSatisfy({
                ["conv", "full_attention", "sliding_attention"].contains($0)
            }),
            text.layerTypes.filter({ $0 != "conv" }).allSatisfy({ text.ropeParameters[$0] != nil })
        else {
            throw LFM3ModelError.unsupportedConfiguration
        }
        let raw = try JSONSerialization.jsonObject(with: data) as? [String: Any]
        let audio = raw?["audio_encoder_config"] as? [String: Any]
        let vision = raw?["vision_config"] as? [String: Any]
        guard audio?["hidden_size"] as? Int == 768, audio?["num_mel_bins"] as? Int == 128,
            audio?["conv_channels"] as? Int == 256, audio?["sliding_window"] as? Int == 32,
            vision?["hidden_size"] as? Int == 1024, vision?["patch_size"] as? Int == 16,
            vision?["temporal_patch_size"] as? Int == 2, vision?["spatial_merge_size"] as? Int == 2
        else {
            throw LFM3ModelError.unsupportedConfiguration
        }
        return config
    }
}

/// Text output only: speech synthesis's codec and depth decoder are deliberately not loaded.
final class LFM3OmniModel: Module, BaseLanguageModel {
    @ModuleInfo(key: "language_model") var language: LFM3LanguageModel
    @ModuleInfo(key: "audio_tower") var audio: LFM3AudioEncoder
    @ModuleInfo(key: "audio_projector") var audioProjector: LFM3AudioProjector
    @ModuleInfo(key: "vision_tower") var vision: LFM3VisionEncoder
    @ModuleInfo(key: "multi_modal_projector") var visionProjector: LFM3VisionProjector

    var excludedCheckpointNamespaces: [String] { ["depth_decoder", "codebook_embedding"] }

    init(_ configuration: LFM3OmniConfiguration) {
        _language.wrappedValue = LFM3LanguageModel(configuration.textConfig)
        _audio.wrappedValue = LFM3AudioEncoder()
        _audioProjector.wrappedValue = LFM3AudioProjector()
        _vision.wrappedValue = LFM3VisionEncoder()
        _visionProjector.wrappedValue = LFM3VisionProjector()
    }

    func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        var result = [String: MLXArray]()
        result.reserveCapacity(weights.count)
        for (name, tensor) in weights {
            var normalized = tensor
            if name.hasSuffix("conv.conv.weight"), tensor.dim(1) == 1 {
                normalized = tensor.transposed(0, 2, 1)
            } else if name.hasSuffix("spatial_conv.weight"), tensor.dim(1) == 1 {
                normalized = tensor.transposed(0, 2, 3, 1)
            } else if name.hasPrefix("audio_tower.downsample.") {
                if tensor.ndim == 4, tensor.dim(2) == tensor.dim(3) {
                    normalized = tensor.transposed(0, 2, 3, 1)
                } else if tensor.ndim == 3 {
                    normalized = tensor[0..., 0..., 0]
                }
            }
            result[name] = normalized
        }
        return result
    }
}

// LFM3 architecture adapted from Hugging Face / mlx-vlm (Apache-2.0).
// Reference: Blaizzy/mlx-vlm@4693cd0, models/lfm3/language.py.
import Foundation
import MLX
import MLXLMCommon
import MLXNN

struct LFM3RotaryConfiguration: Decodable {
    let ropeTheta: Float
    let partialRotaryFactor: Float?
    enum CodingKeys: String, CodingKey {
        case ropeTheta = "rope_theta"
        case partialRotaryFactor = "partial_rotary_factor"
    }
}
struct LFM3TextConfiguration: Decodable {

    let hiddenSize: Int
    let intermediateSize: Int
    let numAttentionHeads: Int
    let numKeyValueHeads: Int
    let headDim: Int
    let layerTypes: [String]
    let normEps: Float
    let convLCache: Int
    let slidingWindow: Int
    let vocabSize: Int
    let useSandwichNorm: Bool
    let ropeParameters: [String: LFM3RotaryConfiguration]

    enum CodingKeys: String, CodingKey {
        case hiddenSize = "hidden_size"
        case intermediateSize = "intermediate_size"
        case numAttentionHeads = "num_attention_heads"
        case numKeyValueHeads = "num_key_value_heads"
        case headDim = "head_dim"
        case layerTypes = "layer_types"
        case normEps = "norm_eps"
        case convLCache = "conv_L_cache"
        case slidingWindow = "sliding_window"
        case vocabSize = "vocab_size"
        case useSandwichNorm = "use_sandwich_norm"
        case ropeParameters = "rope_parameters"
    }
}

final class LFM3ShortConvolution: Module {
    @ModuleInfo(key: "in_proj") var input: Linear
    @ModuleInfo(key: "out_proj") var output: Linear
    let conv: Conv1d
    let historyCount: Int
    let width: Int

    init(width: Int, kernel: Int) {
        self.width = width
        historyCount = kernel - 1
        _input.wrappedValue = Linear(width, width * 3, bias: false)
        _output.wrappedValue = Linear(width, width, bias: false)
        conv = Conv1d(
            inputChannels: width, outputChannels: width, kernelSize: kernel,
            groups: width, bias: false)
    }

    func callAsFunction(_ hidden: MLXArray, cache: MambaCache?) -> MLXArray {
        let parts = input(hidden).split(parts: 3, axis: -1)
        let previous =
            cache?[0] ?? MLXArray.zeros([hidden.dim(0), historyCount, width], dtype: hidden.dtype)
        let joined = concatenated([previous, parts[0] * parts[2]], axis: 1)
        cache?[0] = contiguous(joined[0..., (joined.dim(1) - historyCount)..., 0...])
        cache?.advance(hidden.dim(1))
        return output(parts[1] * conv(joined))
    }
}

final class LFM3FeedForward: Module {
    @ModuleInfo(key: "w1") var gateProjection: Linear
    @ModuleInfo(key: "w2") var outputProjection: Linear
    @ModuleInfo(key: "w3") var upProjection: Linear

    init(width: Int, intermediate: Int) {
        _gateProjection.wrappedValue = Linear(width, intermediate, bias: false)
        _outputProjection.wrappedValue = Linear(intermediate, width, bias: false)
        _upProjection.wrappedValue = Linear(width, intermediate, bias: false)
    }

    func callAsFunction(_ hidden: MLXArray) -> MLXArray {
        outputProjection(silu(gateProjection(hidden)) * upProjection(hidden))
    }
}

final class LFM3Attention: Module {
    @ModuleInfo(key: "q_proj") var query: Linear
    @ModuleInfo(key: "k_proj") var key: Linear
    @ModuleInfo(key: "v_proj") var value: Linear
    @ModuleInfo(key: "o_proj") var output: Linear
    @ModuleInfo(key: "q_norm") var queryNorm: RMSNorm
    @ModuleInfo(key: "k_norm") var keyNorm: RMSNorm
    @ModuleInfo(key: "gate_proj") var gate: Linear
    let heads: Int
    let kvHeads: Int
    let headSize: Int
    let theta: Float
    let angles: Int

    init(_ config: LFM3TextConfiguration, type: String) {
        heads = config.numAttentionHeads
        kvHeads = config.numKeyValueHeads
        headSize = config.headDim
        let rotary = config.ropeParameters[type]!
        theta = rotary.ropeTheta
        angles = Int(Float(headSize) * (rotary.partialRotaryFactor ?? 1)) / 2
        _query.wrappedValue = Linear(config.hiddenSize, heads * headSize, bias: false)
        _key.wrappedValue = Linear(config.hiddenSize, kvHeads * headSize, bias: false)
        _value.wrappedValue = Linear(config.hiddenSize, kvHeads * headSize, bias: false)
        _output.wrappedValue = Linear(heads * headSize, config.hiddenSize, bias: false)
        _gate.wrappedValue = Linear(config.hiddenSize, heads, bias: false)
        _queryNorm.wrappedValue = RMSNorm(dimensions: headSize, eps: config.normEps)
        _keyNorm.wrappedValue = RMSNorm(dimensions: headSize, eps: config.normEps)
    }

    private func rotate(_ tensor: MLXArray, offset: Int) -> MLXArray {
        // Partial RoPE rotates paired coordinates in BOTH head halves, not a contiguous prefix.
        let frequency = pow(
            MLXArray(theta), -MLXArray(0..<angles).asType(.float32) * (2 / Float(headSize)))
        let padded = concatenated([frequency, MLXArray.zeros([headSize / 2 - angles])])
        let positions = MLXArray(offset..<(offset + tensor.dim(2))).asType(.float32)
        let phase = positions.expandedDimensions(axis: -1) * padded
        let doubled = concatenated([phase, phase], axis: -1)
        let halves = tensor.split(parts: 2, axis: -1)
        return tensor * cos(doubled).asType(tensor.dtype) + concatenated(
            [-halves[1], halves[0]], axis: -1)
            * sin(doubled).asType(tensor.dtype)
    }

    func callAsFunction(
        _ hidden: MLXArray, mask: MLXFast.ScaledDotProductAttentionMaskMode,
        cache: KVCache?
    ) -> MLXArray {
        let batch = hidden.dim(0)
        let length = hidden.dim(1)
        var queries = queryNorm(query(hidden).reshaped(batch, length, heads, headSize)).transposed(
            0, 2, 1, 3)
        var keys = keyNorm(key(hidden).reshaped(batch, length, kvHeads, headSize)).transposed(
            0, 2, 1, 3)
        let values = value(hidden).reshaped(batch, length, kvHeads, headSize).transposed(0, 2, 1, 3)
        queries = rotate(queries, offset: cache?.offset ?? 0)
        keys = rotate(keys, offset: cache?.offset ?? 0)
        let attended = attentionWithCacheUpdate(
            queries: queries, keys: keys, values: values,
            cache: cache, scale: pow(Float(headSize), -0.5), mask: mask
        ).transposed(0, 2, 1, 3)
        let gated = attended * sigmoid(gate(hidden)).expandedDimensions(axis: -1)
        return output(gated.reshaped(batch, length, heads * headSize))
    }
}

final class LFM3DecoderLayer: Module {
    @ModuleInfo(key: "self_attn") var attention: LFM3Attention?
    let conv: LFM3ShortConvolution?
    @ModuleInfo(key: "feed_forward") var feedForward: LFM3FeedForward
    @ModuleInfo(key: "operator_norm") var operatorNorm: RMSNorm
    @ModuleInfo(key: "ffn_norm") var feedForwardNorm: RMSNorm
    @ModuleInfo(key: "post_operator_norm") var postOperatorNorm: RMSNorm?
    @ModuleInfo(key: "post_ffn_norm") var postFeedForwardNorm: RMSNorm?
    let type: String

    init(_ config: LFM3TextConfiguration, index: Int) {
        type = config.layerTypes[index]
        conv =
            type == "conv"
            ? LFM3ShortConvolution(width: config.hiddenSize, kernel: config.convLCache) : nil
        _attention.wrappedValue = type == "conv" ? nil : LFM3Attention(config, type: type)
        _feedForward.wrappedValue = LFM3FeedForward(
            width: config.hiddenSize, intermediate: config.intermediateSize)
        _operatorNorm.wrappedValue = RMSNorm(dimensions: config.hiddenSize, eps: config.normEps)
        _feedForwardNorm.wrappedValue = RMSNorm(dimensions: config.hiddenSize, eps: config.normEps)
        if config.useSandwichNorm {
            _postOperatorNorm.wrappedValue = RMSNorm(
                dimensions: config.hiddenSize, eps: config.normEps)
            _postFeedForwardNorm.wrappedValue = RMSNorm(
                dimensions: config.hiddenSize, eps: config.normEps)
        }
    }

    func callAsFunction(
        _ hidden: MLXArray, mask: MLXFast.ScaledDotProductAttentionMaskMode,
        cache: KVCache?
    ) -> MLXArray {
        let normalized = operatorNorm(hidden)
        let operated =
            attention?(normalized, mask: mask, cache: cache)
            ?? conv!(normalized, cache: cache as? MambaCache)
        let residual = hidden + (postOperatorNorm?(operated) ?? operated)
        let feed = feedForward(feedForwardNorm(residual))
        return residual + (postFeedForwardNorm?(feed) ?? feed)
    }
}

final class LFM3Backbone: Module {
    @ModuleInfo(key: "embed_tokens") var embedding: Embedding
    @ModuleInfo(key: "embedding_norm") var norm: RMSNorm
    let layers: [LFM3DecoderLayer]
    let config: LFM3TextConfiguration

    init(_ config: LFM3TextConfiguration) {
        self.config = config
        _embedding.wrappedValue = Embedding(
            embeddingCount: config.vocabSize, dimensions: config.hiddenSize)
        _norm.wrappedValue = RMSNorm(dimensions: config.hiddenSize, eps: config.normEps)
        layers = config.layerTypes.indices.map { LFM3DecoderLayer(config, index: $0) }
    }

    func newCache() -> [KVCache] {
        layers.map { layer in
            if layer.type == "conv" { return MambaCache() }
            if layer.type == "sliding_attention" {
                return RotatingKVCache(maxSize: config.slidingWindow)
            }
            return KVCacheSimple()
        }
    }

    func hiddenStates(_ embeddings: MLXArray, cache: [KVCache]) -> MLXArray {
        var hidden = embeddings
        for (index, layer) in layers.enumerated() {
            let window = layer.type == "sliding_attention" ? config.slidingWindow : nil
            let mask = createAttentionMask(h: hidden, cache: cache[index], windowSize: window)
            hidden = layer(hidden, mask: mask, cache: cache[index])
        }
        return norm(hidden)
    }
}

final class LFM3LanguageModel: Module {
    private static let prefillChunkSize = 2_048
    let model: LFM3Backbone

    init(_ config: LFM3TextConfiguration) { model = LFM3Backbone(config) }

    func logits(embeddings: MLXArray, cache: [KVCache]) -> MLXArray {
        let total = embeddings.dim(1)
        let chunkSize = Self.prefillChunkSize
        if total <= chunkSize {
            let hidden = model.hiddenStates(embeddings, cache: cache)
            return model.embedding.asLinear(hidden[0..., (hidden.dim(1) - 1)..<hidden.dim(1), 0...])
        }
        var offset = 0
        var lastHidden: MLXArray!
        while offset < total {
            let end = min(total, offset + chunkSize)
            let hidden = model.hiddenStates(embeddings[0..., offset..<end, 0...], cache: cache)
            if end < total {
                eval(cache.flatMap(\.state))
            } else {
                lastHidden = hidden[0..., (hidden.dim(1) - 1)..<hidden.dim(1), 0...]
            }
            offset = end
        }
        return model.embedding.asLinear(lastHidden)
    }

    func logitsCancellable(embeddings: MLXArray, cache: [KVCache]) throws -> MLXArray {
        try Task.checkCancellation()
        let total = embeddings.dim(1)
        let chunkSize = Self.prefillChunkSize
        if total <= chunkSize {
            let hidden = model.hiddenStates(embeddings, cache: cache)
            return model.embedding.asLinear(hidden[0..., (hidden.dim(1) - 1)..<hidden.dim(1), 0...])
        }
        var offset = 0
        var lastHidden: MLXArray!
        while offset < total {
            try Task.checkCancellation()
            let end = min(total, offset + chunkSize)
            let hidden = model.hiddenStates(embeddings[0..., offset..<end, 0...], cache: cache)
            if end < total {
                eval(cache.flatMap(\.state))
            } else {
                lastHidden = hidden[0..., (hidden.dim(1) - 1)..<hidden.dim(1), 0...]
            }
            offset = end
        }
        return model.embedding.asLinear(lastHidden)
    }
}

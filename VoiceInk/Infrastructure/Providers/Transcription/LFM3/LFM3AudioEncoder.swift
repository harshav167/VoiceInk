import Foundation
// Audio architecture: Hugging Face LFM3 Omni / mlx-vlm@4693cd0 (Apache-2.0).
import MLX
import MLXLMCommon
import MLXNN

final class LFM3AudioAttention: Module {
    @ModuleInfo(key: "q_proj") var query: Linear
    @ModuleInfo(key: "k_proj") var key: Linear
    @ModuleInfo(key: "v_proj") var value: Linear
    @ModuleInfo(key: "out_proj") var output: Linear
    @ModuleInfo(key: "q_layernorm") var queryNorm: RMSNorm
    @ModuleInfo(key: "k_layernorm") var keyNorm: RMSNorm
    let rope = RoPE(dimensions: 96, traditional: false, base: 1_000_000)

    override init() {
        _query.wrappedValue = Linear(768, 768, bias: false)
        _key.wrappedValue = Linear(768, 384, bias: false)
        _value.wrappedValue = Linear(768, 384, bias: false)
        _output.wrappedValue = Linear(768, 768, bias: false)
        _queryNorm.wrappedValue = RMSNorm(dimensions: 96, eps: 1e-5)
        _keyNorm.wrappedValue = RMSNorm(dimensions: 96, eps: 1e-5)
    }

    func callAsFunction(_ hidden: MLXArray, cache: KVCache) -> MLXArray {
        let count = hidden.dim(1)
        let queries = queryNorm(query(hidden).reshaped(1, count, 8, 96)).transposed(0, 2, 1, 3)
        let keys = keyNorm(key(hidden).reshaped(1, count, 4, 96)).transposed(0, 2, 1, 3)
        let values = value(hidden).reshaped(1, count, 4, 96).transposed(0, 2, 1, 3)
        let mask = createAttentionMask(h: hidden, cache: cache, windowSize: 32)
        let attended = attentionWithCacheUpdate(
            queries: rope(queries, offset: cache.offset),
            keys: rope(keys, offset: cache.offset), values: values, cache: cache,
            scale: pow(Float(96), -0.5), mask: mask)
        return output(attended.transposed(0, 2, 1, 3).reshaped(1, count, 768))
    }
}

final class LFM3AudioLayer: Module {
    @ModuleInfo(key: "self_attn") var attention: LFM3AudioAttention?
    let conv: LFM3ShortConvolution?
    @ModuleInfo(key: "feed_forward") var feedForward: LFM3FeedForward
    @ModuleInfo(key: "operator_norm") var operatorNorm: RMSNorm
    @ModuleInfo(key: "ffn_norm") var feedForwardNorm: RMSNorm

    init(attention: Bool) {
        _attention.wrappedValue = attention ? LFM3AudioAttention() : nil
        conv = attention ? nil : LFM3ShortConvolution(width: 768, kernel: 3)
        // LFM2's auto-adjusted FFN: ceil((2 * 2560 / 3) / 256) * 256 = 1792.
        _feedForward.wrappedValue = LFM3FeedForward(width: 768, intermediate: 1792)
        _operatorNorm.wrappedValue = RMSNorm(dimensions: 768, eps: 1e-5)
        _feedForwardNorm.wrappedValue = RMSNorm(dimensions: 768, eps: 1e-5)
    }

    func callAsFunction(_ hidden: MLXArray, cache: KVCache) -> MLXArray {
        let normalized = operatorNorm(hidden)
        let residual =
            hidden
            + (attention?(normalized, cache: cache)
                ?? conv!(normalized, cache: cache as? MambaCache))
        return residual + feedForward(feedForwardNorm(residual))
    }
}

final class LFM3AudioTrunk: Module {
    let layers: [LFM3AudioLayer]
    @ModuleInfo(key: "embedding_norm") var norm: RMSNorm
    private static let attentionIndices: Set<Int> = [2, 5, 8, 11, 14, 15, 16]

    override init() {
        layers = (0..<17).map { LFM3AudioLayer(attention: Self.attentionIndices.contains($0)) }
        _norm.wrappedValue = RMSNorm(dimensions: 768, eps: 1e-5)
    }

    func newCache() -> [KVCache] {
        (0..<17).map {
            Self.attentionIndices.contains($0) ? RotatingKVCache(maxSize: 32) : MambaCache()
        }
    }

    func callAsFunction(_ embeddings: MLXArray, cache: [KVCache]) -> MLXArray {
        var hidden = embeddings
        for (index, layer) in layers.enumerated() { hidden = layer(hidden, cache: cache[index]) }
        return norm(hidden)
    }
}

final class LFM3AudioDownsampler: Module {
    @ModuleInfo(key: "conv_in") var first: Conv2d
    @ModuleInfo(key: "conv_mid") var middle: Conv2d
    @ModuleInfo(key: "lin_mid") var middleLinear: Conv2d
    @ModuleInfo(key: "conv_end") var last: Conv2d
    @ModuleInfo(key: "lin_end") var lastLinear: Conv2d
    @ModuleInfo(key: "lin_out") var output: Linear

    override init() {
        _first.wrappedValue = Conv2d(
            inputChannels: 1, outputChannels: 256, kernelSize: 3, stride: 2, padding: 1)
        _middle.wrappedValue = Conv2d(
            inputChannels: 256, outputChannels: 256, kernelSize: 3,
            stride: 2, padding: 1, groups: 256)
        _middleLinear.wrappedValue = Conv2d(inputChannels: 256, outputChannels: 256, kernelSize: 1)
        _last.wrappedValue = Conv2d(
            inputChannels: 256, outputChannels: 256, kernelSize: 3,
            stride: 2, padding: 1, groups: 256)
        _lastLinear.wrappedValue = Conv2d(inputChannels: 256, outputChannels: 256, kernelSize: 1)
        _output.wrappedValue = Linear(4096, 768)
    }

    func callAsFunction(_ features: MLXArray) -> MLXArray {
        var hidden = features.expandedDimensions(axis: -1).asType(first.weight.dtype)
        hidden = silu(first(hidden).asType(.float32)).asType(hidden.dtype)
        hidden = silu(middleLinear(middle(hidden)).asType(.float32)).asType(hidden.dtype)
        hidden = silu(lastLinear(last(hidden)).asType(.float32)).asType(hidden.dtype)
        return output(hidden.transposed(0, 2, 3, 1).reshaped(1, hidden.dim(2), 4096))
    }
}

struct LFM3AudioState {
    var features: MLXArray?
    var frames = 0
    var offset = 0
    var tokens = 0
    let cache: [KVCache]
}

final class LFM3AudioEncoder: Module {
    private static let maxChunkFrames = 960
    let downsample = LFM3AudioDownsampler()
    let lfm = LFM3AudioTrunk()

    func newState() -> LFM3AudioState { LFM3AudioState(cache: lfm.newCache()) }

    func stream(_ features: MLXArray, state: inout LFM3AudioState, final: Bool) -> MLXArray {
        let totalFrames = features.dim(2)
        if totalFrames <= Self.maxChunkFrames {
            return streamChunk(features, state: &state, final: final)
        }
        var parts: [MLXArray] = []
        var cursor = 0
        while cursor < totalFrames {
            let end = min(totalFrames, cursor + Self.maxChunkFrames)
            let isLast = end == totalFrames
            let chunk = streamChunk(features[0..., 0..., cursor..<end], state: &state, final: final && isLast)
            if chunk.dim(1) > 0 {
                eval([chunk] + state.cache.flatMap(\.state))
                parts.append(chunk)
            } else {
                eval(state.cache.flatMap(\.state))
            }
            cursor = end
        }
        return parts.isEmpty ? MLXArray.zeros([1, 0, 768]) : concatenated(parts, axis: 1)
    }

    func streamCancellable(_ features: MLXArray, state: inout LFM3AudioState, final: Bool) throws -> MLXArray {
        try Task.checkCancellation()
        let totalFrames = features.dim(2)
        if totalFrames <= Self.maxChunkFrames {
            return streamChunk(features, state: &state, final: final)
        }
        var parts: [MLXArray] = []
        var cursor = 0
        while cursor < totalFrames {
            try Task.checkCancellation()
            let end = min(totalFrames, cursor + Self.maxChunkFrames)
            let isLast = end == totalFrames
            let chunk = streamChunk(features[0..., 0..., cursor..<end], state: &state, final: final && isLast)
            if chunk.dim(1) > 0 {
                eval([chunk] + state.cache.flatMap(\.state))
                parts.append(chunk)
            } else {
                eval(state.cache.flatMap(\.state))
            }
            cursor = end
        }
        return parts.isEmpty ? MLXArray.zeros([1, 0, 768]) : concatenated(parts, axis: 1)
    }

    private func streamChunk(_ features: MLXArray, state: inout LFM3AudioState, final: Bool) -> MLXArray {
        let joined = state.features.map { concatenated([$0, features], axis: -1) } ?? features
        let frames = state.frames + features.dim(2)
        let tokens = (frames + (final ? 7 : 0)) / 8
        let count = tokens - state.tokens
        var encoded = MLXArray.zeros([1, 0, 768])
        if count > 0 {
            let start = state.tokens - state.offset / 8
            let downsampled = downsample(joined)
            encoded = lfm(downsampled[0..., start..<(start + count), 0...], cache: state.cache)
        }
        let offset = max(0, (tokens - 1) * 8)
        state.features = final ? nil : contiguous(joined[0..., 0..., (offset - state.offset)...])
        state.frames = frames
        state.offset = offset
        state.tokens = tokens
        return encoded
    }
}

final class LFM3AudioProjector: Module {
    @ModuleInfo(key: "layer_norm") var norm: LayerNorm
    @ModuleInfo(key: "linear_1") var first: Linear
    @ModuleInfo(key: "linear_2") var second: Linear

    override init() {
        _norm.wrappedValue = LayerNorm(dimensions: 768, eps: 1e-5)
        _first.wrappedValue = Linear(768, 2048)
        _second.wrappedValue = Linear(2048, 2048)
    }

    func callAsFunction(_ features: MLXArray) -> MLXArray {
        let hidden = first(norm(features))
        return second(gelu(hidden.asType(.float32)).asType(hidden.dtype))
    }
}

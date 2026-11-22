// LFM3-VL vision architecture adapted from Hugging Face / mlx-vlm@4693cd0 (Apache-2.0).
import MLX
import MLXNN

final class LFM3VisionAttention: Module {
    @ModuleInfo(key: "q_proj") var query: Linear
    @ModuleInfo(key: "k_proj") var key: Linear
    @ModuleInfo(key: "v_proj") var value: Linear
    @ModuleInfo(key: "out_proj") var output: Linear
    @ModuleInfo(key: "q_layernorm") var queryNorm: RMSNorm
    @ModuleInfo(key: "k_layernorm") var keyNorm: RMSNorm

    override init() {
        _query.wrappedValue = Linear(1024, 1024, bias: false)
        _key.wrappedValue = Linear(1024, 512, bias: false)
        _value.wrappedValue = Linear(1024, 512, bias: false)
        _output.wrappedValue = Linear(1024, 1024, bias: false)
        _queryNorm.wrappedValue = RMSNorm(dimensions: 64, eps: 1e-5)
        _keyNorm.wrappedValue = RMSNorm(dimensions: 64, eps: 1e-5)
    }

    func callAsFunction(_ hidden: MLXArray, cosine: MLXArray, sine: MLXArray) -> MLXArray {
        let count = hidden.dim(0)
        func rotate(_ tensor: MLXArray) -> MLXArray {
            let full = tensor.asType(.float32)
            let halves = full.split(parts: 2, axis: -1)
            return
                (full * cosine.expandedDimensions(axis: 1)
                + concatenated([-halves[1], halves[0]], axis: -1) * sine.expandedDimensions(axis: 1))
                .asType(tensor.dtype)
        }
        let queries = rotate(queryNorm(query(hidden).reshaped(count, 16, 64)))
        let keys = rotate(keyNorm(key(hidden).reshaped(count, 8, 64)))
        let values = value(hidden).reshaped(count, 8, 64)
        let attended = MLXFast.scaledDotProductAttention(
            queries: queries.transposed(1, 0, 2).expandedDimensions(axis: 0),
            keys: keys.transposed(1, 0, 2).expandedDimensions(axis: 0),
            values: values.transposed(1, 0, 2).expandedDimensions(axis: 0), scale: 0.125,
            mask: .none)
        return output(attended[0].transposed(1, 0, 2).reshaped(count, 1024))
    }
}

final class LFM3VisionConvolution: Module {
    @ModuleInfo(key: "in_proj") var input: Linear
    @ModuleInfo(key: "out_proj") var output: Linear
    @ModuleInfo(key: "spatial_conv") var spatial: Conv2d

    override init() {
        _input.wrappedValue = Linear(1024, 3072, bias: false)
        _output.wrappedValue = Linear(1024, 1024, bias: false)
        _spatial.wrappedValue = Conv2d(
            inputChannels: 1024, outputChannels: 1024,
            kernelSize: 3, padding: 1, groups: 1024, bias: false)
    }

    func callAsFunction(_ hidden: MLXArray, height: Int, width: Int) -> MLXArray {
        let parts = input(hidden).split(parts: 3, axis: -1)
        let grid = (parts[0] * parts[2]).reshaped(height / 2, width / 2, 2, 2, 1024)
            .transposed(0, 2, 1, 3, 4).reshaped(1, height, width, 1024)
        let convolved = spatial(grid).reshaped(height / 2, 2, width / 2, 2, 1024)
            .transposed(0, 2, 1, 3, 4).reshaped(height * width, 1024)
        return output(parts[1] * convolved)
    }
}

final class LFM3VisionLayer: Module {
    @ModuleInfo(key: "self_attn") var attention: LFM3VisionAttention?
    let conv: LFM3VisionConvolution?
    @ModuleInfo(key: "feed_forward") var feedForward: LFM3FeedForward
    @ModuleInfo(key: "operator_norm") var operatorNorm: RMSNorm
    @ModuleInfo(key: "ffn_norm") var feedForwardNorm: RMSNorm

    init(attention: Bool) {
        _attention.wrappedValue = attention ? LFM3VisionAttention() : nil
        conv = attention ? nil : LFM3VisionConvolution()
        _feedForward.wrappedValue = LFM3FeedForward(width: 1024, intermediate: 2560)
        _operatorNorm.wrappedValue = RMSNorm(dimensions: 1024, eps: 1e-5)
        _feedForwardNorm.wrappedValue = RMSNorm(dimensions: 1024, eps: 1e-5)
    }

    func callAsFunction(
        _ hidden: MLXArray, height: Int, width: Int,
        cosine: MLXArray, sine: MLXArray
    ) -> MLXArray {
        let normalized = operatorNorm(hidden)
        let operated =
            attention?(normalized, cosine: cosine, sine: sine)
            ?? conv!(normalized, height: height, width: width)
        let residual = hidden + operated
        return residual + feedForward(feedForwardNorm(residual))
    }
}

final class LFM3VisionEncoder: Module {
    let patchifier = Linear(1536, 1024, bias: false)
    @ModuleInfo(key: "embedding_norm") var norm: RMSNorm
    let layers: [LFM3VisionLayer]

    override init() {
        _norm.wrappedValue = RMSNorm(dimensions: 1024, eps: 1e-5)
        layers = (0..<14).map { LFM3VisionLayer(attention: $0 >= 2 && $0.isMultiple(of: 2)) }
    }

    func callAsFunction(_ patches: MLXArray, height: Int, width: Int) -> MLXArray {
        var coordinates = [Int32]()
        coordinates.reserveCapacity(height * width * 2)
        for blockRow in 0..<(height / 2) {
            for blockColumn in 0..<(width / 2) {
                for row in 0..<2 {
                    for column in 0..<2 {
                        coordinates.append(Int32(blockRow * 2 + row))
                        coordinates.append(Int32(blockColumn * 2 + column))
                    }
                }
            }
        }
        let positions = MLXArray(coordinates, [height * width, 2, 1]).asType(.float32)
        let inverse = pow(
            MLXArray(Float(1_000_000)),
            -MLXArray(stride(from: 0, to: 32, by: 2)).asType(.float32) / 32)
        let phase = (positions * inverse).reshaped(height * width, 32)
        let doubled = concatenated([phase, phase], axis: -1)
        let cosine = cos(doubled)
        let sine = sin(doubled)
        var hidden = norm(patchifier(patches.asType(norm.weight.dtype)))
        for layer in layers {
            hidden = layer(hidden, height: height, width: width, cosine: cosine, sine: sine)
        }
        return hidden
    }
}

final class LFM3VisionProjector: Module {
    @ModuleInfo(key: "layer_norm") var norm: LayerNorm
    @ModuleInfo(key: "linear_1") var first: Linear
    @ModuleInfo(key: "linear_2") var second: Linear

    override init() {
        _norm.wrappedValue = LayerNorm(dimensions: 1024, eps: 1e-6)
        _first.wrappedValue = Linear(4096, 4096)
        _second.wrappedValue = Linear(4096, 2048)
    }

    func callAsFunction(_ hidden: MLXArray) -> MLXArray {
        let projected = first(norm(hidden).reshaped(-1, 4096))
        return second(gelu(projected.asType(.float32)).asType(projected.dtype))
    }
}

// Matches LiquidAI's feature_extraction_lfm3_omni.py (Apache-2.0):
// completed 400-sample Hann windows, 160-sample hop, 128 Slaney bins.
import Foundation
import MLX

struct LFM3MelFeatures {
    private var pending: [Float] = []
    private var sampleCount = 0
    private var frameCount = 0
    private(set) var finished = false
    private let window: MLXArray
    private let filters: MLXArray

    init() {
        window = MLXArray(
            (0..<400).map { Float(0.5 - 0.5 * cos(2 * Double.pi * Double($0) / 400)) })
        filters = Self.makeFilters()
    }

    mutating func append(_ samples: [Float], final: Bool = false) throws -> MLXArray {
        guard !finished else { throw LFM3FeatureError.alreadyFinalized }
        pending.append(contentsOf: samples)
        sampleCount += samples.count
        let count: Int
        if final {
            if sampleCount == 0 {
                count = 0
            } else {
                let totalFrames = 1 + max(0, (sampleCount - 400 + 159) / 160)
                count = totalFrames - frameCount
                if count > 0 {
                    let required = (count - 1) * 160 + 400
                    pending.append(
                        contentsOf: repeatElement(0, count: max(0, required - pending.count)))
                }
            }
        } else {
            count = pending.count < 400 ? 0 : 1 + (pending.count - 400) / 160
        }
        var features = MLXArray.zeros([1, 128, 0])
        if count > 0 {
            let indices = MLXArray(0..<count).expandedDimensions(axis: -1) * 160 + MLXArray(0..<400)
            let frames = MLXArray(pending)[indices] * window
            let spectrum = MLXFFT.rfft(frames, axis: -1)
            let power = abs(spectrum).square()
            let melPower = matmul(power, filters)
            let normalized = (maximum(log10(melPower), -12) + 4) / 4
            features = normalized.transposed().expandedDimensions(axis: 0)
        }
        if final {
            pending.removeAll(keepingCapacity: false)
        } else if count > 0 {
            pending.removeFirst(count * 160)
        }
        frameCount += count
        finished = final
        return features
    }

    private static func makeFilters() -> MLXArray {
        // Slaney's piecewise linear/logarithmic mel scale and area normalization.
        func toMel(_ frequency: Double) -> Double {
            frequency < 1000
                ? frequency / (200.0 / 3) : 15 + log(frequency / 1000) / (log(6.4) / 27)
        }
        func toHz(_ mel: Double) -> Double {
            mel < 15 ? mel * (200.0 / 3) : 1000 * exp((mel - 15) * (log(6.4) / 27))
        }
        let maximumMel = toMel(8000)
        let edges = (0..<130).map { toHz(Double($0) * maximumMel / 129) }
        var coefficients = [Float](repeating: 0, count: 201 * 128)
        for frequencyBin in 0..<201 {
            let frequency = Double(frequencyBin) * 40
            for melBin in 0..<128 {
                let rising = (frequency - edges[melBin]) / (edges[melBin + 1] - edges[melBin])
                let falling =
                    (edges[melBin + 2] - frequency) / (edges[melBin + 2] - edges[melBin + 1])
                let area = 2 / (edges[melBin + 2] - edges[melBin])
                coefficients[frequencyBin * 128 + melBin] = Float(
                    max(0, min(rising, falling)) * area)
            }
        }
        return MLXArray(coefficients, [201, 128])
    }
}

enum LFM3FeatureError: LocalizedError {
    case alreadyFinalized

    var errorDescription: String? { "The LFM3 audio stream has already been finalized." }
}

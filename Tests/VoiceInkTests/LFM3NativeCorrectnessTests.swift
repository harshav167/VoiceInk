#if LFM3_HOSTLESS_TESTS
import CoreGraphics
import Foundation
import ImageIO
import MLX
import MLXLMCommon
import Testing
import UniformTypeIdentifiers

@Suite(.serialized)
struct LFM3NativeCorrectnessTests {
    private func configuration(layers: [String]) -> LFM3TextConfiguration {
        LFM3TextConfiguration(
            hiddenSize: 32, intermediateSize: 64, numAttentionHeads: 4,
            numKeyValueHeads: 2, headDim: 8, layerTypes: layers, normEps: 1e-5,
            convLCache: 7, slidingWindow: 256, vocabSize: 64, useSandwichNorm: true,
            ropeParameters: [
                "sliding_attention": LFM3RotaryConfiguration(ropeTheta: 10_000, partialRotaryFactor: 0.5),
                "full_attention": LFM3RotaryConfiguration(ropeTheta: 10_000, partialRotaryFactor: 0.5)
            ])
    }

    private func input(length: Int) -> MLXArray {
        let values = (0..<(length * 32)).map { index in
            Float(sin(Double(index) * 0.017) + 0.3 * cos(Double(index) * 0.071))
        }
        return MLXArray(values, [1, length, 32])
    }

    private func materialize(_ output: MLXArray, cache: [KVCache]) {
        eval([output] + cache.flatMap(\.state))
    }

    @Test func longPrefillMatchesSegmentedPrefillAndContinuation() {
        MLXRandom.seed(131)
        let model = LFM3LanguageModel(configuration(layers: ["conv", "sliding_attention", "full_attention"]))
        let embeddings = input(length: 326)
        let wholeCache = model.model.newCache()
        let whole = model.model.hiddenStates(embeddings[0..., 0..<321, 0...], cache: wholeCache)
        materialize(whole, cache: wholeCache)
        let wholeNext = model.logits(embeddings: embeddings[0..., 321..<326, 0...], cache: wholeCache)
        materialize(wholeNext, cache: wholeCache)

        let segmentedCache = model.model.newCache()
        var pieces: [MLXArray] = []
        for range in [0..<63, 63..<176, 176..<256, 256..<257, 257..<321] {
            let output = model.model.hiddenStates(embeddings[0..., range, 0...], cache: segmentedCache)
            materialize(output, cache: segmentedCache)
            pieces.append(output)
        }
        let segmented = concatenated(pieces, axis: 1)
        let segmentedNext = model.logits(embeddings: embeddings[0..., 321..<326, 0...], cache: segmentedCache)
        materialize(segmentedNext, cache: segmentedCache)
        let tokenCache = model.model.newCache()
        let initial = model.model.hiddenStates(embeddings[0..., 0..<256, 0...], cache: tokenCache)
        materialize(initial, cache: tokenCache)
        var tokenPieces = [initial]
        for index in 256..<321 {
            let output = model.model.hiddenStates(embeddings[0..., index..<(index + 1), 0...], cache: tokenCache)
            materialize(output, cache: tokenCache)
            tokenPieces.append(output)
        }
        let tokenNext = model.logits(embeddings: embeddings[0..., 321..<326, 0...], cache: tokenCache)
        materialize(tokenNext, cache: tokenCache)
        let prefixCache = model.model.newCache()
        let prefixLogits = model.logits(embeddings: embeddings[0..., 0..<321, 0...], cache: prefixCache)
        materialize(prefixLogits, cache: prefixCache)
        let segmentedLogits = model.model.embedding.asLinear(segmented[0..., 320..<321, 0...])
        let tokenLogits = model.model.embedding.asLinear(tokenPieces[tokenPieces.count - 1])
        #expect(whole.shape == [1, 321, 32])
        #expect(segmented.shape == whole.shape)
        #expect(allClose(whole, segmented, rtol: 1e-4, atol: 1e-5).item(Bool.self))
        #expect(allClose(wholeNext, segmentedNext, rtol: 1e-4, atol: 1e-5).item(Bool.self))
        #expect(allClose(whole, concatenated(tokenPieces, axis: 1), rtol: 1e-4, atol: 1e-5).item(Bool.self))
        #expect(allClose(wholeNext, tokenNext, rtol: 1e-4, atol: 1e-5).item(Bool.self))
        #expect(allClose(prefixLogits, segmentedLogits, rtol: 1e-4, atol: 1e-5).item(Bool.self))
        #expect(allClose(prefixLogits, tokenLogits, rtol: 1e-4, atol: 1e-5).item(Bool.self))
    }

    @Test func copiedCachesDoNotChangeTheOriginalStream() {
        MLXRandom.seed(133)
        let model = LFM3LanguageModel(configuration(layers: ["conv", "sliding_attention", "full_attention"]))
        let embeddings = input(length: 600)
        let original = model.model.newCache()
        let control = model.model.newCache()
        let prefix = embeddings[0..., 0..<321, 0...]
        let originalPrefix = model.logits(embeddings: prefix, cache: original)
        materialize(originalPrefix, cache: original)
        let controlPrefix = model.logits(embeddings: prefix, cache: control)
        materialize(controlPrefix, cache: control)
        #expect(allClose(originalPrefix, controlPrefix, rtol: 1e-4, atol: 1e-5).item(Bool.self))

        let fork = original.map { $0.copy() }
        let controlFork = control.map { $0.copy() }
        let suffix = embeddings[0..., 321..<344, 0...]
        let forkResult = model.logits(embeddings: suffix, cache: fork)
        materialize(forkResult, cache: fork)
        let controlForkResult = model.logits(embeddings: suffix, cache: controlFork)
        materialize(controlForkResult, cache: controlFork)
        #expect(allClose(forkResult, controlForkResult, rtol: 1e-4, atol: 1e-5).item(Bool.self))

        let secondFork = original.map { $0.copy() }
        let divergent = model.logits(embeddings: embeddings[0..., 344..<367, 0...], cache: secondFork)
        materialize(divergent, cache: secondFork)
        #expect(!allClose(forkResult, divergent, rtol: 1e-4, atol: 1e-5).item(Bool.self))

        for index in 321..<581 {
            let token = embeddings[0..., index..<(index + 1), 0...]
            let actual = model.logits(embeddings: token, cache: original)
            materialize(actual, cache: original)
            let expected = model.logits(embeddings: token, cache: control)
            materialize(expected, cache: control)
            let matches = allClose(actual, expected, rtol: 1e-4, atol: 1e-5).item(Bool.self)
            #expect(matches, "Original cache changed after forking at token \(index)")
            if !matches { break }
        }
    }

    @Test func convolutionCacheContinuityAndFreshReset() {
        MLXRandom.seed(132)
        let model = LFM3LanguageModel(configuration(layers: ["conv", "conv"]))
        let embeddings = input(length: 24)
        let referenceCache = model.model.newCache()
        let reference = model.model.hiddenStates(embeddings, cache: referenceCache)
        materialize(reference, cache: referenceCache)

        let streamCache = model.model.newCache()
        var pieces: [MLXArray] = []
        for range in [0..<1, 1..<5, 5..<6, 6..<13, 13..<24] {
            let output = model.model.hiddenStates(embeddings[0..., range, 0...], cache: streamCache)
            materialize(output, cache: streamCache)
            pieces.append(output)
        }
        let stream = concatenated(pieces, axis: 1)
        let resetCache = model.model.newCache()
        let reset = model.model.hiddenStates(embeddings, cache: resetCache)
        materialize(reset, cache: resetCache)
        #expect(allClose(reference, stream, rtol: 1e-4, atol: 1e-5).item(Bool.self))
        #expect(allClose(reference, reset, rtol: 1e-4, atol: 1e-5).item(Bool.self))
    }

    private func checkMelChunks(_ samples: [Float]) throws {
        var complete = LFM3MelFeatures()
        let expected = try complete.append(samples, final: true)
        var chunked = LFM3MelFeatures()
        var frames: [MLXArray] = []
        let boundaries = [0, 1, 159, 399, 400, 561, 1_600, 1_601, 3_333, samples.count]
        for index in 0..<(boundaries.count - 1) {
            let range = boundaries[index]..<boundaries[index + 1]
            let part = try chunked.append(Array(samples[range]), final: index == boundaries.count - 2)
            if part.dim(2) > 0 { frames.append(part) }
        }
        let actual = concatenated(frames, axis: 2)
        #expect(expected.shape == [1, 128, 30])
        #expect(actual.shape == expected.shape)
        let expectedValues = expected.asArray(Float.self)
        let actualValues = actual.asArray(Float.self)
        let worst = expectedValues.indices.max {
            abs(expectedValues[$0] - actualValues[$0]) < abs(expectedValues[$1] - actualValues[$1])
        }!
        let location = "mel \(worst / 30), frame \(worst % 30)"
        let values = "\(expectedValues[worst]) vs \(actualValues[worst])"
        #expect(
            allClose(expected, actual, rtol: 1e-5, atol: 1e-5).item(Bool.self),
            "Largest difference at \(location): \(values)")
        #expect(throws: LFM3FeatureError.self) { try chunked.append([]) }
    }

    @Test func chunkedMelPreservesToneBandsAndFrameOrder() throws {
        let samples = (0..<4_957).map { index in
            let frequency = index < 2_400 ? 437.0 : 1_600.0
            return Float(0.4 * sin(Double(index) * 2 * .pi * frequency / 16_000))
        }
        var complete = LFM3MelFeatures()
        let whole = try complete.append(samples, final: true)
        var chunked = LFM3MelFeatures()
        var parts: [MLXArray] = []
        let boundaries = [0, 1, 159, 399, 400, 561, 1_600, 1_601, 3_333, samples.count]
        for index in 0..<(boundaries.count - 1) {
            let range = boundaries[index]..<boundaries[index + 1]
            let part = try chunked.append(Array(samples[range]), final: index == boundaries.count - 2)
            if part.dim(2) > 0 { parts.append(part) }
        }
        let stream = concatenated(parts, axis: 2)
        #expect(whole.shape == [1, 128, 30])
        #expect(stream.shape == whole.shape)
        for output in [whole, stream] {
            let values = output.asArray(Float.self)
            for frame in Array(0..<13) + Array(15..<30) {
                let peak = (0..<128).max { values[$0 * 30 + frame] < values[$1 * 30 + frame] }!
                // The Slaney filter bank centers nearest 437 Hz and 1600 Hz are bins 18 and 61.
                #expect(peak == (frame < 13 ? 18 : 61), "Unexpected tone band at frame \(frame): \(peak)")
            }
        }
    }

    @Test func broadbandAndSilentMelFramesDoNotDependOnChunkBoundaries() throws {
        var state: UInt32 = 0x12345678
        let samples = (0..<4_957).map { _ in
            state = 1_664_525 &* state &+ 1_013_904_223
            return (Float(state >> 8) / Float(1 << 24) - 0.5) * 0.8
        }
        try checkMelChunks(samples)
        try checkMelChunks(Array(repeating: 0, count: samples.count))
    }

    @Test func imagePatchesPreserveSpatialChannelAndTemporalOrder() throws {
        let pixels: [UInt8] = (0..<(256 * 256)).flatMap { index in
            index % 256 < 128 ? [255, 0, 0, 255] : [0, 0, 255, 255]
        }
        let provider = try #require(CGDataProvider(data: Data(pixels) as CFData))
        let image = try #require(CGImage(
            width: 256, height: 256, bitsPerComponent: 8, bitsPerPixel: 32, bytesPerRow: 1024,
            space: CGColorSpaceCreateDeviceRGB(),
            bitmapInfo: CGBitmapInfo(rawValue: CGImageAlphaInfo.noneSkipLast.rawValue),
            provider: provider, decode: nil, shouldInterpolate: false, intent: .defaultIntent))
        let data = NSMutableData()
        let destination = try #require(
            CGImageDestinationCreateWithData(data, UTType.jpeg.identifier as CFString, 1, nil))
        CGImageDestinationAddImage(
            destination, image, [kCGImageDestinationLossyCompressionQuality: 1.0] as CFDictionary)
        #expect(CGImageDestinationFinalize(destination))
        let prepared = try LFM3ImageFeatures(jpeg: data as Data)
        #expect(prepared.height == 16)
        #expect(prepared.width == 16)
        #expect(prepared.patches.shape == [256, 1536])
        let values = prepared.patches.asArray(Float.self)
        for patch in [0, 1, 2, 3, 16, 17, 18, 19] {
            let red = patch < 16
            for temporal in 0..<2 {
                let base = patch * 1536 + temporal * 256 + 8 * 16 + 8
                #expect(abs(values[base] - (red ? 1 : -1)) < 0.05)
                #expect(abs(values[base + 512] + 1) < 0.05)
                #expect(abs(values[base + 1024] - (red ? -1 : 1)) < 0.05)
            }
        }
    }
}
#endif

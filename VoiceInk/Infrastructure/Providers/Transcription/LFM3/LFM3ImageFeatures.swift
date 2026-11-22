import CoreImage
import Foundation
import MLX
import MLXVLM

struct LFM3ImageFeatures {
    let patches: MLXArray
    let height: Int
    let width: Int

    init(jpeg: Data) throws {
        guard let image = CIImage(data: jpeg), image.extent.width > 0, image.extent.height > 0
        else {
            throw LFM3ModelError.invalidImage
        }
        let originalHeight = Double(image.extent.height)
        let originalWidth = Double(image.extent.width)
        guard max(originalHeight, originalWidth) / min(originalHeight, originalWidth) <= 200 else {
            throw LFM3ModelError.invalidImage
        }
        // processor_config.json: aspect preserving, 32-pixel multiples, 64K–2M pixel area.
        var targetHeight = Int((originalHeight / 32).rounded(.toNearestOrEven)) * 32
        var targetWidth = Int((originalWidth / 32).rounded(.toNearestOrEven)) * 32
        if targetHeight * targetWidth > 2_097_152 {
            let ratio = sqrt(originalHeight * originalWidth / 2_097_152)
            targetHeight = max(32, Int(floor(originalHeight / ratio / 32)) * 32)
            targetWidth = max(32, Int(floor(originalWidth / ratio / 32)) * 32)
        } else if targetHeight * targetWidth < 65_536 {
            let ratio = sqrt(65_536 / (originalHeight * originalWidth))
            targetHeight = Int(ceil(originalHeight * ratio / 32)) * 32
            targetWidth = Int(ceil(originalWidth * ratio / 32)) * 32
        }
        height = targetHeight / 16
        width = targetWidth / 16
        let srgb = MediaProcessing.inSRGBToneCurveSpace(image)
        let resized = MediaProcessing.resampleBicubic(
            srgb,
            to: CGSize(width: targetWidth, height: targetHeight))
        let pixels = round(clip(MediaProcessing.asMLXArray(resized), min: 0, max: 1) * 255) / 255
        let normalized = (pixels - 0.5) / 0.5
        // Block-major 2×2 spatial patches, then C,T,H,W. A still image repeats both temporal samples.
        let spatial = normalized.reshaped(1, 3, height / 2, 2, 16, width / 2, 2, 16)
            .transposed(0, 2, 5, 3, 6, 1, 4, 7)
        let temporal = repeated(spatial.expandedDimensions(axis: 6), count: 2, axis: 6)
        patches = temporal.reshaped(height * width, 1536)
    }
}

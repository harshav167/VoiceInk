import Darwin
import Foundation
import MLX
import MLXLMCommon

// The caller owns and serializes the model and caches (LFM3OmniEngine).
enum LFM3GenerationPhase {
    static let contextLimit = 32_768
    case liveChunk, liveFinal, offline, visionCheck
    var limit: Int {
        switch self {
        case .liveChunk: 48
        case .liveFinal: 96
        case .offline: 8_192
        case .visionCheck: 128
        }
    }
}

/// One generation call: the reply plus how long the prompt prefill and the token loop took.
struct LFM3GenerationResult {
    let tokens: [Int]
    let closed: Bool
    let prefillTokens: Int
    let prefillSeconds: Double
    let decodeSeconds: Double
}
/// `budget` caps generated tokens. With a budget the caller decides what an unclosed reply means
/// (`closed == false`); without one, non-live phases throw `generationLimit` at `phase.limit`.
func generateLFM3Tokens(
    input: MLXArray, cache: [KVCache], model: LFM3OmniModel,
    tokenizer: any MLXLMCommon.Tokenizer,
    phase: LFM3GenerationPhase, budget: Int? = nil
) throws -> LFM3GenerationResult {
    try Task.checkCancellation()
    guard (cache.map(\.offset).max() ?? 0) + input.dim(1) < LFM3GenerationPhase.contextLimit else {
        throw LFM3ModelError.contextFull
    }
    let prefillStarted = Date()
    var logits = try model.language.logitsCancellable(embeddings: input, cache: cache)
    eval(logits)
    let prefillSeconds = Date().timeIntervalSince(prefillStarted)
    let decodeStarted = Date()
    func result(_ tokens: [Int], closed: Bool) -> LFM3GenerationResult {
        .init(
            tokens: tokens, closed: closed, prefillTokens: input.dim(1),
            prefillSeconds: prefillSeconds, decodeSeconds: Date().timeIntervalSince(decodeStarted))
    }
    let placeholders = [125029, 125030, 125031, 125032]
    let textOnly = ["<|audio_start|>", "<|mixed_start|>"].compactMap(tokenizer.convertTokenToId)
    let blockedNames = ["<|im_end|>", "<|mixed_end|>", "<|endoftext|>"]
    let blocked =
        placeholders + textOnly
        + (phase == .liveChunk ? blockedNames.compactMap(tokenizer.convertTokenToId) : [])
    let stopping = Set(
        ["<|text_end|>", "<|im_end|>", "<|endoftext|>", "<|im_start|>"].compactMap(
            tokenizer.convertTokenToId))
    var tokens: [Int] = []
    for _ in 0..<(budget ?? phase.limit) {
        try Task.checkCancellation()
        guard (cache.map(\.offset).max() ?? 0) < LFM3GenerationPhase.contextLimit else {
            throw LFM3ModelError.contextFull
        }
        logits[0, 0, MLXArray(blocked)] = MLXArray(-Float.infinity)
        let token = argMax(logits[0, 0]).item(Int.self)
        if token == 125014 || (phase != .liveChunk && stopping.contains(token)) {
            return result(tokens, closed: true)
        }
        tokens.append(token)
        let next = model.language.model.embedding(MLXArray([token]).reshaped(1, 1))
        logits = model.language.logits(embeddings: next, cache: cache)
    }
    if phase != .liveChunk, budget == nil { throw LFM3ModelError.generationLimit }
    return result(tokens, closed: false)
}

/// Per-recording totals, logged once when the live session ends.
struct LFM3LiveMetrics {
    var prefixTokens = 0
    var prefixSeconds = 0.0
    var visionSeconds = 0.0
    var encoderTokens = 0
    var encoderSeconds = 0.0
    var audioPrefillTokens = 0
    var audioPrefillSeconds = 0.0
    var replyPrefillTokens = 0
    var replyPrefillSeconds = 0.0
    var decodeTokens = 0
    var decodeSeconds = 0.0
    var previews = 0
    var previewSeconds = 0.0
    var previewMaxSeconds = 0.0

    mutating func add(_ generation: LFM3GenerationResult) {
        replyPrefillTokens += generation.prefillTokens
        replyPrefillSeconds += generation.prefillSeconds
        decodeTokens += generation.tokens.count
        decodeSeconds += generation.decodeSeconds
    }

    mutating func addPreview(seconds: Double) {
        previews += 1
        previewSeconds += seconds
        previewMaxSeconds = max(previewMaxSeconds, seconds)
    }

    var summary: String {
        let averagePreview = previews > 0 ? previewSeconds / Double(previews) : 0
        return """
            prefixTokens=\(prefixTokens) prefixPrefillTokPerSec=\(lfm3Rate(prefixTokens, prefixSeconds)) \
            visionSeconds=\(lfm3Fixed(visionSeconds)) \
            encoderTokPerSec=\(lfm3Rate(encoderTokens, encoderSeconds)) encoderSeconds=\(lfm3Fixed(encoderSeconds)) \
            audioPrefillTokPerSec=\(lfm3Rate(audioPrefillTokens, audioPrefillSeconds)) \
            replyPrefillTokPerSec=\(lfm3Rate(replyPrefillTokens, replyPrefillSeconds)) \
            decodeTokens=\(decodeTokens) decodeTokPerSec=\(lfm3Rate(decodeTokens, decodeSeconds)) \
            previews=\(previews) previewAvgSeconds=\(lfm3Fixed(averagePreview)) \
            previewMaxSeconds=\(lfm3Fixed(previewMaxSeconds))
            """
    }
}

/// MLX allocator state plus the process footprint. On Apple silicon MLX buffers live in unified
/// memory, so `mlxActive` is the model's GPU-resident memory.
enum LFM3MemoryProbe {
    static func resetPeak() { Memory.peakMemory = 0 }

    static var summary: String {
        let megabyte = 1_048_576
        let footprint = physicalFootprint.map { String($0 / megabyte) } ?? "n/a"
        return """
            mlxActiveMB=\(Memory.activeMemory / megabyte) mlxCacheMB=\(Memory.cacheMemory / megabyte) \
            mlxPeakMB=\(Memory.peakMemory / megabyte) footprintMB=\(footprint)
            """
    }

    /// The figure Activity Monitor reports as the process's Memory.
    private static var physicalFootprint: Int? {
        var info = task_vm_info_data_t()
        var count = mach_msg_type_number_t(
            MemoryLayout<task_vm_info_data_t>.size / MemoryLayout<natural_t>.size)
        let status = withUnsafeMutablePointer(to: &info) { pointer in
            pointer.withMemoryRebound(to: integer_t.self, capacity: Int(count)) {
                task_info(mach_task_self_, task_flavor_t(TASK_VM_INFO), $0, &count)
            }
        }
        return status == KERN_SUCCESS ? Int(info.phys_footprint) : nil
    }
}

func lfm3Rate(_ tokens: Int, _ seconds: Double) -> String {
    seconds > 0 ? String(format: "%.0f", Double(tokens) / seconds) : "n/a"
}

func lfm3Fixed(_ value: Double) -> String { String(format: "%.2f", value) }

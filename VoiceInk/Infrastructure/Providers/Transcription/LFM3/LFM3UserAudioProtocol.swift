import Foundation

/// Omni receives audio inside the user turn. Provisional replies close a copied input cache.
///
/// Cleanup instructions live in the system turn, the same layout as the offline prompt. The user
/// turn holds only reference data, the optional screenshot and then the audio, so the last text the
/// model reads before speech is never the instruction block it could copy into its reply.
enum LFM3UserAudioProtocol {
    enum Part: Equatable {
        case text(String)
        case image
    }

    static func sanitizeReferenceText(_ text: String) -> String {
        text
            .replacingOccurrences(of: "<|", with: "< |")
            .replacingOccurrences(of: "|>", with: "| >")
    }

    static func prefix(context: LFM3SessionContext, hasImage: Bool) -> [Part] {
        let safeContext = sanitizeReferenceText(context.contextText)
        var system: String
        if let custom = context.customSystemPrompt {
            system = custom
        } else {
            system = "Perform ASR in English."
            if !context.instructions.isEmpty { system += "\n" + context.instructions }
        }
        if context.systemReference {
            if !safeContext.isEmpty {
                system += "\n<REFERENCE_DATA>\n\(safeContext)\n</REFERENCE_DATA>"
            }
            var parts: [Part] = [.text("<|startoftext|><|im_start|>system\n" + system)]
            if hasImage {
                parts.append(.text("<|vision_start|>"))
                parts.append(.image)
                parts.append(.text("<|vision_end|>"))
            }
            parts.append(.text("<|im_end|>\n<|im_start|>user\n<|vision_start|>"))
            return parts
        }
        var parts: [Part] = [
            .text(
                "<|startoftext|><|im_start|>system\n" + system
                    + "<|im_end|>\n<|im_start|>user\n" + safeContext)
        ]
        if hasImage {
            parts.append(.text("<|vision_start|>"))
            parts.append(.image)
            parts.append(.text("<|vision_end|>"))
        }
        parts.append(.text("<|vision_start|>"))
        return parts
    }
    static let completion =
        "<|vision_end|><|im_end|>\n<|im_start|>assistant\n<think></think>"
}

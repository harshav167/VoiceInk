import Foundation
import LLMkit

/// Native Gemini multimodal client that sends a raw screenshot inline with the
/// user's transcript. Bypasses LLMkit's text-only Interactions API so the
/// enhancement model sees the actual screen pixels instead of lossy OCR output.
enum GeminiMultimodalClient {

    struct Request: Encodable, Sendable {
        let contents: [Content]
        let systemInstruction: SystemInstruction?
        let generationConfig: GenerationConfig?

        struct Content: Encodable, Sendable {
            let role: String
            let parts: [Part]
        }

        struct SystemInstruction: Encodable, Sendable {
            let parts: [Part]
        }

        struct Part: Encodable, Sendable {
            let text: String?
            let inlineData: InlineData?

            enum CodingKeys: String, CodingKey {
                case text
                case inlineData = "inline_data"
            }
        }

        struct InlineData: Encodable, Sendable {
            let mimeType: String
            let data: String

            enum CodingKeys: String, CodingKey {
                case mimeType = "mime_type"
                case data
            }
        }

        struct GenerationConfig: Encodable, Sendable {
            let thinkingConfig: ThinkingConfig?

            enum CodingKeys: String, CodingKey {
                case thinkingConfig = "thinking_config"
            }
        }

        struct ThinkingConfig: Encodable, Sendable {
            let thinkingLevel: String?

            enum CodingKeys: String, CodingKey {
                case thinkingLevel = "thinking_level"
            }
        }
    }

    struct Response: Decodable, Sendable {
        let candidates: [Candidate]?

        struct Candidate: Decodable, Sendable {
            let content: Content?
        }

        struct Content: Decodable, Sendable {
            let parts: [Part]?
        }

        struct Part: Decodable, Sendable {
            let text: String?
        }
    }

    static func chatCompletion(
        apiKey: String,
        model: String,
        userPrompt: String,
        systemPrompt: String?,
        screenImageData: Data,
        screenImageMimeType: String,
        thinkingLevel: String?,
        timeout: TimeInterval
    ) async throws -> String {
        let encodedModel = model.addingPercentEncoding(withAllowedCharacters: .urlPathAllowed) ?? model
        guard let endpoint = URL(string: "https://generativelanguage.googleapis.com/v1beta/models/\(encodedModel):generateContent") else {
            throw LLMKitError.encodingError
        }

        let imagePart = Request.Part(
            text: nil,
            inlineData: Request.InlineData(
                mimeType: screenImageMimeType,
                data: screenImageData.base64EncodedString()
            )
        )
        let textPart = Request.Part(text: userPrompt, inlineData: nil)
        let contents = [Request.Content(role: "user", parts: [imagePart, textPart])]
        let systemInstruction = systemPrompt.map {
            Request.SystemInstruction(parts: [Request.Part(text: $0, inlineData: nil)])
        }
        let generationConfig = thinkingLevel.map {
            Request.GenerationConfig(thinkingConfig: Request.ThinkingConfig(thinkingLevel: $0))
        }

        let body = Request(
            contents: contents,
            systemInstruction: systemInstruction,
            generationConfig: generationConfig
        )

        var request = URLRequest(url: endpoint)
        request.httpMethod = "POST"
        request.timeoutInterval = timeout
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        request.setValue(apiKey, forHTTPHeaderField: "x-goog-api-key")
        request.httpBody = try JSONEncoder().encode(body)

        let (data, response) = try await URLSession.shared.data(for: request)
        guard let httpResponse = response as? HTTPURLResponse else {
            throw LLMKitError.networkError("Invalid response")
        }
        guard (200...299).contains(httpResponse.statusCode) else {
            let message = String(data: data, encoding: .utf8) ?? "Unknown error"
            throw LLMKitError.httpError(statusCode: httpResponse.statusCode, message: message)
        }

        let decoded = try JSONDecoder().decode(Response.self, from: data)
        let text = decoded.candidates?
            .compactMap { candidate in
                candidate.content?.parts?.compactMap(\.text).joined()
            }
            .joined()
        guard let text, !text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else {
            throw LLMKitError.noResultReturned
        }
        return text
    }
}
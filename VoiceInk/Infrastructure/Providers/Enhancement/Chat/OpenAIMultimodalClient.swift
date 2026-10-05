import Foundation
import LLMkit

/// Universal OpenAI-compatible multimodal client.
///
/// Sends the raw screenshot as a Base64 Data URI using the standard OpenAI
/// vision message format (`image_url` parts), which is supported by Groq,
/// Cerebras, OpenRouter, and every other OpenAI-compatible endpoint.
enum OpenAIMultimodalClient {

    struct Request: Encodable, Sendable {
        let model: String
        let messages: [Message]
        let temperature: Double?
        let maxTokens: Int?
        let stream: Bool

        enum CodingKeys: String, CodingKey {
            case model
            case messages
            case temperature
            case maxTokens = "max_tokens"
            case stream
        }
    }

    struct Message: Encodable, Sendable {
        let role: String
        let content: [ContentPart]
    }

    struct ContentPart: Encodable, Sendable {
        let type: String
        let text: String?
        let imageURL: ImageURL?

        enum CodingKeys: String, CodingKey {
            case type
            case text
            case imageURL = "image_url"
        }
    }

    struct ImageURL: Encodable, Sendable {
        let url: String
    }

    struct Response: Decodable, Sendable {
        let choices: [Choice]?
        let error: APIError?

        struct Choice: Decodable, Sendable {
            let message: MessageResponse?
        }

        struct MessageResponse: Decodable, Sendable {
            let content: String?
        }

        struct APIError: Decodable, Sendable {
            let message: String?
        }
    }

    static func chatCompletion(
        baseURL: String,
        apiKey: String,
        model: String,
        userPrompt: String,
        systemPrompt: String?,
        screenImageData: Data,
        screenImageMimeType: String,
        temperature: Double? = 0.3,
        timeout: TimeInterval
    ) async throws -> String {
        // Resolve the chat completions endpoint robustly: some providers (e.g.
        // Cerebras) expose a full endpoint URL, while others expose just a base.
        let normalizedBase = baseURL.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !normalizedBase.isEmpty else {
            throw LLMKitError.encodingError
        }
        var trimmedBase = normalizedBase
        while trimmedBase.hasSuffix("/") {
            trimmedBase.removeLast()
        }
        let endpointURL: URL
        if trimmedBase.hasSuffix("/chat/completions") {
            guard let url = URL(string: trimmedBase) else {
                throw LLMKitError.encodingError
            }
            endpointURL = url
        } else {
            guard let url = URL(string: "\(trimmedBase)/chat/completions") else {
                throw LLMKitError.encodingError
            }
            endpointURL = url
        }

        let dataURI = "data:\(screenImageMimeType);base64,\(screenImageData.base64EncodedString())"
        var messages: [Message] = []
        if let systemPrompt, !systemPrompt.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
            messages.append(
                Message(
                    role: "system",
                    content: [ContentPart(type: "text", text: systemPrompt, imageURL: nil)]
                )
            )
        }
        messages.append(
            Message(
                role: "user",
                content: [
                    ContentPart(type: "image_url", text: nil, imageURL: ImageURL(url: dataURI)),
                    ContentPart(type: "text", text: userPrompt, imageURL: nil),
                ]
            )
        )

        let body = Request(
            model: model,
            messages: messages,
            temperature: temperature,
            maxTokens: nil,
            stream: false
        )

        var request = URLRequest(url: endpointURL)
        request.httpMethod = "POST"
        request.timeoutInterval = timeout
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        if !apiKey.isEmpty {
            request.setValue("Bearer \(apiKey)", forHTTPHeaderField: "Authorization")
        }
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
        if let errorMessage = decoded.error?.message {
            throw LLMKitError.httpError(statusCode: httpResponse.statusCode, message: errorMessage)
        }
        let text = decoded.choices?.compactMap { choice in
            choice.message?.content
        }.joined()
        guard let text, !text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else {
            throw LLMKitError.noResultReturned
        }
        return text
    }
}
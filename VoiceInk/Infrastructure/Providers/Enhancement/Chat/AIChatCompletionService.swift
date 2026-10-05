import Foundation
import LLMkit

struct AIChatCompletionResult: Sendable {
    let text: String
    let openRouterCompletion: OpenRouterCompletion?
}

extension AIService {
    func performChatCompletion(
        provider: AIProvider,
        modelName: String?,
        messages: [ChatMessage],
        systemPrompt: String? = nil,
        localUserPrompt: String? = nil,
        screenImageData: Data? = nil,
        screenImageMimeType: String? = nil,
        timeout: TimeInterval = 30
    ) async throws -> AIChatCompletionResult {
        let resolvedModel = modelName?.isEmpty == false ? modelName! : selectedModel(for: provider)

        let result: String
        var openRouterCompletion: OpenRouterCompletion? = nil
        switch provider {
        case .gemini:
            result = try await performGeminiCompletion(
                resolvedModel: resolvedModel,
                messages: messages,
                systemPrompt: systemPrompt,
                localUserPrompt: localUserPrompt,
                screenImageData: screenImageData,
                screenImageMimeType: screenImageMimeType,
                timeout: timeout
            )
        case .anthropic:
            result = try await AnthropicLLMClient.chatCompletion(
                apiKey: try chatAPIKey(for: provider, modelName: resolvedModel),
                model: resolvedModel,
                messages: messages,
                systemPrompt: systemPrompt,
                timeout: timeout
            )
        case .openRouter:
            if let screenImageData, let screenImageMimeType {
                result = try await performOpenAIMultimodalCompletion(
                    baseURL: "https://openrouter.ai/api/v1",
                    apiKey: try chatAPIKey(for: provider, modelName: resolvedModel),
                    modelName: resolvedModel,
                    systemPrompt: systemPrompt,
                    localUserPrompt: localUserPrompt,
                    messages: messages,
                    screenImageData: screenImageData,
                    screenImageMimeType: screenImageMimeType,
                    timeout: timeout
                )
            } else {
                let completion = try await performOpenRouterCompletion(
                    resolvedModel: resolvedModel,
                    messages: messages,
                    systemPrompt: systemPrompt,
                    timeout: timeout
                )
                guard !OpenRouterRequestPolicy.outputWasTruncated(finishReason: completion.finishReason) else {
                    throw EnhancementError.outputTruncated
                }
                result = completion.text
                openRouterCompletion = completion
            }
        case .custom:
            guard
                let customConfiguration = CustomAIProviderManager.shared.requestConfiguration(forModel: resolvedModel),
                let baseURL = URL(string: customConfiguration.baseURL)
            else {
                throw EnhancementError.notConfigured
            }
            if let screenImageData, let screenImageMimeType {
                result = try await performOpenAIMultimodalCompletion(
                    baseURL: customConfiguration.baseURL,
                    apiKey: customConfiguration.apiKey,
                    modelName: customConfiguration.modelName,
                    systemPrompt: systemPrompt,
                    localUserPrompt: localUserPrompt,
                    messages: messages,
                    screenImageData: screenImageData,
                    screenImageMimeType: screenImageMimeType,
                    timeout: timeout
                )
            } else {
                result = try await OpenAILLMClient.chatCompletion(
                    baseURL: baseURL,
                    apiKey: customConfiguration.apiKey,
                    model: customConfiguration.modelName,
                    messages: messages,
                    systemPrompt: systemPrompt,
                    temperature: 0.3,
                    timeout: timeout
                )
            }
        case .voiceInkRefine:
            throw EnhancementError.customError(
                String(localized: "VoiceInk Refine only supports transcript cleanup.")
            )
        case .ollama:
            result = try await enhanceWithOllama(
                text: localUserPrompt ?? chatPrompt(from: messages),
                systemPrompt: systemPrompt ?? "",
                model: resolvedModel,
                timeout: timeout
            )
        case .localCLI:
            result = try await enhanceWithLocalCLI(
                systemPrompt: systemPrompt ?? "",
                userPrompt: localUserPrompt ?? chatPrompt(from: messages)
            )
        default:
            guard let baseURL = URL(string: provider.baseURL) else {
                throw EnhancementError.notConfigured
            }
            let apiKey = try chatAPIKey(for: provider, modelName: resolvedModel)
            if let screenImageData, let screenImageMimeType {
                result = try await performOpenAIMultimodalCompletion(
                    baseURL: baseURL.absoluteString,
                    apiKey: apiKey,
                    modelName: resolvedModel,
                    systemPrompt: systemPrompt,
                    localUserPrompt: localUserPrompt,
                    messages: messages,
                    screenImageData: screenImageData,
                    screenImageMimeType: screenImageMimeType,
                    timeout: timeout
                )
            } else {
                let reasoningEffort = ReasoningConfig.getReasoningParameter(
                    for: provider,
                    modelName: resolvedModel
                )
                let extraBody = ReasoningConfig.getExtraBodyParameters(
                    for: provider,
                    modelName: resolvedModel
                )
                result = try await OpenAILLMClient.chatCompletion(
                    baseURL: baseURL,
                    apiKey: apiKey,
                    model: resolvedModel,
                    messages: messages,
                    systemPrompt: systemPrompt,
                    temperature: 0.3,
                    reasoningEffort: reasoningEffort,
                    extraBody: extraBody,
                    timeout: timeout
                )
            }
        }

        return AIChatCompletionResult(text: result, openRouterCompletion: openRouterCompletion)
    }

    func completeChat(
        provider: AIProvider,
        modelName: String?,
        messages: [ChatMessage],
        systemPrompt: String? = nil,
        timeout: TimeInterval = 30
    ) async throws -> String {
        let completion = try await performChatCompletion(
            provider: provider,
            modelName: modelName,
            messages: messages,
            systemPrompt: systemPrompt,
            timeout: timeout
        )
        let result = completion.text
        let filteredResult = AIEnhancementOutputFilter.filter(result)
        guard provider != .openRouter || !filteredResult.isEmpty else {
            throw EnhancementError.enhancementFailed
        }
        return filteredResult
    }

    private func performOpenAIMultimodalCompletion(
        baseURL: String,
        apiKey: String,
        modelName: String,
        systemPrompt: String?,
        localUserPrompt: String?,
        messages: [ChatMessage],
        screenImageData: Data,
        screenImageMimeType: String,
        timeout: TimeInterval
    ) async throws -> String {
        try await OpenAIMultimodalClient.chatCompletion(
            baseURL: baseURL,
            apiKey: apiKey,
            model: modelName,
            userPrompt: localUserPrompt ?? chatPrompt(from: messages),
            systemPrompt: systemPrompt,
            screenImageData: screenImageData,
            screenImageMimeType: screenImageMimeType,
            temperature: 0.3,
            timeout: timeout
        )
    }

    private func performGeminiCompletion(
        resolvedModel: String,
        messages: [ChatMessage],
        systemPrompt: String?,
        localUserPrompt: String?,
        screenImageData: Data?,
        screenImageMimeType: String?,
        timeout: TimeInterval
    ) async throws -> String {
        let apiKey = try chatAPIKey(for: .gemini, modelName: resolvedModel)

        // Route multimodal requests through the native Gemini generateContent API
        // so the raw screenshot reaches the model instead of lossy OCR output.
        if let screenImageData, let screenImageMimeType {
            return try await GeminiMultimodalClient.chatCompletion(
                apiKey: apiKey,
                model: resolvedModel,
                userPrompt: localUserPrompt ?? chatPrompt(from: messages),
                systemPrompt: systemPrompt,
                screenImageData: screenImageData,
                screenImageMimeType: screenImageMimeType,
                thinkingLevel: ReasoningConfig.geminiThinkingLevel(for: resolvedModel)?.rawValue,
                timeout: timeout
            )
        }

        return try await GeminiLLMClient.chatCompletion(
            apiKey: apiKey,
            model: resolvedModel,
            messages: messages,
            systemPrompt: systemPrompt,
            thinkingLevel: ReasoningConfig.geminiThinkingLevel(for: resolvedModel),
            store: false,
            timeout: timeout
        )
    }

    private func performOpenRouterCompletion(
        resolvedModel: String,
        messages: [ChatMessage],
        systemPrompt: String?,
        timeout: TimeInterval
    ) async throws -> OpenRouterCompletion {
        let policy = OpenRouterRequestPolicy.lowLatency(
            modelName: resolvedModel,
            modelMetadata: openRouterModelMetadata(for: resolvedModel)
        )
        return try await OpenRouterClient.chatCompletion(
            apiKey: try chatAPIKey(for: .openRouter, modelName: resolvedModel),
            model: policy.model,
            messages: messages,
            systemPrompt: systemPrompt,
            temperature: policy.temperature,
            reasoning: policy.reasoning,
            provider: policy.provider,
            includeRouterMetadata: true,
            appReferer: URL(string: "https://tryvoiceink.com"),
            appTitle: "VoiceInk",
            timeout: timeout
        )
    }

    private func chatAPIKey(for provider: AIProvider, modelName: String) throws -> String {
        if provider == .custom {
            guard let customConfiguration = CustomAIProviderManager.shared.requestConfiguration(forModel: modelName)
            else {
                throw EnhancementError.notConfigured
            }
            return customConfiguration.apiKey
        }

        guard let key = APIKeyManager.shared.getAPIKey(forProvider: provider.rawValue), !key.isEmpty else {
            throw EnhancementError.notConfigured
        }
        return key
    }

    private func chatPrompt(from messages: [ChatMessage]) -> String {
        let formattedMessages = messages.map { message in
            let label: String
            switch message.role {
            case "assistant":
                label = "assistant"
            case "user":
                label = "user"
            case "system":
                label = "system"
            default:
                label = "other"
            }
            return """
                <message role="\(label)">
                \(message.content)
                </message>
                """
        }
        .joined(separator: "\n\n")

        return """
            <conversation>
            \(formattedMessages)
            </conversation>
            """
    }
}

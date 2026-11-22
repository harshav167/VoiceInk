import AppKit
import Foundation
import os

struct RecordingContextSnapshot {
    var capturedAt = Date()
    var selectedText: String?
    var clipboardText: String?
    var screenText: String?
    var screenImageData: Data?
    var screenImageMimeType: String?
}

@MainActor
final class RecordingContextSnapshotStore {
    private(set) var snapshot = RecordingContextSnapshot()

    func updateSelectedText(_ text: String?) {
        snapshot.selectedText = Self.normalized(text)
    }

    func updateClipboardText(_ text: String?) {
        snapshot.clipboardText = Self.normalized(text)
    }

    func updateScreenText(_ text: String?) {
        snapshot.screenText = Self.normalized(text)
    }

    func updateScreenContext(text: String?, imageData: Data?, mimeType: String?) {
        snapshot.screenText = Self.normalized(text)
        snapshot.screenImageData = imageData
        snapshot.screenImageMimeType = mimeType
    }

    private static func normalized(_ text: String?) -> String? {
        guard let text else { return nil }
        let trimmed = text.trimmingCharacters(in: .whitespacesAndNewlines)
        return trimmed.isEmpty ? nil : trimmed
    }
}

@MainActor
struct RecordingContextCaptureTasks {
    let clipboard: Task<Void, Never>
    let selectedText: Task<Void, Never>
    let screen: Task<Void, Never>

    func cancelAll() {
        clipboard.cancel()
        selectedText.cancel()
        screen.cancel()
    }
}

@MainActor
enum RecordingContextCaptureService {
    private struct LFM3CleanupConfiguration {
        let strategy: LFM3POCStrategy
        let prompt: CustomPrompt?
        let useClipboardContext: Bool
        let useSelectedTextContext: Bool
        let useScreenCaptureContext: Bool
    }

    private static let logger = Logger(
        subsystem: "com.prakashjoshipax.voiceink", category: "LFM3Context")

    static func startCapture(into store: RecordingContextSnapshotStore) -> RecordingContextCaptureTasks {
        let started = Date()
        func finished(_ source: String, captured: Bool) {
            logger.notice(
                """
                Context capture source=\(source, privacy: .public) \
                seconds=\(Date().timeIntervalSince(started), privacy: .public) captured=\(captured, privacy: .public)
                """)
        }
        let clipboard = Task { @MainActor in
            let clipboardText = NSPasteboard.general.string(forType: .string)
            store.updateClipboardText(clipboardText)
            finished("clipboard", captured: clipboardText != nil)
        }
        let selectedText = Task { @MainActor in
            guard !Task.isCancelled else { return }
            let text = await SelectedTextService.fetchSelectedText()
            guard !Task.isCancelled else { return }
            store.updateSelectedText(text)
            finished("selectedText", captured: text != nil)
        }
        let screen = Task { @MainActor in
            guard CGPreflightScreenCaptureAccess(), !Task.isCancelled else { return }
            let screenCaptureService = ScreenCaptureService()
            let result = await screenCaptureService.captureContext()
            guard !Task.isCancelled else { return }
            store.updateScreenContext(
                text: result?.textContent,
                imageData: result?.imageData,
                mimeType: result?.imageData != nil ? "image/jpeg" : nil
            )
            finished("screen", captured: result?.imageData != nil)
        }
        return RecordingContextCaptureTasks(clipboard: clipboard, selectedText: selectedText, screen: screen)
    }

    /// Longest LFM3 transcription waits for context before starting without the sources not yet in.
    /// Live text must never sit behind a stalled screen or selected-text capture.
    nonisolated static let lfm3ContextWait: Duration = .seconds(2)

    /// Resolves true when every capture finished, false when `timeout` elapsed first. Unfinished
    /// captures keep running; their results simply arrive too late for this recording's context.
    nonisolated private static func waitForCaptures(
        _ captures: [Task<Void, Never>], timeout: Duration
    ) async -> Bool {
        guard !captures.isEmpty else { return true }
        return await withCheckedContinuation { (continuation: CheckedContinuation<Bool, Never>) in
            let resumed = OSAllocatedUnfairLock(initialState: false)
            let finish: @Sendable (Bool) -> Void = { completed in
                let first = resumed.withLock { done -> Bool in
                    defer { done = true }
                    return !done
                }
                if first { continuation.resume(returning: completed) }
            }
            Task {
                for capture in captures { await capture.value }
                finish(true)
            }
            Task {
                try? await Task.sleep(for: timeout)
                finish(false)
            }
        }
    }

    /// Per-source cap. A copied document or log must not fill LFM3's context window.
    nonisolated static let lfm3ReferenceCharacterLimit = 4_000

    static func lfm3InitialContext(
        for transcription: TranscriptionRuntimeConfiguration,
        enhancementService: AIEnhancementService?,
        store: RecordingContextSnapshotStore?,
        captures: RecordingContextCaptureTasks?
    ) -> () async -> LFM3SessionContext {
        guard transcription.model.provider == .lfm3,
            let cleanup = lfm3Cleanup(for: transcription.mode, enhancementService: enhancementService)
        else { return { .baseline } }
        var requiredCaptures: [Task<Void, Never>] = []
        if let captures {
            if cleanup.useClipboardContext { requiredCaptures.append(captures.clipboard) }
            if cleanup.useSelectedTextContext { requiredCaptures.append(captures.selectedText) }
            if cleanup.useScreenCaptureContext { requiredCaptures.append(captures.screen) }
        }
        let skipReviewThreshold = currentSkipShortEnhancementThreshold()
        return {
            if !requiredCaptures.isEmpty {
                let waitStarted = Date()
                let complete = await waitForCaptures(requiredCaptures, timeout: lfm3ContextWait)
                if !complete {
                    logger.error(
                        """
                        LFM3 context capture timed out after \(Date().timeIntervalSince(waitStarted), privacy: .public)s; \
                        starting with the sources captured so far
                        """)
                }
            }
            return buildLFM3Context(
                cleanup: cleanup,
                snapshot: store?.snapshot,
                skipReviewWordThreshold: skipReviewThreshold
            )
        }
    }

    /// Re-transcription from History or Audio Import: the same LFM3 cleanup instructions as the mode,
    /// with no reference text or screenshot, because neither is stored with the recording.
    static func lfm3RetranscriptionContext(
        for transcription: TranscriptionRuntimeConfiguration, enhancementService: AIEnhancementService?
    ) -> LFM3SessionContext {
        guard transcription.model.provider == .lfm3,
            let cleanup = lfm3Cleanup(for: transcription.mode, enhancementService: enhancementService)
        else { return .baseline }
        return LFM3SessionContext(
            strategy: cleanup.strategy,
            instructions: LFM3SessionContext.cleanupInstructions(prompt: cleanup.prompt?.promptText),
            contextText: "",
            screenshot: nil,
            promptName: cleanup.prompt?.title,
            skipReviewWordThreshold: nil
        )
    }

    /// Trigger-word mode switch to an LFM3 cleanup mode: builds the cleanup context from the
    /// target mode and the already-captured recording snapshot.
    static func lfm3CleanupContext(
        for mode: ModeConfig,
        enhancementService: AIEnhancementService?,
        snapshot: RecordingContextSnapshot?
    ) -> LFM3SessionContext? {
        guard let cleanup = lfm3Cleanup(for: mode, enhancementService: enhancementService) else {
            return nil
        }
        return buildLFM3Context(cleanup: cleanup, snapshot: snapshot, skipReviewWordThreshold: nil)
    }

    private static func buildLFM3Context(
        cleanup: LFM3CleanupConfiguration,
        snapshot: RecordingContextSnapshot?,
        skipReviewWordThreshold: Int?
    ) -> LFM3SessionContext {
        var reference: [String] = []
        var sourceChars: [String: Int] = [:]
        func add(_ tag: String, _ text: String?) {
            guard let text else { return }
            sourceChars[tag] = text.count
            reference.append("<\(tag)>\n\(boundedReference(text))\n</\(tag)>")
        }
        if cleanup.useClipboardContext { add("CLIPBOARD_REFERENCE", snapshot?.clipboardText) }
        if cleanup.useSelectedTextContext { add("SELECTED_TEXT_REFERENCE", snapshot?.selectedText) }
        let image = cleanup.useScreenCaptureContext ? snapshot?.screenImageData : nil
        if cleanup.useScreenCaptureContext, let screenText = snapshot?.screenText {
            if image == nil {
                add("SCREEN_REFERENCE", screenText)
            } else {
                let header = screenText
                    .components(separatedBy: .newlines)
                    .prefix(2)
                    .map { $0.trimmingCharacters(in: .whitespacesAndNewlines) }
                    .filter { !$0.isEmpty }
                    .joined(separator: "\n")
                if !header.isEmpty {
                    add("SCREEN_REFERENCE", header)
                }
            }
        }
        let contextText = reference.joined(separator: "\n")
        logger.notice(
            """
            LFM3 context resolved strategy=\(cleanup.strategy.rawValue, privacy: .public) \
            screenshotPresent=\(image != nil, privacy: .public) screenshotBytes=\(image?.count ?? 0, privacy: .public) \
            clipboardChars=\(sourceChars["CLIPBOARD_REFERENCE"] ?? 0, privacy: .public) \
            selectedChars=\(sourceChars["SELECTED_TEXT_REFERENCE"] ?? 0, privacy: .public) \
            screenTextChars=\(sourceChars["SCREEN_REFERENCE"] ?? 0, privacy: .public) \
            referenceChars=\(contextText.count, privacy: .public) \
            limitPerSource=\(lfm3ReferenceCharacterLimit, privacy: .public)
            """)
        return LFM3SessionContext(
            strategy: cleanup.strategy,
            instructions: LFM3SessionContext.cleanupInstructions(prompt: cleanup.prompt?.promptText),
            contextText: contextText,
            screenshot: image,
            promptName: cleanup.prompt?.title,
            skipReviewWordThreshold: skipReviewWordThreshold
        )
    }

    private static func lfm3Cleanup(
        for mode: ModeConfig,
        enhancementService: AIEnhancementService?
    ) -> LFM3CleanupConfiguration? {
        guard mode.outputMode != .respond else { return nil }
        let strategy = mode.effectiveLFM3CleanupPlan
        guard strategy != .baseline else { return nil }
        let prompt: CustomPrompt? = mode.selectedPrompt
            .flatMap(UUID.init(uuidString:))
            .flatMap { id in enhancementService?.allPrompts.first { $0.id == id } }
        return LFM3CleanupConfiguration(
            strategy: strategy,
            prompt: prompt,
            useClipboardContext: mode.useClipboardContext,
            useSelectedTextContext: mode.useSelectedTextContext,
            useScreenCaptureContext: mode.useScreenCapture
        )
    }

    private static func currentSkipShortEnhancementThreshold() -> Int? {
        guard UserDefaults.standard.bool(forKey: "SkipShortEnhancement") else { return nil }
        let savedThreshold = UserDefaults.standard.integer(forKey: "ShortEnhancementWordThreshold")
        return savedThreshold > 0 ? savedThreshold : 3
    }

    nonisolated private static func boundedReference(_ text: String) -> String {
        guard text.count > lfm3ReferenceCharacterLimit else { return text }
        return String(text.prefix(lfm3ReferenceCharacterLimit)) + "\n[truncated]"
    }
}

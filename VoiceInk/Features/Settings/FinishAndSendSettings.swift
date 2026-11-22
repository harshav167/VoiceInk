import Foundation

enum FinishAndSendKey: String, Codable, CaseIterable {
    case none = "none"
    case enter = "enter"
    case shiftEnter = "shiftEnter"
    case commandEnter = "commandEnter"

    var displayName: String {
        switch self {
        case .none: return String(localized: "None")
        case .enter: return String(localized: "Return (⏎)")
        case .shiftEnter: return String(localized: "Shift + Return (⇧⏎)")
        case .commandEnter: return String(localized: "Command + Return (⌘⏎)")
        }
    }

    var isEnabled: Bool {
        self != .none
    }
}

/// What ends a recording with Auto Send: Return while recording (upstream), the recording shortcut, or both.
enum FinishAndSendTrigger: String, CaseIterable {
    case returnKey = "return"
    case recordingShortcut = "shortcut"
    case both = "both"

    var displayName: String {
        switch self {
        case .returnKey: return String(localized: "Return While Recording")
        case .recordingShortcut: return String(localized: "Recording Shortcut")
        case .both: return String(localized: "Both")
        }
    }
}

enum FinishAndSendSettings {
    static let key = "finishAndSendKey"

    static var selectedKey: FinishAndSendKey {
        guard let rawValue = UserDefaults.standard.string(forKey: key) else { return .none }
        return FinishAndSendKey(rawValue: rawValue) ?? .none
    }

    static let triggerKey = "finishAndSendTrigger"

    static var trigger: FinishAndSendTrigger {
        UserDefaults.standard.string(forKey: triggerKey).flatMap(FinishAndSendTrigger.init(rawValue:)) ?? .returnKey
    }

    /// Return while recording stops, pastes and sends.
    static var sendsOnReturn: Bool {
        selectedKey.isEnabled && trigger != .recordingShortcut
    }

    /// Stopping with the recording shortcut (or stop button) pastes and sends.
    static var sendsOnShortcutStop: Bool {
        selectedKey.isEnabled && trigger != .returnKey
    }
}

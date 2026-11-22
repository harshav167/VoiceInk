import Foundation

struct LFM3ModelAssets {
    static let repository = "LiquidAI/LFM3-Omni-2B-MLX-8bit"
    static let revision = "f3b873aaa6b253e60eadb5f2388db53b84b9a5aa"
    static let files: [VoiceInkRefineModelDownloader.ModelFile] = [
        .init(
            path: "model.safetensors", size: 2_713_590_678,
            sha256: "3a6b0836c6c84caa6e479ce650389bea27cb4dba2f56e593e4f78f294bdeba97"),
        .init(
            path: "tokenizer.json", size: 17_908_634,
            sha256: "41927bd314d17d7d7fadb44e10cd0919f7b71528d7741596d8cbd15c6945ec44"),
        .init(path: "config.json", size: 7_941),
        .init(path: "tokenizer_config.json", size: 674),
        .init(path: "chat_template.jinja", size: 13_513),
        .init(path: "processor_config.json", size: 2_089),
        .init(path: "model.safetensors.index.json", size: 80_412),
        .init(path: "LICENSE", size: 10_574),
        .init(path: "README.md", size: 4_118)
    ]

    static let totalBytes: Int64 = files.reduce(0) { $0 + $1.size }
    private static let downloadSpaceHeadroomBytes: Int64 = 1_000_000_000

    // This is a checkout-local POC, not an installed model-cache convention.
    // The selected weights, integrity record and test outputs stay under the existing ignored root.
    static let directory: URL = {
        var checkout = URL(fileURLWithPath: #filePath)
        for _ in 0..<5 { checkout.deleteLastPathComponent() }
        return checkout.appendingPathComponent(
            ".local-dependencies/lfm3-omni/\(revision)", isDirectory: true)
    }()

    static var partialsDirectory: URL {
        directory.appendingPathComponent(".voiceink-download-\(revision)", isDirectory: true)
    }

    static var isDownloaded: Bool {
        VoiceInkRefineModelDownloader.isSnapshotComplete(
            at: directory, files: files, requireIntegrityRecord: true)
    }

    static var hasFilesOnDisk: Bool {
        let fileManager = FileManager.default
        guard fileManager.fileExists(atPath: directory.path),
            let contents = try? fileManager.contentsOfDirectory(atPath: directory.path)
        else { return false }
        return !contents.isEmpty
    }

    static func existingDownloadBytes() -> Int64 {
        let fileManager = FileManager.default
        var existing: Int64 = 0
        for file in files {
            let finalURL = directory.appendingPathComponent(file.path)
            if let attributes = try? fileManager.attributesOfItem(atPath: finalURL.path),
                let size = (attributes[.size] as? NSNumber)?.int64Value
            {
                existing += min(size, file.size)
                continue
            }
            let partialURL = partialsDirectory.appendingPathComponent(file.path + ".part")
            if let attributes = try? fileManager.attributesOfItem(atPath: partialURL.path),
                let size = (attributes[.size] as? NSNumber)?.int64Value
            {
                existing += min(size, file.size)
            }
        }
        return existing
    }

    static func checkDownloadSpace() throws {
        let checkout = directory.deletingLastPathComponent().deletingLastPathComponent()
            .deletingLastPathComponent()
        let capacity = try checkout.resourceValues(forKeys: [
            .volumeAvailableCapacityForImportantUsageKey
        ])
        guard let available = capacity.volumeAvailableCapacityForImportantUsage else {
            throw CocoaError(
                .fileWriteUnknown,
                userInfo: [
                    NSLocalizedDescriptionKey:
                        "Could not determine disk space for the repo-local model download."
                ])
        }
        let remainingBytes = max(Int64(0), totalBytes - existingDownloadBytes())
        let requiredBytes = remainingBytes + downloadSpaceHeadroomBytes
        guard available >= requiredBytes else { throw CocoaError(.fileWriteOutOfSpace) }
    }
}

enum LFM3POCStrategy: String, CaseIterable, Codable, Identifiable, Sendable {
    case baseline
    case contextual
    case contextualReview

    static let preferenceKey = "LFM3OmniPOCStrategy"
    var id: String { rawValue }
    var title: String {
        switch self {
        case .baseline: String(localized: "Transcription only")
        case .contextual: String(localized: "Live cleanup")
        case .contextualReview: String(localized: "Live cleanup + full-audio review")
        }
    }
    static var current: Self {
        Self(rawValue: UserDefaults.standard.string(forKey: preferenceKey) ?? "") ?? .baseline
    }
    static var defaultCleanupPlan: Self {
        current == .contextualReview ? .contextualReview : .contextual
    }
}

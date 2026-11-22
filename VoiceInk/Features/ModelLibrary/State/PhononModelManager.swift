import AppKit
import Foundation
import os

enum PhononModelPhase: Equatable {
    case downloading(fraction: Double)
    /// The one-time Core ML compile for this Mac, done right after the download so no dictation pays it.
    case preparing
}

/// Download, preparation and removal of the Phonon-2 Core ML bundle (FermionResearch/Phonon-2-CoreML, pinned).
@MainActor
final class PhononModelManager: ObservableObject {
    static let shared = PhononModelManager()

    static let repositoryID = "FermionResearch/Phonon-2-CoreML"
    static let pinnedRevision = "e931079df1f6bff26f5f416c1c8880e76a0cf2a3"
    /// Sizes and digests from the repository's SHA256SUMS at the pinned revision.
    nonisolated static let files: [VoiceInkRefineModelDownloader.ModelFile] = [
        .init(
            path: "manifest.json", size: 1_428,
            sha256: "ab3fb5dfd3fc07b1d8a24379d04418f458d0443ba6359d3a5ab2f098a7d64089"),
        .init(
            path: "decoder.bin", size: 13_301_279,
            sha256: "36fa7202dda85c603af60d930ab88d7e9f0d2a619490aff46f5384db924cfa70"),
        .init(
            path: "Phonon-2.mlpackage/Manifest.json", size: 617,
            sha256: "6ebc9309aea0e16bda9c7558521fef6307fac0afba9a5b4959fea45ed7cd0429"),
        .init(
            path: "Phonon-2.mlpackage/Data/com.apple.CoreML/model.mlmodel", size: 2_500_482,
            sha256: "4f6790ec94fe4429b10397c013563d0a51119953ff6d3da466bbcb23a25dee2a"),
        .init(
            path: "Phonon-2.mlpackage/Data/com.apple.CoreML/weights/weight.bin", size: 329_186_560,
            sha256: "93aa991318a00ed492fdbb724e7bca4e6a69c6cf671ab4f14a14ef6101e7648b"),
        .init(
            path: "LICENSE-WEIGHTS-CC-BY-4.0.txt", size: 18_657,
            sha256: "9ba9550ad48438d0836ddab3da480b3b69ffa0aac7b7878b5a0039e7ab429411"),
        .init(
            path: "NOTICE", size: 3_084,
            sha256: "b468a23a1ce2c5181ea4050432bb68713a6468ce2e452927228d24f3d56226be"),
    ]

    @Published private(set) var phase: PhononModelPhase?
    @Published private(set) var isDownloaded = false
    @Published private(set) var preparationFailed = false
    @Published private(set) var isInUse = false

    var onModelDeleted: ((String) -> Void)?
    var onModelsChanged: (() -> Void)?

    let bundleDirectory: URL
    private let preparedMarkerURL: URL
    private var downloadTask: Task<Void, Never>?
    private var verificationTask: Task<Void, Never>?
    private let logger = Logger(subsystem: "com.prakashjoshipax.voiceink", category: "PhononModelManager")

    private init() {
        bundleDirectory = FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask)[0]
            .appendingPathComponent("com.prakashjoshipax.VoiceInk")
            .appendingPathComponent("Phonon")
            .appendingPathComponent("Phonon-2-CoreML")
        preparedMarkerURL = bundleDirectory.appendingPathComponent(".phonon-prepared")

        PhononEngine.shared.setActiveSessionsHandler { [weak self] inUse in
            self?.isInUse = inUse
        }
        refreshDownloadState()
    }

    func isModelDownloaded(_ model: PhononModel) -> Bool {
        isModelDownloaded(named: model.name)
    }

    func isModelDownloaded(named modelName: String) -> Bool {
        phase == nil && isDownloaded
    }

    var hasFilesOnDisk: Bool {
        FileManager.default.fileExists(atPath: bundleDirectory.path)
    }

    var isBusy: Bool { phase != nil }

    func refreshDownloadState() {
        verificationTask?.cancel()
        verificationTask = nil

        let bundleDirectory = self.bundleDirectory
        let preparedMarkerURL = self.preparedMarkerURL
        let isPrepared = FileManager.default.fileExists(atPath: preparedMarkerURL.path)
            && PhononEngine.compiledModelExists(for: bundleDirectory)
        guard isPrepared else {
            isDownloaded = false
            return
        }

        let fastStatus = VoiceInkRefineModelDownloader.snapshotCompletionFastStatus(
            at: bundleDirectory,
            files: Self.files,
            requireIntegrityRecord: true
        )
        switch fastStatus {
        case .complete:
            isDownloaded = true
            preparationFailed = false
        case .incomplete:
            isDownloaded = false
        case .needsVerification:
            isDownloaded = false
            verificationTask = Task.detached(priority: .utility) { [weak self] in
                let verified = VoiceInkRefineModelDownloader.isSnapshotComplete(
                    at: bundleDirectory,
                    files: Self.files,
                    requireIntegrityRecord: true
                )
                guard !Task.isCancelled else { return }
                await MainActor.run { [weak self] in
                    guard let self else { return }
                    self.isDownloaded = verified
                        && FileManager.default.fileExists(atPath: preparedMarkerURL.path)
                        && PhononEngine.compiledModelExists(for: bundleDirectory)
                    if self.isDownloaded {
                        self.preparationFailed = false
                    }
                    self.onModelsChanged?()
                }
            }
        }
    }

    func startDownload(_ model: PhononModel) {
        guard downloadTask == nil, !isModelDownloaded(model) else { return }
        preparationFailed = false
        downloadTask = Task { [weak self] in
            guard let self else { return }
            await self.download(model)
            self.downloadTask = nil
        }
    }

    func cancelDownload(_ model: PhononModel) {
        downloadTask?.cancel()
    }

    private func download(_ model: PhononModel) async {
        verificationTask?.cancel()
        verificationTask = nil
        isDownloaded = false
        preparationFailed = false
        phase = .downloading(fraction: 0)
        defer {
            phase = nil
            onModelsChanged?()
        }

        let downloader = VoiceInkRefineModelDownloader(
            repositoryID: Self.repositoryID,
            revision: Self.pinnedRevision,
            modelRootDirectory: bundleDirectory,
            files: Self.files
        )
        let progressTask = Task { @MainActor [weak self, downloader] in
            while !Task.isCancelled {
                let progress = downloader.progress
                if progress.totalBytes > 0 {
                    self?.phase = .downloading(
                        fraction: min(1, Double(progress.downloadedBytes) / Double(progress.totalBytes)))
                }
                try? await Task.sleep(for: .seconds(1))
            }
        }
        defer { progressTask.cancel() }

        var enteredPreparation = false
        do {
            let operation = Task.detached(priority: .utility) { try await downloader.download() }
            try await withTaskCancellationHandler {
                try await operation.value
            } onCancel: {
                operation.cancel()
            }
            try Task.checkCancellation()
            progressTask.cancel()

            enteredPreparation = true
            phase = .preparing
            try? FileManager.default.removeItem(at: preparedMarkerURL)
            try await PhononEngine.shared.open(bundle: bundleDirectory)
            try Data("prepared".utf8).write(to: preparedMarkerURL, options: .atomic)
            isDownloaded = true
            preparationFailed = false
            logger.notice("\(model.displayName, privacy: .public) installed and prepared")
        } catch is CancellationError {
            isDownloaded = false
            logger.notice("\(model.displayName, privacy: .public) download cancelled")
        } catch {
            isDownloaded = false
            if enteredPreparation {
                preparationFailed = true
                logger.error("\(model.displayName, privacy: .public) preparation failed: \(error, privacy: .public)")
                NotificationManager.shared.showNotification(
                    title: String(format: String(localized: "%@ preparation failed"), model.displayName),
                    type: .error
                )
            } else {
                logger.error("\(model.displayName, privacy: .public) download failed: \(error, privacy: .public)")
                NotificationManager.shared.showNotification(
                    title: String(format: String(localized: "%@ download failed"), model.displayName),
                    type: .error
                )
            }
        }
    }

    func deleteModel(_ model: PhononModel) {
        guard !isBusy else { return }
        if PhononEngine.shared.hasActiveSessions {
            NotificationManager.shared.showNotification(
                title: String(
                    format: String(localized: "Cannot delete %@ while a transcription is in progress."),
                    model.displayName
                ),
                type: .warning
            )
            return
        }

        verificationTask?.cancel()
        verificationTask = nil
        objectWillChange.send()
        Task {
            guard !PhononEngine.shared.hasActiveSessions else {
                NotificationManager.shared.showNotification(
                    title: String(
                        format: String(localized: "Cannot delete %@ while a transcription is in progress."),
                        model.displayName
                    ),
                    type: .warning
                )
                return
            }
            await PhononEngine.shared.unload()
            if let cacheDirectory = PhononEngine.compiledModelCacheDirectory(for: bundleDirectory),
                FileManager.default.fileExists(atPath: cacheDirectory.path)
            {
                try? FileManager.default.removeItem(at: cacheDirectory)
            }
            do {
                if FileManager.default.fileExists(atPath: bundleDirectory.path) {
                    try FileManager.default.removeItem(at: bundleDirectory)
                }
                isDownloaded = false
                preparationFailed = false
                onModelDeleted?(model.name)
                onModelsChanged?()
            } catch {
                logger.error("Failed to delete \(model.displayName, privacy: .public): \(error, privacy: .public)")
                refreshDownloadState()
                NotificationManager.shared.showNotification(
                    title: String(format: String(localized: "Failed to delete %@"), model.displayName),
                    type: .error
                )
            }
        }
    }

    func showModelInFinder(_ model: PhononModel) {
        guard isModelDownloaded(model) else { return }
        NSWorkspace.shared.activateFileViewerSelecting([bundleDirectory])
    }
}

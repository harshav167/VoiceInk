import Foundation
import SwiftUI

@MainActor
final class LFM3ModelManager: ObservableObject {
    static let shared = LFM3ModelManager()
    @Published private(set) var isDownloaded = false
    @Published private(set) var isDownloading = false
    @Published private(set) var isPreparing = false
    @Published private(set) var isInUse = false
    @Published private(set) var hasPartialFiles = false
    @Published private(set) var progress: Double = 0
    @Published private(set) var error: String?
    var onModelsChanged: (() -> Void)?
    private var task: Task<Void, Never>?
    private var verificationTask: Task<Void, Never>?

    private init() {
        Task { [weak self] in
            await LFM3OmniEngine.shared.setActiveSessionsHandler { [weak self] inUse in
                self?.isInUse = inUse
            }
        }
        refreshDownloadState()
    }

    func refreshDownloadState() {
        verificationTask?.cancel()
        verificationTask = nil

        let fastStatus = VoiceInkRefineModelDownloader.snapshotCompletionFastStatus(
            at: LFM3ModelAssets.directory,
            files: LFM3ModelAssets.files,
            requireIntegrityRecord: true
        )
        switch fastStatus {
        case .complete:
            isDownloaded = true
            hasPartialFiles = false
        case .incomplete:
            isDownloaded = false
            hasPartialFiles = LFM3ModelAssets.hasFilesOnDisk
        case .needsVerification:
            isDownloaded = false
            hasPartialFiles = LFM3ModelAssets.hasFilesOnDisk
            verificationTask = Task.detached(priority: .utility) { [weak self] in
                let verified = LFM3ModelAssets.isDownloaded
                guard !Task.isCancelled else { return }
                await MainActor.run { [weak self] in
                    guard let self else { return }
                    self.isDownloaded = verified
                    self.hasPartialFiles = !verified && LFM3ModelAssets.hasFilesOnDisk
                    self.onModelsChanged?()
                }
            }
        }
    }

    func download() {
        guard !isDownloading else { return }
        verificationTask?.cancel()
        verificationTask = nil
        isDownloading = true
        isPreparing = false
        error = nil
        task = Task {
            let downloader = VoiceInkRefineModelDownloader(
                repositoryID: LFM3ModelAssets.repository,
                revision: LFM3ModelAssets.revision, modelRootDirectory: LFM3ModelAssets.directory,
                files: LFM3ModelAssets.files)
            let progressTask = Task {
                while !Task.isCancelled {
                    let snapshot = downloader.progress
                    progress =
                        Double(snapshot.downloadedBytes) / Double(max(1, snapshot.totalBytes))
                    try? await Task.sleep(for: .milliseconds(200))
                }
            }
            defer {
                progressTask.cancel()
                isDownloading = false
                isPreparing = false
                task = nil
                refreshDownloadState()
                onModelsChanged?()
            }
            do {
                try LFM3ModelAssets.checkDownloadSpace()
                let downloadOperation = Task.detached(priority: .utility) {
                    try await downloader.download()
                }
                try await withTaskCancellationHandler {
                    try await downloadOperation.value
                } onCancel: {
                    downloadOperation.cancel()
                }
                try Task.checkCancellation()
                progressTask.cancel()
                isPreparing = true
                try await LFM3OmniEngine.shared.prepare()
            } catch is CancellationError {
                error = nil
            } catch {
                self.error = error.localizedDescription
            }
        }
    }

    func cancelDownload() { task?.cancel() }

    func delete() async {
        guard !isDownloading else { return }
        if await LFM3OmniEngine.shared.hasActiveSessions {
            error = String(localized: "Cannot delete LFM3 Omni while a transcription is in progress.")
            return
        }
        verificationTask?.cancel()
        verificationTask = nil
        let unloaded = await LFM3OmniEngine.shared.unloadIfIdle()
        guard unloaded else {
            error = String(localized: "Cannot delete LFM3 Omni while a transcription is in progress.")
            return
        }
        do {
            if FileManager.default.fileExists(atPath: LFM3ModelAssets.directory.path) {
                try FileManager.default.removeItem(at: LFM3ModelAssets.directory)
            }
            isDownloaded = false
            hasPartialFiles = false
            error = nil
            onModelsChanged?()
        } catch {
            self.error = error.localizedDescription
            refreshDownloadState()
        }
    }
}

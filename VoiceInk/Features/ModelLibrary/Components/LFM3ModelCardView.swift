import AppKit
import SwiftUI

struct LFM3ModelCardView: View {
    let model: any TranscriptionModel
    @ObservedObject private var manager = LFM3ModelManager.shared

    var body: some View {
        VStack(alignment: .leading, spacing: 10) {
            HStack {
                Text(model.displayName).font(.system(size: 13, weight: .semibold))
                Spacer()
                if manager.isDownloading {
                    if manager.isPreparing {
                        modelStatusPill("Preparing", systemImage: "cpu")
                    }
                    Button("Cancel", action: manager.cancelDownload)
                } else if manager.isDownloaded {
                    modelStatusPill("Downloaded", systemImage: "checkmark.circle")
                    Menu {
                        Button("Show in Finder") {
                            NSWorkspace.shared.activateFileViewerSelecting([
                                LFM3ModelAssets.directory
                            ])
                        }
                        Button("Delete Model", role: .destructive) {
                            Task { await manager.delete() }
                        }
                        .disabled(manager.isInUse)
                    } label: {
                        Image(systemName: "ellipsis.circle")
                    }
                } else {
                    Button("Download 2.73 GB", action: manager.download)
                    if manager.hasPartialFiles {
                        Menu {
                            Button("Show in Finder") {
                                NSWorkspace.shared.activateFileViewerSelecting([
                                    LFM3ModelAssets.directory
                                ])
                            }
                            Button("Delete Partial Download", role: .destructive) {
                                Task { await manager.delete() }
                            }
                        } label: {
                            Image(systemName: "ellipsis.circle")
                        }
                    }
                }
            }
            Text(model.description).font(.system(size: 11)).foregroundStyle(.secondary)
            Text("Liquid AI · LFM Open License v1.0 · On-Device")
                .font(.system(size: 10)).foregroundStyle(.tertiary)
            if manager.isDownloading {
                if manager.isPreparing {
                    ProgressView()
                        .controlSize(.small)
                } else {
                    ProgressView(value: manager.progress)
                }
            }
            if let error = manager.error { Text(error).font(.caption).foregroundStyle(.red) }
        }
        .padding(16)
        .background(AppMaterialCardBackground())
    }
}

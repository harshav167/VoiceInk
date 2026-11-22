import SwiftUI

struct PhononModelCardView: View {
    let model: PhononModel
    @ObservedObject private var modelManager = PhononModelManager.shared

    private var isDownloaded: Bool { modelManager.isModelDownloaded(model) }
    private var isBusy: Bool { modelManager.isBusy }

    var body: some View {
        HStack(alignment: .top, spacing: 16) {
            VStack(alignment: .leading, spacing: 6) {
                HStack(alignment: .firstTextBaseline) {
                    Text(model.displayName)
                        .font(.system(size: 13, weight: .semibold))
                        .foregroundColor(Color(.labelColor))

                    Spacer()
                }

                HStack(spacing: 12) {
                    Label(model.language, systemImage: "globe")
                    Label(model.size, systemImage: "internaldrive")
                    HStack(spacing: 3) {
                        Text("Speed")
                        progressDotsWithNumber(value: model.speed * 10)
                    }
                    .fixedSize(horizontal: true, vertical: false)
                    HStack(spacing: 3) {
                        Text("Accuracy")
                        progressDotsWithNumber(value: model.accuracy * 10)
                    }
                    .fixedSize(horizontal: true, vertical: false)
                }
                .font(.system(size: 11))
                .foregroundColor(Color(.secondaryLabelColor))
                .lineLimit(1)

                Text(model.description)
                    .font(.system(size: 11))
                    .foregroundColor(Color(.secondaryLabelColor))
                    .lineLimit(2)
                    .fixedSize(horizontal: false, vertical: true)
                    .padding(.top, 4)

                Text("Phonon-2 by Fermion Research · CC-BY-4.0 · based on NVIDIA Parakeet TDT 0.6B v3")
                    .font(.system(size: 10))
                    .foregroundColor(Color(.tertiaryLabelColor))
                    .lineLimit(1)

                progressSection
            }
            .frame(maxWidth: .infinity, alignment: .leading)

            actionSection
        }
        .padding(16)
        .background(AppMaterialCardBackground())
    }

    @ViewBuilder
    private var progressSection: some View {
        if let phase = modelManager.phase {
            VStack(alignment: .leading, spacing: 6) {
                HStack {
                    switch phase {
                    case .downloading(let fraction):
                        Text("Downloading...")
                            .lineLimit(1)
                        Spacer()
                        Text(fraction, format: .percent.precision(.fractionLength(0)))
                            .fontDesign(.monospaced)
                    case .preparing:
                        Text("Preparing for the Neural Engine (about a minute, once)...")
                            .lineLimit(1)
                        ProgressView()
                            .controlSize(.small)
                            .scaleEffect(0.65)
                        Spacer()
                    }
                }
                .font(.system(size: 11, weight: .medium))
                .foregroundColor(Color(.secondaryLabelColor))

                if case .downloading(let fraction) = phase {
                    ProgressView(value: fraction)
                        .progressViewStyle(.linear)
                }
            }
            .frame(maxWidth: .infinity, alignment: .leading)
            .padding(.top, 8)
        } else if modelManager.preparationFailed {
            Text("Preparation failed. Try again.")
                .font(.system(size: 11, weight: .medium))
                .foregroundColor(AppTheme.Status.error)
                .padding(.top, 8)
        }
    }

    private var actionSection: some View {
        HStack(spacing: 8) {
            if isDownloaded {
                modelStatusPill("Downloaded", systemImage: "checkmark.circle")

                Menu {
                    Button(role: .destructive) {
                        modelManager.deleteModel(model)
                    } label: {
                        Label("Delete Model", systemImage: "trash")
                    }
                    .disabled(modelManager.isInUse)

                    Button {
                        modelManager.showModelInFinder(model)
                    } label: {
                        Label("Show in Finder", systemImage: "folder")
                    }
                } label: {
                    Image(systemName: "ellipsis.circle")
                        .font(.system(size: 14))
                }
                .menuStyle(.borderlessButton)
                .menuIndicator(.hidden)
                .frame(width: 20, height: 20)
            } else if case .preparing = modelManager.phase {
                modelStatusPill("Preparing", systemImage: "cpu")
            } else {
                Button {
                    if isBusy {
                        modelManager.cancelDownload(model)
                    } else {
                        modelManager.startDownload(model)
                    }
                } label: {
                    HStack(spacing: 4) {
                        Text(LocalizedStringKey(isBusy ? "Cancel" : (modelManager.preparationFailed ? "Retry" : "Download")))
                        Image(
                            systemName: isBusy
                                ? "xmark.circle"
                                : (modelManager.preparationFailed ? "arrow.clockwise.circle" : "arrow.down.circle")
                        )
                    }
                    .font(.system(size: 12, weight: .medium))
                    .foregroundColor(.white)
                    .padding(.horizontal, 12)
                    .padding(.vertical, 6)
                    .background(
                        Capsule().fill(isBusy ? AppTheme.Action.destructiveFill : AppTheme.Accent.primary)
                    )
                }
                .buttonStyle(.plain)

                if !isBusy && modelManager.hasFilesOnDisk {
                    Menu {
                        Button(role: .destructive) {
                            modelManager.deleteModel(model)
                        } label: {
                            Label("Delete Partial Download", systemImage: "trash")
                        }
                    } label: {
                        Image(systemName: "ellipsis.circle")
                            .font(.system(size: 14))
                    }
                    .menuStyle(.borderlessButton)
                    .menuIndicator(.hidden)
                    .frame(width: 20, height: 20)
                }
            }
        }
    }
}

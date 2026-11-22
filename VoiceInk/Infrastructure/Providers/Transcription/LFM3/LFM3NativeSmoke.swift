#if LOCAL_BUILD
    import CryptoKit
    import Foundation

    /// Dev-only replay through the recorder's session/engine. No mic, IPC, or cloud calls.
    @MainActor
    enum LFM3NativeSmoke {
        private struct Input {
            let audioURL: URL
            let samples: [Float]
            let audioHash: String
            let image: Data?
            let strategy: LFM3POCStrategy
            let repetitions: Int
            let systemReference: Bool
            let reportDirectory: URL?
            /// Mode prompt text; the instruction block is built by the same function live recording uses.
            let prompt: String?
            let contextText: String
            let chunkSamples: Int

            var context: LFM3SessionContext {
                .init(
                    strategy: strategy,
                    instructions: strategy == .baseline ? "" : LFM3SessionContext.cleanupInstructions(prompt: prompt),
                    contextText: contextText, screenshot: image, systemReference: systemReference)
            }
        }

        private struct PartialEvent: Encodable {
            let elapsedSeconds: Double
            let text: String
        }

        private struct Report: Encodable {
            let modelRevision: String
            let strategy: String
            let audioSHA256: String
            let imageSHA256: String?
            let sampleCount: Int
            let audioSeconds: Double
            let modelLoadSeconds: Double
            let firstTextSeconds: Double?
            let stopToFinalSeconds: Double
            let replayTotalSeconds: Double
            let partialUpdates: Int
            let text: String
            let processID: Int32
            let replayIndex: Int
            let systemReference: Bool
            let partialEvents: [PartialEvent]
        }

        private struct BatchCase: Decodable {
            let label: String
            let audio: String
            let seconds: Double?
            let image: String?
            let systemPrompt: String?
            let contextText: String?
            let systemReference: Bool?
            let userSuffix: String?
            let secondTurnPrompt: String?
        }

        private struct BatchResult: Encodable {
            let label: String
            let seconds: Double
            let elapsed: Double
            let text: String
            let error: String?
        }

        static func run(arguments: [String]) async throws {
            if let batchIdx = arguments.firstIndex(of: "--batch"), batchIdx + 1 < arguments.count {
                try await runBatch(path: arguments[batchIdx + 1])
                return
            }
            let input = try await loadInput(arguments)
            if !LFM3ModelAssets.isDownloaded {
                guard arguments.contains("--download-model") else {
                    throw LFM3ModelError.missingModel
                }
                try LFM3ModelAssets.checkDownloadSpace()
                print(
                    "Downloading selected 8-bit Omni snapshot to \(LFM3ModelAssets.directory.path)")
                let downloader = VoiceInkRefineModelDownloader(
                    repositoryID: LFM3ModelAssets.repository,
                    revision: LFM3ModelAssets.revision,
                    modelRootDirectory: LFM3ModelAssets.directory,
                    files: LFM3ModelAssets.files)
                try await downloader.download()
            }
            if arguments.contains("--check-vision") {
                guard let image = input.image else { throw SmokeError.arguments }
                print("Native image sanity check: \(try await LFM3OmniEngine.shared.inspectInitialImage(image))")
                await LFM3OmniEngine.shared.unload()
                return
            }
            for index in 0..<input.repetitions {
                let loadStart = Date()
                try await LFM3OmniEngine.shared.prepare()
                let loadSeconds = Date().timeIntervalSince(loadStart)
                let report = try await replay(input, loadSeconds: loadSeconds, index: index)
                try write(report, directory: input.reportDirectory)
            }
            await LFM3OmniEngine.shared.unload()
        }

        private static func runBatch(path: String) async throws {
            let data = try Data(contentsOf: URL(fileURLWithPath: path))
            let cases = try JSONDecoder().decode([BatchCase].self, from: data)
            try await LFM3OmniEngine.shared.prepare()
            var results: [BatchResult] = []
            for item in cases {
                let started = Date()
                do {
                    var samples = try await AudioProcessor().processAudioToSamples(
                        URL(fileURLWithPath: item.audio), normalize: false)
                    if let limit = item.seconds, limit > 0 {
                        samples = Array(samples.prefix(Int(min(limit * 16_000, Double(samples.count)))))
                    }
                    let image = try item.image.map { try Data(contentsOf: URL(fileURLWithPath: $0)) }
                    let context = LFM3SessionContext(
                        strategy: .contextual,
                        instructions: "",
                        contextText: item.contextText ?? "",
                        screenshot: image,
                        systemReference: item.systemReference ?? false,
                        customSystemPrompt: item.systemPrompt,
                        userSuffix: item.userSuffix,
                        secondTurnPrompt: item.secondTurnPrompt
                    )
                    let text = try await LFM3OmniEngine.shared.transcribe(samples: samples, context: context)
                    results.append(
                        BatchResult(
                            label: item.label,
                            seconds: Double(samples.count) / 16_000,
                            elapsed: Date().timeIntervalSince(started),
                            text: text.trimmingCharacters(in: .whitespacesAndNewlines),
                            error: nil
                        ))
                } catch {
                    results.append(
                        BatchResult(
                            label: item.label,
                            seconds: 0,
                            elapsed: Date().timeIntervalSince(started),
                            text: "",
                            error: error.localizedDescription
                        ))
                }
            }
            await LFM3OmniEngine.shared.unload()
            let encoder = JSONEncoder()
            encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
            let out = try encoder.encode(results)
            if let str = String(data: out, encoding: .utf8) {
                print("BATCH_RESULTS_JSON_START")
                print(str)
                print("BATCH_RESULTS_JSON_END")
            }
        }
        private static func loadInput(_ arguments: [String]) async throws -> Input {
            func argument(_ key: String) -> String? {
                guard let index = arguments.firstIndex(of: key), index + 1 < arguments.count else {
                    return nil
                }
                return arguments[index + 1]
            }
            guard let audioPath = argument("--lfm3-native-smoke"), audioPath.hasPrefix("/"),
                let strategy = LFM3POCStrategy(rawValue: argument("--variant") ?? "baseline")
            else {
                throw SmokeError.arguments
            }
            let repetitions = Int(argument("--repeat") ?? "1") ?? 0
            guard (1...3).contains(repetitions) else { throw SmokeError.arguments }
            let audioURL = URL(fileURLWithPath: audioPath)
            let hash = digest(try Data(contentsOf: audioURL))
            var samples = try await AudioProcessor().processAudioToSamples(audioURL, normalize: false)
            if let seconds = argument("--seconds").flatMap(Double.init), seconds > 0 {
                samples = Array(samples.prefix(Int(min(seconds * 16_000, Double(samples.count)))))
            }
            let image: Data?
            if strategy != .baseline || arguments.contains("--check-vision"), let path = argument("--image") {
                image = try Data(contentsOf: URL(fileURLWithPath: path))
            } else {
                image = nil
            }
            let reportDirectory: URL?
            if let path = argument("--report-directory") {
                let root = LFM3ModelAssets.directory.deletingLastPathComponent()
                    .resolvingSymlinksInPath().standardizedFileURL
                let requested = URL(fileURLWithPath: path).resolvingSymlinksInPath().standardizedFileURL
                guard path.hasPrefix("/"), requested.path.hasPrefix(root.path + "/") else {
                    throw SmokeError.arguments
                }
                reportDirectory = requested
            } else {
                reportDirectory = nil
            }
            let prompt = try argument("--prompt-file").map { try String(contentsOfFile: $0, encoding: .utf8) }
            let contextText = try argument("--context-file").map { try String(contentsOfFile: $0, encoding: .utf8) }
            // The microphone delivers ~512 frames at 48 kHz per callback, about 170 samples at 16 kHz.
            let chunkSamples = Int(argument("--chunk-samples") ?? "170") ?? 0
            guard chunkSamples > 0 else { throw SmokeError.arguments }
            return Input(
                audioURL: audioURL, samples: samples, audioHash: hash, image: image,
                strategy: strategy, repetitions: repetitions,
                systemReference: arguments.contains("--system-reference"),
                reportDirectory: reportDirectory, prompt: prompt, contextText: contextText ?? "",
                chunkSamples: chunkSamples)
        }

        private static func replay(_ input: Input, loadSeconds: Double, index: Int) async throws -> Report {
            let started = Date()
            var firstText: Double?
            var partialUpdates = 0
            var partialEvents: [PartialEvent] = []
            let session = LFM3TranscriptionSession(
                context: { input.context },
                onPartial: { text in
                    guard !text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else {
                        return
                    }
                    if firstText == nil { firstText = Date().timeIntervalSince(started) }
                    partialUpdates += 1
                    partialEvents.append(.init(elapsedSeconds: Date().timeIntervalSince(started), text: text))
                    print("[\(Date().timeIntervalSince(started))s] \(text)")
                })
            let configuration = TranscriptionRuntimeConfiguration(
                mode: ModeConfig(
                    name: "Native Omni replay", isAIEnhancementEnabled: input.strategy != .baseline),
                model: LFM3TranscriptionModel(), language: "en", isRealtimeEnabled: true)
            guard let callback = try await session.prepare(configuration: configuration) else {
                throw SmokeError.callback
            }
            do {
                try await feed(input.samples, chunk: input.chunkSamples, started: started, callback: callback)
                let stopped = Date()
                let text = try await session.transcribe(audioURL: input.audioURL)
                return Report(
                    modelRevision: LFM3ModelAssets.revision, strategy: input.strategy.rawValue,
                    audioSHA256: input.audioHash, imageSHA256: input.image.map(digest),
                    sampleCount: input.samples.count,
                    audioSeconds: Double(input.samples.count) / 16_000,
                    modelLoadSeconds: loadSeconds, firstTextSeconds: firstText,
                    stopToFinalSeconds: Date().timeIntervalSince(stopped),
                    replayTotalSeconds: Date().timeIntervalSince(started),
                    partialUpdates: partialUpdates,
                    text: text, processID: ProcessInfo.processInfo.processIdentifier, replayIndex: index,
                    systemReference: input.systemReference, partialEvents: partialEvents)
            } catch {
                session.cancel()
                throw error
            }
        }

        private static func feed(
            _ samples: [Float], chunk: Int, started: Date, callback: (Data) -> Void
        )
            async throws {
            for start in stride(from: 0, to: samples.count, by: chunk) {
                let end = min(start + chunk, samples.count)
                let pcm = samples[start..<end].map {
                    Int16(max(-1, min(1, $0)) * 32767).littleEndian
                }
                callback(pcm.withUnsafeBytes { Data($0) })
                let remaining = Double(end) / 16_000 - Date().timeIntervalSince(started)
                if remaining > 0 { try await Task.sleep(for: .seconds(remaining)) }
            }
        }

        private static func write(_ report: Report, directory: URL?) throws {
            let encoder = JSONEncoder()
            encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
            let data = try encoder.encode(report)
            let resultDirectory = directory ?? LFM3ModelAssets.directory.deletingLastPathComponent()
            try FileManager.default.createDirectory(at: resultDirectory, withIntermediateDirectories: true)
            let suffix = directory == nil ? "" : "-\(UUID().uuidString)"
            let variant = report.strategy + (report.systemReference ? "-system" : "")
            let resultURL = resultDirectory.appendingPathComponent(
                "smoke-\(variant)-\(report.replayIndex)\(suffix).json")
            try data.write(to: resultURL, options: .atomic)
            if let text = String(data: data, encoding: .utf8) { print(text) }
            print("Report: \(resultURL.path)")
        }

        private static func digest(_ data: Data) -> String {
            SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined()
        }

        private enum SmokeError: LocalizedError {
            case arguments, callback
            var errorDescription: String? {
                switch self {
                case .arguments:
                    "Pass an absolute WAV path after --lfm3-native-smoke and a valid --variant."
                case .callback:
                    "The native streaming session did not return its audio input callback."
                }
            }
        }
    }
#endif

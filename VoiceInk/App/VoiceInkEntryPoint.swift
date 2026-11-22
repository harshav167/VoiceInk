import Foundation

@main
struct VoiceInkEntryPoint {
    @MainActor
    static func main() async {
        #if LOCAL_BUILD
            if CommandLine.arguments.contains("--lfm3-native-smoke") {
                do {
                    try await LFM3NativeSmoke.run(arguments: CommandLine.arguments)
                    exit(0)
                } catch {
                    FileHandle.standardError.write(
                        Data("Native LFM3 smoke failed: \(error)\n".utf8))
                    exit(1)
                }
            }
        #endif
        VoiceInkApp.main()
    }
}

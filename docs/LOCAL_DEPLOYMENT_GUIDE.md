# How to install a local VoiceInk Dev build

Install a Release build into `/Applications/VoiceInk Dev.app` and sign it with the MAILSHIELD Developer ID. Use this recipe on this machine. Do not use `make local`, `make build`, or `make release` for this install.

If `/Applications/VoiceInk Dev.app` is already running, leave it running until you are ready to replace it. Do not `pkill`, `killall`, or AppleScript-quit VoiceInk without approval.

## Check the signing identity

1. Run `security find-identity -v -p codesigning`.
2. Confirm this identity exists: `Developer ID Application: MAILSHIELD AI PTE. LTD. (R8JB57W7HF)`.
3. If that identity is missing, stop.

On this machine the SHA is `74F633B5EE1FBA87D4028DCE7D13BF760B896108`. You can pass the name or the SHA.

Do not use these identities for VoiceInk Dev:

- `Apple Development: Harsha vardhan Alagurajan (63NCHG7H4Z)`
- `Apple Distribution: MAILSHIELD AI PTE. LTD. (R8JB57W7HF)`
- `Codex++ Local Signing`

Apple Development is TCC-hostile here. Apple Distribution is for App Store export. Ad-hoc and unsigned builds have died on launch.

## Skip the repo shortcuts

`make local` builds Debug with Apple Development or ad-hoc signing. It uses `VoiceInk.local.entitlements`, writes `~/Downloads/VoiceInk.app`, then runs `xattr -cr`, which strips signature metadata.

`make build` builds Debug with `CODE_SIGN_IDENTITY=""`.

`make release` and `scripts/release.sh` archive for team `V6J6A3VWY2` (Prakash Joshi), then notarize and emit a Sparkle DMG. That identity is not on this machine. The script also refuses a dirty tree.

Do not edit `DEVELOPMENT_TEAM` in `VoiceInk.xcodeproj/project.pbxproj`. Pass the MAILSHIELD team only on the `xcodebuild` line.

## Build Release

If `~/VoiceInk-Dependencies/whisper.cpp/build-apple/whisper.xcframework` is missing, run `make whisper` first.

From the repo root run `make local-dev`. That target only calls `scripts/local-dev-release.sh`, which is this command:

```bash
xcodebuild \
	-project VoiceInk.xcodeproj \
	-scheme VoiceInk \
	-configuration Release \
	-destination 'platform=macOS,arch=arm64' \
	-derivedDataPath .dev-build \
	ARCHS=arm64 \
	ONLY_ACTIVE_ARCH=YES \
	EXCLUDED_ARCHS=x86_64 \
	CODE_SIGN_INJECT_BASE_ENTITLEMENTS=NO \
	SWIFT_ACTIVE_COMPILATION_CONDITIONS='$(inherited) LOCAL_BUILD' \
	DEVELOPMENT_TEAM=R8JB57W7HF \
	CODE_SIGN_STYLE=Manual \
	CODE_SIGN_IDENTITY="Developer ID Application: MAILSHIELD AI PTE. LTD. (R8JB57W7HF)" \
	CODE_SIGN_ENTITLEMENTS="$(pwd)/VoiceInk/VoiceInk.local.entitlements" \
	build
```

Do not type that `xcodebuild` line by hand. Do not use Xcode’s Build button or XcodeBuildMCP for this install.

The app is at `.dev-build/Build/Products/Release/VoiceInk.app`.

Use Release only. A Debug build has died on launch on this machine.

The scheme embeds `VoiceInkRefineXPC.xpc` with `CodeSignOnCopy`. Xcode signs the helper when the parent identity is set. Do not re-sign unless verify fails.

`.gitignore` already ignores `.local-build/`. Keep `.dev-build` separate so a Debug `make local` tree cannot mix with this Release tree.

## Verify the signature

```bash
codesign --verify --deep --strict .dev-build/Build/Products/Release/VoiceInk.app
codesign -dv --verbose=2 .dev-build/Build/Products/Release/VoiceInk.app
codesign -dv --verbose=2 \
	.dev-build/Build/Products/Release/VoiceInk.app/Contents/XPCServices/VoiceInkRefineXPC.xpc
```

You should see all of these:

- `Authority=Developer ID Application: MAILSHIELD AI PTE. LTD. (R8JB57W7HF)`
- `TeamIdentifier=R8JB57W7HF`
- `flags=0x10000(runtime)`
- app `Identifier=com.prakashjoshipax.VoiceInk`
- XPC `Identifier=com.prakashjoshipax.VoiceInk.RefineXPC`

If verify fails, stop. Do not install.

If you must re-sign, sign the XPC first. Then sign the app. Use the same identity and Hardened Runtime on both. Do not copy `VoiceInk.entitlements` onto the XPC.

```bash
codesign --force --timestamp --options runtime \
	--sign "Developer ID Application: MAILSHIELD AI PTE. LTD. (R8JB57W7HF)" \
	.dev-build/Build/Products/Release/VoiceInk.app/Contents/XPCServices/VoiceInkRefineXPC.xpc

codesign --force --timestamp --options runtime \
	--sign "Developer ID Application: MAILSHIELD AI PTE. LTD. (R8JB57W7HF)" \
	--entitlements VoiceInk/VoiceInk.local.entitlements \
	.dev-build/Build/Products/Release/VoiceInk.app
```

Run `codesign --verify --deep --strict` again before you install.

## Install

Do not use `cp -R`. It drops extended attributes and can break the signature.

Do not run `xattr -cr` on the built app or the installed app.

If VoiceInk Dev is running, ask the person using it to quit. Then install.

If the destination is not running:

```bash
ditto .dev-build/Build/Products/Release/VoiceInk.app "/Applications/VoiceInk Dev.app"
```

If you want an atomic replace after quit:

```bash
ditto .dev-build/Build/Products/Release/VoiceInk.app /tmp/VoiceInkDev.next.app
mv "/Applications/VoiceInk Dev.app" /tmp/VoiceInkDev.prev.app
mv /tmp/VoiceInkDev.next.app "/Applications/VoiceInk Dev.app"
```

Then verify the installed copy:

```bash
codesign --verify --deep --strict "/Applications/VoiceInk Dev.app"
open "/Applications/VoiceInk Dev.app"
```

A living install on this machine (checked 2026-08-28) matches that signature. Folder name is `VoiceInk Dev.app`. Timestamp on that copy is 27 Aug 2026, 10:19 PM.

## What this install does not change

The folder is named `VoiceInk Dev.app`. The display name is still `VoiceInk`. The bundle ID is still `com.prakashjoshipax.VoiceInk`. TCC is shared with the store VoiceInk.

The XPC service name is `com.prakashjoshipax.VoiceInk.RefineXPC` in `Shared/VoiceInkRefineXPCProtocol.swift`. If you change only the app bundle ID, the helper lookup breaks unless you change that string and the XPC target ID in the same pass.

Do not change the bundle ID unless you are ready to redo TCC and the XPC name together.

The project default team stays `V6J6A3VWY2`. Override it only on the `xcodebuild` line.

## If the app dies on launch

1. Confirm the configuration was Release.
2. Confirm the identity was Developer ID Application MAILSHIELD, not Apple Development and not unsigned.
3. Confirm you used `ditto`, not `cp -R`.
4. Confirm you did not run `xattr -cr`.
5. Run `codesign --verify --deep --strict` on `/Applications/VoiceInk Dev.app`.

## Related files

- `BUILDING.md` and the `local` target in `Makefile` are the wrong recipe for this install.
- `scripts/release.sh` and `release/ExportOptions.plist` are the upstream Prakash Joshi release.
- `VoiceInk/VoiceInk.entitlements` is the upstream Release file. It includes iCloud and push. Do not use it for local Dev. Those keys need a provisioning profile.
- `VoiceInk/VoiceInk.local.entitlements` is the local Dev file. Mic, screen, network, Apple Events. No iCloud. No push. Pass it on the build line. `make local` also uses it, with the wrong signing.

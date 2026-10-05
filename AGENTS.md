# Agent rules for this checkout

## Local Dev install

The only recipe is [`docs/LOCAL_DEPLOYMENT_GUIDE.md`](docs/LOCAL_DEPLOYMENT_GUIDE.md).

Run `make local-dev` from the repo root. That target only calls [`scripts/local-dev-release.sh`](scripts/local-dev-release.sh). It is the VoiceInk scheme, Release configuration, arm64 only. It must pass `LOCAL_BUILD` so CloudKit stays off. Without that flag the app dies at launch after we strip iCloud entitlements. Do not invent a second build. Do not call `xcodebuild` or XcodeBuildMCP by hand.

`make local` is the upstream Debug path. It signs with Apple Development or ad-hoc, writes `~/Downloads/VoiceInk.app`, and runs `xattr -cr`. Do not use it for VoiceInk Dev.

Do not use `make build`, `make release`, `make dev`, unsigned compiles, Apple Development, Apple Distribution, ad-hoc signing, `CODE_SIGNING_ALLOWED=NO`, or `xattr -cr`.

Do not edit `DEVELOPMENT_TEAM` in the pbxproj. The project team stays `V6J6A3VWY2`. The MAILSHIELD team is an override on the build line only.

Identity:

`Developer ID Application: MAILSHIELD AI PTE. LTD. (R8JB57W7HF)`

SHA on this machine: `74F633B5EE1FBA87D4028DCE7D13BF760B896108`.

Product: `.dev-build/Build/Products/Release/VoiceInk.app`

Install target: `/Applications/VoiceInk Dev.app` via `ditto`. Ask before quitting a running Dev app.

## Which MCP to use

`xcrun mcpbridge` (Xcode) is for the open Xcode window. Scheme, destination, navigator, Run, debugger, previews. Do not use it to produce VoiceInk Dev.

XcodeBuildMCP is not the Dev Release entry point. `make local-dev` is.

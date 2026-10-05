#!/bin/zsh
set -euo pipefail

# Canonical local Dev Release. Keep in lockstep with docs/LOCAL_DEPLOYMENT_GUIDE.md.
# Do not add CODE_SIGNING_ALLOWED, Automatic signing, or a different identity.

cd "$(dirname "$0")/.."

if [[ ! -d "$HOME/VoiceInk-Dependencies/whisper.cpp/build-apple/whisper.xcframework" ]]; then
	echo "missing whisper.xcframework; run make whisper" >&2
	exit 1
fi

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

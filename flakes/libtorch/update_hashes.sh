#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FLAKE_FILE="${SCRIPT_DIR}/flake.nix"

# Color output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${BLUE}Updating LibTorch hashes for all supported platforms...${NC}\n"

# Define platform URLs
declare -A URLS=(
    ["macos-arm64"]="https://download.pytorch.org/libtorch/cpu/libtorch-macos-arm64-2.8.0.zip"
    ["linux-cu126"]="https://download.pytorch.org/libtorch/cu126/libtorch-shared-with-deps-2.8.0%2Bcu126.zip"
)

# Function to fetch hash for a URL
fetch_hash() {
    local url="$1"
    local platform="$2"

    echo -e "${YELLOW}Fetching hash for ${platform}...${NC}"
    echo "URL: ${url}"

    # Use nix-prefetch-url to get the hash in SRI format
    # --unpack for zip files, --type sha256 for the hash type
    local hash
    hash=$(nix-prefetch-url --unpack --type sha256 "${url}" 2>&1 | tail -n1)

    # Convert to SRI format (sha256-...)
    local sri_hash
    sri_hash=$(nix hash to-sri --type sha256 "${hash}")

    echo -e "${GREEN}Hash: ${sri_hash}${NC}\n"
    echo "${sri_hash}"
}

# Function to update hash in flake.nix
update_flake_hash() {
    local platform="$1"
    local new_hash="$2"
    local url_pattern="$3"

    echo -e "${BLUE}Updating hash for ${platform} in flake.nix...${NC}"

    # Create a temporary file
    local temp_file="${FLAKE_FILE}.tmp"

    # Use awk to update the hash after the specific URL
    awk -v url="${url_pattern}" -v hash="${new_hash}" '
    {
        print
        if ($0 ~ url && !updated) {
            getline
            if ($0 ~ /sha256 = /) {
                gsub(/"[^"]*"/, "\"" hash "\"")
                print
                updated = 1
                next
            }
        }
    }
    END {
        if (!updated) {
            print "Warning: Could not find hash line for " url > "/dev/stderr"
        }
    }
    ' "${FLAKE_FILE}" > "${temp_file}"

    mv "${temp_file}" "${FLAKE_FILE}"
    echo -e "${GREEN}Updated!${NC}\n"
}

# Process each platform
declare -A HASHES

for platform in "${!URLS[@]}"; do
    url="${URLS[$platform]}"
    hash=$(fetch_hash "${url}" "${platform}")
    HASHES[$platform]="${hash}"
done

# Update flake.nix with the new hashes
echo -e "${BLUE}Updating flake.nix with new hashes...${NC}\n"

# Update macOS ARM64 hash
if [[ -n "${HASHES[macos-arm64]:-}" ]]; then
    update_flake_hash "macOS ARM64" "${HASHES[macos-arm64]}" "libtorch-macos-arm64"
fi

# Update Linux hash
if [[ -n "${HASHES[linux-cu126]:-}" ]]; then
    update_flake_hash "Linux CUDA 12.6" "${HASHES[linux-cu126]}" "libtorch-shared-with-deps"
fi

echo -e "${GREEN}All hashes updated successfully!${NC}"
echo -e "\n${BLUE}Summary:${NC}"
for platform in "${!HASHES[@]}"; do
    echo "  ${platform}: ${HASHES[$platform]}"
done

echo -e "\n${YELLOW}Note: You may want to run 'nix flake lock --update-input nixpkgs' to update dependencies.${NC}"

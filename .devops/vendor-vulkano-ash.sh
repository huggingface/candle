#!/bin/sh
# Vendor the git dependencies of the candle Vulkan backend as local path
# dependencies, so the workspace builds without depending on unreleased
# commits in the amejia1 forks of vulkano and ash. Neither crate is expected
# to see a new crates.io release soon (ash: see ash-rs/ash#1001), so their
# source is checked into the repository instead.
#
# Vendored crates:
#   vulkano  https://github.com/amejia1/vulkano.git @ f37721a7b4cf930f963793aca6d137346f939e24
#   ash      https://github.com/amejia1/ash            @ 69ef938a8f94602defc6a5862faebe26b661e7e4
#
# What the script does:
#   1. Shallow-clones each repo (with shallow submodules) into vendor/<name>.
#   2. Checks out the exact revision and re-syncs submodules to it.
#   3. Strips the .git metadata (the top-level .git directory and any
#      submodule .git pointer files), so only the source tree remains.
#   4. Rewrites ash's dependency inside vulkano to the local path.
#   5. Rewrites candle's vulkano dependencies (the root workspace dependency
#      and candle-vulkan-kernels') to the local path.
#   6. Adds "vendor" to the workspace `exclude` list so cargo treats the
#      vendored workspaces as nested workspaces (vulkano uses workspace
#      inheritance, which otherwise resolves against candle's root and
#      fails because candle's workspace.package has no rust-version).
#
# Usage: .devops/vendor-vulkano-ash.sh
#
# Plain POSIX sh (verified with shellcheck -s dash).

set -eu

VULKANO_URL="https://github.com/amejia1/vulkano.git"
VULKANO_REV="f37721a7b4cf930f963793aca6d137346f939e24"
ASH_URL="https://github.com/amejia1/ash"
ASH_REV="69ef938a8f94602defc6a5862faebe26b661e7e4"

# Canonical repository root: the parent of this script's directory.
# shellcheck disable=SC1007 # CDPATH= is a temporary empty assignment prefix
SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
REPO_ROOT=$(cd -- "$SCRIPT_DIR/.." && pwd)
VENDOR_DIR="$REPO_ROOT/vendor"

# In-place edit of a file with a POSIX sed expression (portable, no `sed -i`).
sed_replace() {
    file=$1
    expr=$2
    tmp="${file}.$$.tmp"
    sed "$expr" "$file" > "$tmp"
    mv "$tmp" "$file"
}

# Shallow-clone a repo at a specific revision, shallow-clone its submodules,
# and strip all .git metadata so only the source tree remains.
clone_vendored() {
    url=$1
    rev=$2
    name=$3
    dest="$VENDOR_DIR/$name"

    # Re-running the script re-clones fresh so the vendored tree always
    # matches the pinned revision exactly.
    rm -rf "$dest"
    mkdir -p "$VENDOR_DIR"

    echo "==> shallow cloning $url @ $rev -> $dest"
    git clone --depth 1 --recurse-submodules --shallow-submodules "$url" "$dest"
    # Check out the exact revision (shallow) and re-sync submodules to it.
    git -C "$dest" fetch --depth 1 origin "$rev"
    git -C "$dest" checkout --detach -q FETCH_HEAD
    git -C "$dest" submodule update --init --recursive --depth 1 2>/dev/null || true

    echo "==> stripping .git metadata from $dest"
    rm -rf "$dest/.git"
    find "$dest" -name .git -type f -exec rm -f {} + 2>/dev/null || true
    find "$dest" -name .git -type d -exec rm -rf {} + 2>/dev/null || true
}

echo "==> vendoring ash and vulkano into $VENDOR_DIR"
clone_vendored "$ASH_URL" "$ASH_REV" "ash"
clone_vendored "$VULKANO_URL" "$VULKANO_REV" "vulkano"

# 4. vulkano's ash dependency -> the local ash path.
echo "==> patching vulkano's ash dependency to the local path"
sed_replace "$VENDOR_DIR/vulkano/Cargo.toml" \
    's|^ash = { git = "https://github.com/amejia1/ash", rev = "[^"]*",|ash = { path = "../ash/ash",|'

# 5. candle's vulkano dependency -> the local vulkano path.
echo "==> patching candle's vulkano dependency to the local path"
sed_replace "$REPO_ROOT/Cargo.toml" \
    's|^vulkano = { git = "https://github.com/amejia1/vulkano.git", rev = "[^"]*" }|vulkano = { path = "vendor/vulkano/vulkano" }|'
sed_replace "$REPO_ROOT/candle-vulkan-kernels/Cargo.toml" \
    's|^vulkano = { git = "https://github.com/amejia1/vulkano.git", rev = "[^"]*" }|vulkano = { path = "../vendor/vulkano/vulkano" }|'

# 6. Exclude the vendored trees from the workspace so their nested workspaces
# (vulkano's and ash's) provide workspace-package/dependencies inheritance.
if ! grep -q '^    "vendor",$' "$REPO_ROOT/Cargo.toml"; then
    echo "==> adding vendor/ to the workspace exclude list"
    awk '/^exclude = \[/ { print; print "    \"vendor\","; next } { print }' \
        "$REPO_ROOT/Cargo.toml" > "$REPO_ROOT/Cargo.toml.$$.tmp" \
        && mv "$REPO_ROOT/Cargo.toml.$$.tmp" "$REPO_ROOT/Cargo.toml"
fi

# Verify the vendored packages and the patches are in place.
for f in \
    "$VENDOR_DIR/ash/ash/Cargo.toml" \
    "$VENDOR_DIR/vulkano/vulkano/Cargo.toml"; do
    if [ ! -f "$f" ]; then
        echo "error: expected vendored manifest not found: $f" >&2
        exit 1
    fi
done
grep -q '^ash = { path = "../ash/ash"' "$VENDOR_DIR/vulkano/Cargo.toml" \
    || { echo "error: failed to patch vulkano's ash dependency" >&2; exit 1; }
grep -q '^vulkano = { path = "vendor/vulkano/vulkano" }' "$REPO_ROOT/Cargo.toml" \
    || { echo "error: failed to patch the workspace vulkano dependency" >&2; exit 1; }
grep -q '^vulkano = { path = "../vendor/vulkano/vulkano" }' "$REPO_ROOT/candle-vulkan-kernels/Cargo.toml" \
    || { echo "error: failed to patch candle-vulkan-kernels' vulkano dependency" >&2; exit 1; }
grep -q '^    "vendor",$' "$REPO_ROOT/Cargo.toml" \
    || { echo "error: failed to add vendor/ to the workspace exclude list" >&2; exit 1; }

echo "==> done. Run 'cargo build' to regenerate Cargo.lock against the vendored paths."
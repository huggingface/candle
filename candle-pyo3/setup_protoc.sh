#!/bin/bash

# Define the desired version of protoc
PROTOC_VERSION="25.0"

# Map the architecture names to the ones used in protoc release filenames
declare -A ARCH_MAP=(
  [x86_64]="linux-x86_64"
  [x86]="linux-x86_32"
  [aarch64]="linux-aarch_64"
  [s390x]="linux-s390_64"
  [ppc64le]="linux-ppcle_64"
)

# Detect the machine architecture
MACHINE_ARCH=$(uname -m)

# Find the correct architecture from the map
PROTOC_ARCH=${ARCH_MAP[$MACHINE_ARCH]}

if [ -z "$PROTOC_ARCH" ]; then
  echo "Architecture not supported!"
  exit 1
fi

# Construct the download URL
DOWNLOAD_URL="https://github.com/protocolbuffers/protobuf/releases/download/v$PROTOC_VERSION/protoc-$PROTOC_VERSION-$PROTOC_ARCH.zip"

# Download the correct protoc version for the detected architecture
wget $DOWNLOAD_URL -O protoc.zip

# Unzip the protoc binary
unzip protoc.zip -d protoc

# Cleanup the zip file
rm protoc.zip

# Add the protoc bin directory to the PATH
export PATH="$PATH:$(pwd)/protoc/bin"

# Verify the installation
protoc --version
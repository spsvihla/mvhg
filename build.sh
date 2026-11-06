#!/bin/bash
set -euo pipefail

BUILD_TYPE="Release"
CLEAN=false

while [[ $# -gt 0 ]]; do
    case "$1" in
        --build-type)
            case "$2" in
                debug) BUILD_TYPE="Debug" ;;
                profile) BUILD_TYPE="RelWithDebInfo" ;;
                release) BUILD_TYPE="Release" ;;
                *) echo "Unknown build type: $2"; exit 1 ;;
            esac
            shift 2
            ;;
        --clean)
            CLEAN=true
            shift
            ;;
        *)
            echo "Unknown option $1"; exit 1
            ;;
    esac
done

if $CLEAN; then
    echo "Cleaning build artifacts..."
    rm -rf build/ dist/ *.egg-info src/_mvhg*.so src/_mvhg*.pyd
    echo "Clean complete."
    exit 0
fi

# Clean previous local builds
rm -rf dist/ build/

echo "Building and installing package locally (Build Type: $BUILD_TYPE)..."
pip install --force-reinstall --no-build-isolation -v . \
    --config-settings=cmake.build-type="$BUILD_TYPE"

echo "Done! Package 'mvhg' is successfully installed."
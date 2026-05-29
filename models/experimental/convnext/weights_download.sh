#!/bin/bash
# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

# Downloads pretrained ConvNeXt weights published by torchvision.
# Usage: weights_download.sh [tiny|small|base|large]   (default: small)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VARIANT="${1:-small}"

case "${VARIANT}" in
    tiny)  FILE="convnext_tiny-983f1562.pth";;
    small) FILE="convnext_small-0c510722.pth";;
    base)  FILE="convnext_base-6075fbad.pth";;
    large) FILE="convnext_large-ea097f82.pth";;
    *) echo "unknown variant: ${VARIANT} (expected tiny|small|base|large)" >&2; exit 1;;
esac

WEIGHTS_URL="https://download.pytorch.org/models/${FILE}"
WEIGHTS_DIR="${SCRIPT_DIR}/weight"
WEIGHTS_PATH="${WEIGHTS_DIR}/${FILE}"

mkdir -p "${WEIGHTS_DIR}"

if [ -f "${WEIGHTS_PATH}" ]; then
    echo "Weights already present at ${WEIGHTS_PATH}"
    exit 0
fi

echo "Downloading ConvNeXt-${VARIANT} weights to ${WEIGHTS_PATH}"
if command -v wget >/dev/null 2>&1; then
    wget -O "${WEIGHTS_PATH}" "${WEIGHTS_URL}"
else
    curl -L -o "${WEIGHTS_PATH}" "${WEIGHTS_URL}"
fi

#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SRC_DIR="${SRC_DIR:-$ROOT/ThreeLPee}"
BUILD_DIR="${BUILD_DIR:-$SRC_DIR/build-py}"
PYTHON="${PYTHON:-$ROOT/.venv/bin/python}"
BUILD_TYPE="${CMAKE_BUILD_TYPE:-Release}"
OPEN3D_DIR="${OPEN3D_DIR:-/usr/local/lib/cmake/Open3D}"
HIGHS_DIR="${HIGHS_DIR:-/usr/local/lib/cmake/highs}"
PY_VISUALIZER="${PY_VISUALIZER:-OFF}"

if [[ ! -x "$PYTHON" ]]; then
  echo "Python executable not found at $PYTHON" >&2
  echo "Hint: set PYTHON=/path/to/venv/bin/python or create .venv." >&2
  exit 1
fi

PY_SITE="$("$PYTHON" - <<'PY'
import sysconfig
print(sysconfig.get_paths()["platlib"])
PY
)"

PYBIND11_DIR="$("$PYTHON" - <<'PY'
import pybind11
print(pybind11.get_cmake_dir())
PY
)"

PREFIX_PATHS=("$PYBIND11_DIR")
[[ -d "$OPEN3D_DIR" ]] && PREFIX_PATHS+=("$OPEN3D_DIR")
[[ -d "$HIGHS_DIR" ]] && PREFIX_PATHS+=("$HIGHS_DIR")
if [[ -n "${CMAKE_PREFIX_PATH:-}" ]]; then
  PREFIX_PATHS+=("$CMAKE_PREFIX_PATH")
fi
PREFIX_PATH="$(IFS=';'; echo "${PREFIX_PATHS[*]}")"

echo "Configuring ThreeLPee pybind module..."
cmake -S "$SRC_DIR" -B "$BUILD_DIR" \
  -DCMAKE_BUILD_TYPE="$BUILD_TYPE" \
  -DTHREELP_BUILD_PYTHON=ON \
  -DTHREELP_PYTHON_ENABLE_VISUALIZER="$PY_VISUALIZER" \
  -DPython3_EXECUTABLE="$PYTHON" \
  -DCMAKE_LIBRARY_OUTPUT_DIRECTORY="$PY_SITE" \
  -DCMAKE_PREFIX_PATH="$PREFIX_PATH" \
  -DOpen3D_DIR="$OPEN3D_DIR" \
  -Dhighs_DIR="$HIGHS_DIR" \
  ${EXTRA_CMAKE_ARGS:-}

JOBS="${JOBS:-$(nproc 2>/dev/null || sysctl -n hw.logicalcpu 2>/dev/null || echo 4)}"
echo "Building threelp with $JOBS jobs..."
cmake --build "$BUILD_DIR" --target threelp -j"$JOBS"

echo "Built threelp into $PY_SITE"

#!/bin/bash
set -euo pipefail

TARGET_ROOT="/Volumes/T7/20251010_halfCell/20251119_amira_o/2_seperate"
PIPELINE_SCRIPT="/Volumes/T7/20251010_projection/code/pipeline.py"
CONDA_ENV="env_cp311_ymz"

if ! command -v conda >/dev/null 2>&1; then
  echo "未找到 conda 命令，请先安装或加载 conda。" >&2
  exit 1
fi

if [ ! -d "$TARGET_ROOT" ]; then
  echo "目标目录不存在：$TARGET_ROOT" >&2
  exit 1
fi

for base_dir in "$TARGET_ROOT"/*; do
  [ -d "$base_dir" ] || continue
  echo "=== 运行 pipeline.py，base_dir=$base_dir ==="
  conda run -n "$CONDA_ENV" python "$PIPELINE_SCRIPT" --base-dir "$base_dir"
done

echo "全部子目录处理完成。"


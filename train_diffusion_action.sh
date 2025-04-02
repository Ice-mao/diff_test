#!/bin/bash
# filepath: /home/dell-t3660tow/data/manipulation/diff_test/train_sd_model.sh

# ------ 参数解析与错误处理 ------
set -e  # 任何命令失败则退出

# 解析命令行参数
while getopts ":g:" opt; do
  case $opt in
    g) export CUDA_VISIBLE_DEVICES="$OPTARG" ;;
    \?) echo "无效选项: -$OPTARG" >&2; exit 1 ;;
  esac
done

# ------ 基础参数设置 ------
PRETRAINED_MODEL="runwayml/stable-diffusion-v1-5"
OUTPUT_DIR="./output/my-finetuned-model"
RESOLUTION=512

# 数据集参数（二选一）
DATASET_NAME="lambdalabs/naruto-blip-captions"  # HuggingFace 数据集
# TRAIN_DATA_DIR="./my_dataset"                  # 本地数据集（启用需注释上一行）

# 检查参数冲突
if [ -n "$DATASET_NAME" ] && [ -n "$TRAIN_DATA_DIR" ]; then
  echo "错误：不能同时设置 DATASET_NAME 和 TRAIN_DATA_DIR"
  exit 1
fi

# ------ 训练参数 ------
TRAIN_BATCH_SIZE=4
GRADIENT_ACCUMULATION_STEPS=4
MAX_TRAIN_STEPS=2000
LEARNING_RATE=1e-5
VALIDATION_PROMPTS=("warrior with a sword" "wizard casting a spell")
VALIDATION_EPOCHS=5

# ------ 优化参数 ------
MIXED_PRECISION="fp16"
ENABLE_XFORMERS=true
GRADIENT_CHECKPOINTING=true
USE_8BIT_ADAM=true

# ------ 先进技术 ------
USE_SNR_GAMMA=true
USE_DREAM_TRAINING=true
USE_EMA=true

# ------ 其他设置 ------
SEED=42
CHECKPOINTING_STEPS=500
CHECKPOINTS_TOTAL_LIMIT=3

# ------ 依赖检查 ------
if [ "$ENABLE_XFORMERS" = true ] && ! python -c "import xformers" &>/dev/null; then
  pip install xformers
fi
if [ "$USE_8BIT_ADAM" = true ] && ! python -c "import bitsandbytes" &>/dev/null; then
  pip install bitsandbytes
fi

# ------ 构建命令 ------
CMD=(python train_text_to_image.py
  --pretrained_model_name_or_path="$PRETRAINED_MODEL"
  --output_dir="$OUTPUT_DIR"
  --resolution="$RESOLUTION"
  --train_batch_size="$TRAIN_BATCH_SIZE"
  --gradient_accumulation_steps="$GRADIENT_ACCUMULATION_STEPS"
  --max_train_steps="$MAX_TRAIN_STEPS"
  --learning_rate="$LEARNING_RATE"
  --validation_epochs="$VALIDATION_EPOCHS"
  --mixed_precision="$MIXED_PRECISION"
  --seed="$SEED"
  --checkpointing_steps="$CHECKPOINTING_STEPS"
  --checkpoints_total_limit="$CHECKPOINTS_TOTAL_LIMIT"
  --report_to="tensorboard"
)

# 添加数据集参数
if [ -n "$DATASET_NAME" ]; then
  CMD+=(--dataset_name="$DATASET_NAME")
elif [ -n "$TRAIN_DATA_DIR" ]; then
  CMD+=(--train_data_dir="$TRAIN_DATA_DIR")
fi

# 添加验证提示词
for prompt in "${VALIDATION_PROMPTS[@]}"; do
  CMD+=(--validation_prompts "$prompt")
done

# 添加优化参数
[ "$ENABLE_XFORMERS" = true ] && CMD+=(--enable_xformers_memory_efficient_attention)
[ "$GRADIENT_CHECKPOINTING" = true ] && CMD+=(--gradient_checkpointing)
[ "$USE_8BIT_ADAM" = true ] && CMD+=(--use_8bit_adam)

# 添加先进技术参数
[ "$USE_SNR_GAMMA" = true ] && CMD+=(--snr_gamma=5.0)
[ "$USE_DREAM_TRAINING" = true ] && CMD+=(--dream_training)
[ "$USE_EMA" = true ] && CMD+=(--use_ema)

# ------ 执行训练 ------
mkdir -p "$OUTPUT_DIR"
LOG_FILE="${OUTPUT_DIR}/train_$(date +%Y%m%d_%H%M%S).log"
echo "执行命令: ${CMD[*]}" | tee "$LOG_FILE"
"${CMD[@]}" 2>&1 | tee -a "$LOG_FILE"
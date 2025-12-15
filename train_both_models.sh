#!/bin/bash

# train both tiny and small models for 3d medical segmentation
# runs them sequentially to avoid gpu memory conflicts

set -e  # exit on error

# default args
DATASET="btcv"
DATA_PATH="./data/btcv"
IMAGE_SIZE=1024
VAL_FREQ=1
PROMPT="bbox"
PROMPT_FREQ=2
BATCH_SIZE_TINY=1
BATCH_SIZE_SMALL=1
HIGH_MEMORY=false

# detect high-end gpu and suggest better defaults
if command -v nvidia-smi &> /dev/null; then
    GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n 1)
    if [[ "$GPU_NAME" == *"H200"* ]] || [[ "$GPU_NAME" == *"H100"* ]] || [[ "$GPU_NAME" == *"A100"* ]]; then
        echo "detected high-end gpu: $GPU_NAME"
        echo "suggest using --high-memory flag for optimal batch sizes"
        echo ""
    fi
fi

# parse command line args if provided
while [[ $# -gt 0 ]]; do
    case $1 in
        --dataset)
            DATASET="$2"
            shift 2
            ;;
        --data-path)
            DATA_PATH="$2"
            shift 2
            ;;
        --image-size)
            IMAGE_SIZE="$2"
            shift 2
            ;;
        --batch-size)
            BATCH_SIZE_TINY="$2"
            BATCH_SIZE_SMALL="$2"
            shift 2
            ;;
        --batch-size-tiny)
            BATCH_SIZE_TINY="$2"
            shift 2
            ;;
        --batch-size-small)
            BATCH_SIZE_SMALL="$2"
            shift 2
            ;;
        --high-memory)
            HIGH_MEMORY=true
            BATCH_SIZE_TINY=16
            BATCH_SIZE_SMALL=12
            shift
            ;;
        --prompt)
            PROMPT="$2"
            shift 2
            ;;
        --prompt-freq)
            PROMPT_FREQ="$2"
            shift 2
            ;;
        --val-freq)
            VAL_FREQ="$2"
            shift 2
            ;;
        *)
            echo "unknown option: $1"
            echo "usage: $0 [options]"
            echo ""
            echo "options:"
            echo "  --dataset DATASET           dataset name (default: btcv)"
            echo "  --data-path PATH            path to data (default: ./data/btcv)"
            echo "  --image-size SIZE           image size (default: 1024)"
            echo "  --batch-size SIZE           batch size for both models (default: 1)"
            echo "  --batch-size-tiny SIZE       batch size for tiny model only"
            echo "  --batch-size-small SIZE      batch size for small model only"
            echo "  --high-memory               use aggressive batch sizes for h200/h100/a100 (tiny: 16, small: 12)"
            echo "  --prompt TYPE                prompt type: bbox or click (default: bbox)"
            echo "  --prompt-freq FREQ          prompt frequency (default: 2)"
            echo "  --val-freq FREQ             validation frequency (default: 1)"
            echo ""
            echo "examples:"
            echo "  $0 --high-memory"
            echo "  $0 --batch-size 8"
            echo "  $0 --batch-size-tiny 16 --batch-size-small 8"
            exit 1
            ;;
    esac
done

echo "=========================================="
echo "training both tiny and small models"
echo "=========================================="
echo "dataset: $DATASET"
echo "data path: $DATA_PATH"
echo "image size: $IMAGE_SIZE"
echo "batch size (tiny): $BATCH_SIZE_TINY"
echo "batch size (small): $BATCH_SIZE_SMALL"
echo "prompt: $PROMPT"
echo "prompt freq: $PROMPT_FREQ"
echo "val freq: $VAL_FREQ"
if [ "$HIGH_MEMORY" = true ]; then
    echo "high-memory mode: enabled"
fi
echo "=========================================="
echo ""

# train tiny model
echo "starting training for hiera-tiny (batch size: $BATCH_SIZE_TINY)..."
python train_3d.py \
    -net sam2 \
    -exp_name ${DATASET}_MedSAM2_Tiny \
    -sam_ckpt ./checkpoints/sam2_hiera_tiny.pt \
    -sam_config sam2_hiera_t \
    -image_size $IMAGE_SIZE \
    -val_freq $VAL_FREQ \
    -prompt $PROMPT \
    -prompt_freq $PROMPT_FREQ \
    -dataset $DATASET \
    -data_path $DATA_PATH \
    -b $BATCH_SIZE_TINY

if [ $? -eq 0 ]; then
    echo ""
    echo "tiny model training completed successfully!"
    echo ""
else
    echo ""
    echo "tiny model training failed!"
    exit 1
fi

# train small model
echo "starting training for hiera-small (batch size: $BATCH_SIZE_SMALL)..."
python train_3d.py \
    -net sam2 \
    -exp_name ${DATASET}_MedSAM2_Small \
    -sam_ckpt ./checkpoints/sam2_hiera_small.pt \
    -sam_config sam2_hiera_s \
    -image_size $IMAGE_SIZE \
    -val_freq $VAL_FREQ \
    -prompt $PROMPT \
    -prompt_freq $PROMPT_FREQ \
    -dataset $DATASET \
    -data_path $DATA_PATH \
    -b $BATCH_SIZE_SMALL

if [ $? -eq 0 ]; then
    echo ""
    echo "small model training completed successfully!"
    echo ""
    echo "=========================================="
    echo "both models trained successfully!"
    echo "checkpoints saved in logs/${DATASET}_MedSAM2_*/Model/"
    echo "=========================================="
else
    echo ""
    echo "small model training failed!"
    exit 1
fi


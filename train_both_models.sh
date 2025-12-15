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
BATCH_SIZE=1

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
            BATCH_SIZE="$2"
            shift 2
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
            echo "usage: $0 [--dataset DATASET] [--data-path PATH] [--image-size SIZE] [--batch-size SIZE] [--prompt TYPE] [--prompt-freq FREQ] [--val-freq FREQ]"
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
echo "batch size: $BATCH_SIZE"
echo "prompt: $PROMPT"
echo "prompt freq: $PROMPT_FREQ"
echo "val freq: $VAL_FREQ"
echo "=========================================="
echo ""

# train tiny model
echo "starting training for hiera-tiny..."
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
    -b $BATCH_SIZE

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
echo "starting training for hiera-small..."
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
    -b $BATCH_SIZE

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


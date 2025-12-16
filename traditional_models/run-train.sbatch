#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=20GB
#SBATCH --time=36:00:00
#SBATCH --account=pr_130_general
#SBATCH --gres=gpu:a100:1
#SBATCH --mail-type=END
#SBATCH --mail-user=dr3432@nyu.edu
#SBATCH --job-name=Train-Seg
#SBATCH --output=slurm-TrainSeg-%j.out

# Define the list of allowed models (No commas!)
models=("sam3" "resnet" "yolo")

# Initialize a flag to track if we found a match
match_found=false

if [ -n "$1" ]; then
    # Check if input matches one of the models
    for item in "${models[@]}"; do
        if [[ "$item" == "$1" ]]; then
            echo "Model type: $1"
            ARGS="--model $1"
            match_found=true
            break # Stop looping once we find a match
        fi
    done

    # If the loop finishes and match_found is still false, error out
    if [ "$match_found" = false ]; then
        echo "Error encountered: $1 not in expected options: [sam3, resnet, yolo]"
        exit 1
    fi

else
    # No model provided
    echo "Error, no model provided. Provide one of the following options: [sam3, resnet, yolo]."
    exit 1
fi

module purge

# Note: Ideally load singularity module here if required (Torch might ask for it)
# module load singularity-

singularity exec --nv \
    --overlay /scratch/$USER/computerVision/pytorch_env.ext3:ro \
    /scratch/work/public/singularity/cuda12.8.1-cudnn9.8.0-ubuntu24.04.2.sif \
    /bin/bash -c "source /ext3/env.sh; wandb login 5175a68411ba45be63d8e92c0919df86ad2eb3d7; \ 
python train.py $ARGS"

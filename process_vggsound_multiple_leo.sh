#!/bin/bash
#SBATCH --job-name="jid:vla-fn48"
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:2
#SBATCH --gpus-per-task=1
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=boost_qos_dbg
#SBATCH --mem=180GB
#SBATCH --time=00:20:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=a-sophia.koepke@uni-tuebingen.de
#SBATCH --output=./logs/slurm-%j.out
#SBATCH --error=./logs/slurm-%j.out

nvidia-smi

# Activate your conda environment (adjust if needed)
set -x

export SRUN_CPUS_PER_TASK=$SLURM_CPUS_PER_TASK

modality=$1
echo "This is $modality"

# Set the appropriate prompt based on the modality
PROMPT="Do you see or hear \"{cl}\" class in this video? Answer only with yes or no."


srun bash -c "python code/process_vggsound.py \
    --output_csv ./csv/$modality/predictions.csv \
    --dataset_path /leonardo_work/EUHPC_E03_068/akoepke/vs \
    --frames_dataset_path /leonardo_scratch/large/userexternal/akoepke0/data_tmp/cav-mae/cav-mae-test \
    --video_csv ../../data/test.csv \
    --page \$SLURM_PROCID \
    --per_page 49 \
    --modality $modality \
    --device cuda:${SLURM_LOCALID} \
    --prompt_mode multi \
    --prompt \"$PROMPT\" \
    --temperature 1.0
    "


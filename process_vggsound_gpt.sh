#!/bin/sh
#SBATCH --job-name="pandagpt"
#SBATCH --array=0-0
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --partition=mcml-dgx-a100-40x8,mcml-hgx-a100-80x4
#SBATCH --qos=mcml
#SBATCH --mem=48G
#SBATCH --time=48:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=zverev@in.tum.de
#SBATCH --output=./logs/slurm-%A_%a.out
#SBATCH --error=./logs/slurm-%A_%a.out
 
nvidia-smi

# Activate your conda environment (adjust if needed)
set -x

export SRUN_CPUS_PER_TASK=$SLURM_CPUS_PER_TASK

modality=$1
echo "This is $modality, page $SLURM_ARRAY_TASK_ID"

# Set the appropriate prompt based on the modality
if [ "$modality" = "a" ]; then
    PROMPT="What actions are being performed in this audio, explain all sounds and actions in the audio? Please provide a short answer."
else
    PROMPT="What actions are being performed in this video, explain all sounds and actions in the video? Please provide a short answer."
fi

# Run the script on each node, assigning each task to a different GPU
python code/process_vggsound.py \
  --output_csv ./csv/$modality/predictions.csv \
  --dataset_path $MCMLSCRATCH/datasets/vggsound_test \
  --frames_dataset_path $MCMLSCRATCH/datasets/cav-mae-test/ \
  --video_csv ../../data/test_sample.csv \
  --page $SLURM_ARRAY_TASK_ID \
  --per_page 100 \
  --modality $modality \
  --device cuda:0 \
  --prompt_mode gpt \
  --prompt "$PROMPT"
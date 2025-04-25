#!/bin/bash
#SBATCH --job-name="jobid:v-1882js8"
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=boost_qos_dbg
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=thaddaeus.wiedemer@bethgelab.org
#SBATCH --output=./logs/slurm-%j.out
#SBATCH --error=./logs/slurm-%j.out

nvidia-smi

# Activate your conda environment (adjust if needed)
source activate avllama
set -x

export SRUN_CPUS_PER_TASK=$SLURM_CPUS_PER_TASK

modality=$1
echo "This is $modality"

# Run the script on each node, assigning each task to a different GPU
srun --exclusive --ntasks=1 python code/process_vggsound.py \
    --output_csv ./csv/$modality/llama2_predictions_ask.csv \
    --dataset_path ADD_VGGSOUND_PATH \
    --frames_dataset_path /tmp/cav-mae/vggsound/ \
    --video_csv ../../data/test.csv \
    --page $((SLURM_PROCID + 1)) \
    --per_page 250 \
    --processor $modality \
    --device cuda:$SLURM_LOCALID \
    --prompt_mode single \
    --prompt "classes: {cl}. What classes do you see in this video? Answer only with the class names."
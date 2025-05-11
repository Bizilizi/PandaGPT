#!/bin/sh
#SBATCH --job-name="pandagpt"
#SBATCH --array=0-16
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --partition=mcml-dgx-a100-40x8,mcml-hgx-a100-80x4,mcml-hgx-h100-94x4
#SBATCH --qos=mcml
#SBATCH --mem=48G
#SBATCH --time=48:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=zverev@in.tum.de
#SBATCH --output=./logs/slurm-%A_%a.out
#SBATCH --error=./logs/slurm-%A_%a.out
 
nvidia-smi

# Mount squashfs files
cleanup () {
    fusermount -u /tmp/zverev/$SLURM_JOB_ID/vggsound
    rmdir /tmp/zverev/$SLURM_JOB_ID/vggsound
    rm -rf $MCMLSCRATCH/datasets/$SLURM_JOB_ID
}

trap cleanup EXIT

echo "Mounting VGGsound"
mkdir -p /tmp/akoepke/$SLURM_JOB_ID/vggsound
mkdir -p /tmp/akoepke/$SLURM_JOB_ID/cav-mae
mkdir -p $MCMLSCRATCH/datasets/$SLURM_JOB_ID

echo "Copying VGGsound"
rsync -av --progress --ignore-existing /dss/dssmcmlfs01/pn67gu/pn67gu-dss-0000/zverev/datasets/vggsound_test.squashfs $MCMLSCRATCH/datasets/vggsound_test.squashfs
rsync -av --progress --ignore-existing /dss/dssmcmlfs01/pn67gu/pn67gu-dss-0000/zverev/datasets/cav-mae-test.squashfs $MCMLSCRATCH/datasets/cav-mae-test.squashfs

echo "Copying VGGsound to job directory"
rsync -av --progress --ignore-existing $MCMLSCRATCH/datasets/vggsound_test.squashfs $MCMLSCRATCH/datasets/$SLURM_JOB_ID/vggsound_test.squashfs
rsync -av --progress --ignore-existing $MCMLSCRATCH/datasets/cav-mae-test.squashfs $MCMLSCRATCH/datasets/$SLURM_JOB_ID/cav-mae-test.squashfs

squashfuse $MCMLSCRATCH/datasets/$SLURM_JOB_ID/vggsound_test.squashfs /tmp/akoepke/$SLURM_JOB_ID/vggsound
squashfuse $MCMLSCRATCH/datasets/$SLURM_JOB_ID/cav-mae-test.squashfs /tmp/akoepke/$SLURM_JOB_ID/cav-mae

# Activate your conda environment (adjust if needed)
set -x

export SRUN_CPUS_PER_TASK=$SLURM_CPUS_PER_TASK

modality=$1
echo "This is $modality, page $SLURM_ARRAY_TASK_ID"

# Set the appropriate prompt based on the modality
if [ "$modality" = "a" ]; then
    PROMPT="Classes: {cl}. From the given list of classes, which ones do you hear in this audio? Answer using the exact names of the classes, separated by commas."
else
    PROMPT="Classes: {cl}. From the given list of classes, which ones do you see or hear in this video? Answer using the exact names of the classes, separated by commas."
fi

# Run the script on each node, assigning each task to a different GPU
python code/process_vggsound.py \
  --output_csv ./csv/$modality/predictions.csv \
  --dataset_path /tmp/akoepke/$SLURM_JOB_ID/vggsound \
  --frames_dataset_path /tmp/akoepke/$SLURM_JOB_ID/cav-mae/vggsound \
  --video_csv ../../data/test.csv \
  --page $SLURM_ARRAY_TASK_ID \
  --per_page 1000 \
  --modality $modality \
  --device cuda:0 \
  --prompt_mode single \
  --prompt "$PROMPT"
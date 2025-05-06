#SBATCH --job-name="jobid:v-1882js8"
#SBATCH --nodes=16
#SBATCH --ntasks=64
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:4
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=normal
#SBATCH --mem=256G
#SBATCH --time=05:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=a-sophia.koepke@uni-tuebingen.de
#SBATCH --output=./logs/slurm-%j.out
#SBATCH --error=./logs/slurm-%j.out

nvidia-smi

# Activate your conda environment (adjust if needed)
micromamba activate pandagpt
set -x

export SRUN_CPUS_PER_TASK=$SLURM_CPUS_PER_TASK

modality=$1
echo "This is $modality"

CMD="
python code/process_vggsound.py \
    --output_csv ./csv/$modality/llama2_predictions_ask.csv \
    --dataset_path /leonardo_work/EUHPC_E03_068/akoepke/vs \
    --frames_dataset_path ./tmp \
    --video_csv ../../data/test.csv \
    --page \$SLURM_PROCID \
    --per_page 250 \
    --modality $modality \
    --device cuda:\$SLURM_LOCALID \
    --prompt_mode single \
    --prompt 'classes: {cl}. What classes do you see in this video? Answer only with the class names.'
    "

SRUN_ARGS=" \
    --wait=60 \
    --kill-on-bad-exit=1 \
    --jobid $SLURM_JOB_ID \
    --exclusive\
     --ntasks=1 \
    "
# Run the script on each node, assigning each task to a different GPU
srun $SRUN_ARGS bash -c "$CMD"

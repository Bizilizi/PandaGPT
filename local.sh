modality=$1
echo "This is $modality, page $2"

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
  --frames_dataset_path $MCMLSCRATCH/datasets/cav-mae-test/vggsound \
  --video_csv ../../data/test_sample.csv \
  --page 0 \
  --per_page 10 \
  --modality $modality \
  --device cuda:0 \
  --prompt_mode gpt \
  --prompt "$PROMPT"

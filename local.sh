modality=$1
echo "This is $modality, page $2"

# Set the appropriate prompt based on the modality
if [ "$modality" = "a" ]; then
    PROMPT="Classes: {cl}. From the given list of classes, which ones do you hear in this audio? Answer using the exact names of the classes, separated by commas."
else
    PROMPT="Classes: {cl}. From the given list of classes, which ones do you see or hear in this video? Answer using the exact names of the classes, separated by commas."
fi

# Run the script on each node, assigning each task to a different GPU
python code/process_vggsound.py \
  --output_csv ./csv/$modality/predictions.csv \
  --dataset_path /tmp/zverev/vggsound \
  --frames_dataset_path /tmp/zverev/cav-mae/vggsound \
  --video_csv ../../data/test.csv \
  --page 0 \
  --per_page 10 \
  --modality $modality \
  --device cuda:0 \
  --prompt_mode single \
  --prompt "$PROMPT"

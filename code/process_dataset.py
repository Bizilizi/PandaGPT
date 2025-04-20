#!/usr/bin/env python
import os
import sys
import torch
import shutil
import traceback
import pandas as pd
import argparse
from tqdm.auto import tqdm, trange
from model.openllama import OpenLLAMAPEFTModel

CLASSES = pd.read_csv("../../../data/audio_classes.csv")["display_name"].tolist()


def predict(
    model,
    input,
    image_paths=[],
    audio_paths=[],
    video_paths=[],
    thermal_paths=[],
    max_length=256,
    top_p=0.1,
    temperature=1,
    modality_cache=[],
):

    prompt_text = input
    response = model.generate(
        {
            "prompt": prompt_text,
            "image_paths": image_paths,
            "audio_paths": audio_paths,
            "video_paths": video_paths,
            "thermal_paths": thermal_paths,
            "top_p": top_p,
            "temperature": temperature,
            "max_tgt_len": max_length,
            "modality_embeds": modality_cache,
        }
    )
    return response


@torch.inference_mode()
def process_video(
    model,
    dataset_path,
    frames_dataset_path,
    video_id,
    temperature,
    top_p,
    max_output_tokens,
    processor="av",
    prompt="Do you hear or see '{cl}' class in this video? Answer only with yes or no.",
    prompt_mode="single",
):
    """
    Process a single video file and detect classes.
    Returns a list of detected classes based on whether the generated response
    contains the word "yes".
    """
    image_paths = []
    audio_paths = []
    video_paths = []

    if processor == "a":
        video_id = video_id.replace(".mp4", ".wav")
        audio_paths = [os.path.join(dataset_path, "audio", video_id)]
    elif processor == "v":
        image_paths = [
            os.path.join(
                frames_dataset_path,
                "sample_frames",
                f"frame_{i}.jpg",
                f"{video_id}.jpg",
            )
            for i in range(10)
        ]
        image_paths = [path for path in image_paths if os.path.exists(path)]
    elif processor == "av":
        video_paths = [os.path.join(dataset_path, "video", video_id)]
    else:
        raise ValueError(f"Invalid processor: {processor}")

    print(
        f"""
        Modality input:
        image_paths: {len(image_paths)}
        audio_paths: {len(audio_paths)}
        video_paths: {len(video_paths)}
    """
    )

    detected = []
    response = ""
    
    # Process detection classes in batches:
    if prompt_mode == "single":
        prompt = prompt.format(cl=", ".join(CLASSES))
        response = predict(
            model=model,
            input=prompt,
            image_paths=image_paths,
            audio_paths=audio_paths,
            video_paths=video_paths,
            thermal_paths=[],
            max_length=max_output_tokens,
            top_p=top_p,
            temperature=temperature,
            modality_cache=[],
        )
        for cl in CLASSES:
            if cl in response.lower():
                detected.append(cl)
                
        response = response
        
    elif prompt_mode == "multi":
        all_responses = []
        for cl in tqdm(CLASSES, desc="Processing classes", leave=False):
            prompt = prompt.format(cl=cl)
            response = predict(
                model=model,
                input=prompt,
                image_paths=image_paths,
                audio_paths=audio_paths,
                    video_paths=video_paths,
                    thermal_paths=[],
                    max_length=max_output_tokens,
                    top_p=top_p,
                    temperature=temperature,
                    modality_cache=[],
                )

            if "yes" in response.lower():
                detected.append(cl)
            all_responses.append(f"{cl}: {response}")
            
        response = "\n".join(all_responses)
    else:
        raise ValueError(f"Invalid prompt mode: {prompt_mode}. Supported modes: 'single', 'multi'")

    # Return the unique set of detected classes.
    return list(set(detected)), response


def get_model():
    # init the model
    args = {
        "model": "openllama_peft",
        "imagebind_ckpt_path": "../pretrained_ckpt/imagebind_ckpt",
        "vicuna_ckpt_path": "../pretrained_ckpt/vicuna_ckpt/7b_v0",
        "delta_ckpt_path": "../pretrained_ckpt/pandagpt_ckpt/7b/pytorch_model.pt",
        "stage": 2,
        "max_tgt_len": 128,
        "lora_r": 32,
        "lora_alpha": 32,
        "lora_dropout": 0.1,
    }
    model = OpenLLAMAPEFTModel(**args)
    delta_ckpt = torch.load(args["delta_ckpt_path"], map_location=torch.device("cpu"))
    model.load_state_dict(delta_ckpt, strict=False)
    model = model.eval().half().cuda()
    
    return model


def get_video_list(csv_path):
    """
    Reads video IDs from a CSV file.
    Assumes CSV with two columns: video_id and label. If the video_id does not
    end with '.mp4', it appends '.mp4'.
    """
    if not os.path.exists(csv_path):
        print(f"CSV file {csv_path} not found.")
        return []
    df = pd.read_csv(csv_path, names=["video_id", "label"], header=None)
    video_ids = df["video_id"].tolist()
    video_ids = [vid if vid.endswith(".mp4") else vid + ".mp4" for vid in video_ids]
    return video_ids


def write_predictions_csv(predictions, responses, output_csv):
    """
    Writes the predictions dictionary to a CSV file.
    The CSV will have two columns: video_id and suggestions.
    """
    df_table = {
        i: {"video_id": vid, "suggestions": list(predictions[vid]), "response": responses[vid]}
        for i, vid in enumerate(predictions.keys())
    }
    df = pd.DataFrame.from_dict(df_table, orient="index")
    df.to_csv(output_csv, index=False)
    print(f"Predictions CSV saved to {output_csv}")


def main():
    parser = argparse.ArgumentParser(
        description="Process videos and generate a predictions CSV using VideoLLaMA2"
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        default="/storage/slurm/zverev/datasets/vggsound",
        help="Path to the directory containing video files",
    )
    parser.add_argument(
        "--frames_dataset_path",
        type=str,
        default="/storage/slurm/zverev/datasets/cav-mae/vggsound",
        help="Path to the directory containing video files",
    )
    parser.add_argument(
        "--video_csv",
        type=str,
        default="../../data/train.csv",
        help="CSV file that contains the list of video IDs",
    )
    parser.add_argument(
        "--output_csv",
        type=str,
        default="../../data/llama2_predictions.csv",
        help="Output CSV file for writing predictions",
    )
    parser.add_argument(
        "--batch_size", type=int, default=2, help="Batch size for processing classes"
    )
    parser.add_argument(
        "--temperature", type=float, default=0.2, help="Temperature for generation"
    )
    parser.add_argument("--top_p", type=float, default=0.9, help="Top P for generation")
    parser.add_argument(
        "--max_output_tokens", type=int, default=2048, help="Maximum output tokens"
    )
    parser.add_argument(
        "--page",
        type=int,
        default=1,
        help="Page number to process",
    )
    parser.add_argument(
        "--per_page",
        type=int,
        default=1000,
        help="Number of videos to process per page",
    )
    parser.add_argument(
        "--modality",
        type=str,
        default="av",
        help="Processor to use",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="Device to use",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="Do you hear or see '{cl}' class in this video? Answer only with yes or no.",
        help="Prompt to use",
    )
    parser.add_argument(
        "--prompt_mode",
        type=str,
        default="single",
        help="Prompt mode to use",
    )

    args = parser.parse_args()

    # Initialize the LLama2 handler.
    print(f"Using device: {args.device}")
    model = get_model()

    # Get list of videos to process.
    video_list = get_video_list(args.video_csv)
    if not video_list:
        print("No videos found to process.")
        return

    page_videos = video_list[
        args.page * args.per_page : (args.page + 1) * args.per_page
    ]
    args.output_csv = args.output_csv.replace(".csv", f"{args.prompt_mode}_page_{args.page}.csv")

    predictions = {}
    responses = {}
    
    for video_id in tqdm(page_videos, desc="Processing Videos"):
        detected_classes, response = process_video(
            model=model,
            dataset_path=args.dataset_path,
            frames_dataset_path=args.frames_dataset_path,
            video_id=video_id,
            temperature=args.temperature,
            top_p=args.top_p,
            max_output_tokens=args.max_output_tokens,
            processor=args.modality,
            prompt=args.prompt,
            prompt_mode=args.prompt_mode,
        )
        predictions[video_id] = detected_classes
        responses[video_id] = response
        
        # Write predictions to CSV.
        write_predictions_csv(predictions, responses, args.output_csv)


if __name__ == "__main__":
    main()

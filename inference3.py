import os
import json
import librosa
import numpy as np
import pandas as pd
import random
import argparse
from tqdm import tqdm
from datetime import datetime
import torch

import soundfile as sf
import torchaudio
from torch.utils.data import Dataset, DataLoader
from transformers import SchedulerType, get_scheduler
import sed_eval
import dcase_util

import sys
WORKSPACE_PATH = r"C:\Users\Rushali Shah\Desktop\FakeSound-main"
sys.path.extend([WORKSPACE_PATH])
from train import DeepfakeDetectionDataset
from models import detection_model
import eval_util

class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

def parse_args():
    parser = argparse.ArgumentParser(description="Test a deepfake audio detection model.")
    parser.add_argument(
        "--test_files", '-f', default={
            "easy": f"{WORKSPACE_PATH}/deepfake_data/output2.json",  # Update if needed
        }
    )
    parser.add_argument(
        "--batch_size", '-b', type=int, default=8,
        help="Batch size for the dataloader.",
    )
    parser.add_argument(
        "--threshold", '-th', type=float, default=0.5,  # Lowered to capture more fake predictions
        help="Threshold for classification.",
    )
    parser.add_argument(
        "--exp_path", '-e', type=str, default=r"C:\Users\Rushali Shah\Desktop\FakeSound-main",
        help="Load experiment path."
    )
    parser.add_argument(
        "--original_args", type=str, default="summary.jsonl",
        help="Path for summary jsonl file saved during training."
    )
    parser.add_argument(
        "--model_pt", type=str, default="best.pt",
        help="Path for saved model bin file."
    )
    args = parser.parse_args()
    args.original_args = os.path.join(args.exp_path, args.original_args)    
    args.model_pt = os.path.join(args.exp_path, args.model_pt)
    return args
    
def main():
    args = parse_args()
    train_args = dotdict(json.loads(open(args.original_args).readlines()[0]))

    # Set the training seed
    seed = train_args.seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Init
    device = "cpu"
    model = getattr(detection_model, train_args.model_class)(multi_task=train_args.multi_task).to(device).eval()
    model.load_state_dict(torch.load(args.model_pt, map_location=torch.device('cpu')))

    for test_set, test_file in args.test_files.items():
        dataset = DeepfakeDetectionDataset(test_file, train_args)       
        dataloader = DataLoader(dataset, shuffle=False, batch_size=args.batch_size, collate_fn=dataset.collate_fn)
        print(f"Num instances in test set {test_set}: {len(dataset)}")
        ref_list, pred_list = [], []
        total_num, correct_num = 0, 0
        time_resolution, threshold = train_args.time_resolution, args.threshold
        for step, batch in tqdm(enumerate(dataloader)):
            audio, binary_label, tgt, audio_id = batch
            # Debug: Print label distribution
            print(f"Batch {step} binary_label: {binary_label.tolist()}")
            with torch.no_grad():
                output = model(audio.to(device))
                pred = torch.sigmoid(output["pred"].squeeze(-1)).cpu().numpy()
                filtered_prob = eval_util.median_filter(pred, window_size=1, threshold=threshold)
            # Debug: Print prediction distribution
            print(f"Batch {step} filtered_prob mean: {np.mean(filtered_prob):.3f}, min: {np.min(filtered_prob):.3f}, max: {np.max(filtered_prob):.3f}")
            # Accuracy calculation
            if hasattr(model, "multi_task_classifier"):
                pred_binary = output["pred_binary"].squeeze(-1).cpu().numpy() > threshold
            else:
                pred_binary = np.max(filtered_prob, axis=1) > threshold
            correct_num += (binary_label == pred_binary).sum()
            total_num += binary_label.shape[0]
            # Segment F1 calculation
            for idx, _ in enumerate(pred):         
                # Predicted info
                is_fake = np.max(filtered_prob[idx]) > threshold
                event_label = 'fake' if is_fake else 'real'
                change_indices = eval_util.find_contiguous_regions(filtered_prob[idx])
                # Debug: Print change_indices
                print(f"audio_id: {audio_id[idx]}, is_fake: {is_fake}, change_indices: {change_indices}")
                if not is_fake:
                    pred_list.append({
                        'event_label': 'real',
                        'onset': 0.0,
                        'offset': 0.0,
                        'filename': audio_id[idx],
                    })
                elif len(change_indices) == 0:
                    pred_list.append({
                        'event_label': 'fake',
                        'onset': 0.0,
                        'offset': 10.0,  # Default for fake
                        'filename': audio_id[idx],
                    })
                else:
                    for row in change_indices:
                        pred_list.append({
                            'event_label': 'fake',
                            'onset': row[0] * time_resolution,
                            'offset': row[1] * time_resolution,
                            'filename': audio_id[idx],
                        })
                # Reference info
                ref_event_label = 'real' if binary_label[idx] == 0 else 'fake'
                if binary_label[idx] == 0:
                    ref_onset, ref_offset = 0.0, 0.0
                else:
                    try:
                        onset_offset = tgt[idx] if isinstance(tgt[idx], str) else "0_10"
                        parts = onset_offset.split("_")
                        if len(parts) == 2:
                            ref_onset, ref_offset = float(parts[0]), float(parts[1])
                        else:
                            print(f"Warning: Invalid onset_offset '{onset_offset}' for audio_id '{audio_id[idx]}'. Using defaults.")
                            ref_onset, ref_offset = 0.0, 10.0
                    except Exception as e:
                        print(f"Error processing onset_offset for audio_id '{audio_id[idx]}': {e}. Using defaults.")
                        ref_onset, ref_offset = 0.0, 10.0
                ref_list.append({
                    'event_label': ref_event_label,
                    'onset': ref_onset,
                    'offset': ref_offset,
                    'filename': audio_id[idx],
                })
        print("test set:", test_set)
        print("acc:", round(correct_num / total_num, 2))
        print("ref_list sample:", ref_list[:5])
        print("pred_list sample:", pred_list[:5])
        calculate_sed_metric(ref_list, pred_list)

def calculate_sed_metric(ref_list, pred_list):
    reference_event_list = dcase_util.containers.MetaDataContainer(ref_list)
    estimated_event_list = dcase_util.containers.MetaDataContainer(pred_list)

    segment_based_metrics_1s = sed_eval.sound_event.SegmentBasedMetrics(
        event_label_list=reference_event_list.unique_event_labels,
        time_resolution=1.0
    )
    segment_based_metrics_002s = sed_eval.sound_event.SegmentBasedMetrics(
        event_label_list=reference_event_list.unique_event_labels,
        time_resolution=0.1
    )
    event_based_metrics = sed_eval.sound_event.EventBasedMetrics(
        event_label_list=reference_event_list.unique_event_labels,
        t_collar=0.5
    )

    for filename in reference_event_list.unique_files:
        reference_event_list_for_current_file = reference_event_list.filter(
            filename=filename
        )
        estimated_event_list_for_current_file = estimated_event_list.filter(
            filename=filename
        )
        segment_based_metrics_1s.evaluate(
            reference_event_list=reference_event_list_for_current_file,
            estimated_event_list=estimated_event_list_for_current_file
        )
        segment_based_metrics_002s.evaluate(
            reference_event_list=reference_event_list_for_current_file,
            estimated_event_list=estimated_event_list_for_current_file
        )
        event_based_metrics.evaluate(
            reference_event_list=reference_event_list_for_current_file,
            estimated_event_list=estimated_event_list_for_current_file
        )

    f_measure_1s = segment_based_metrics_1s.results_overall_metrics()['f_measure']['f_measure']
    f_measure_002s = segment_based_metrics_002s.results_overall_metrics()['f_measure']['f_measure']
    print("segment_based 1s Result:", f_measure_1s if not np.isnan(f_measure_1s) else "NaN (no valid events detected)")
    print("segment_based 0.02s Result:", f_measure_002s if not np.isnan(f_measure_002s) else "NaN (no valid events detected)")

if __name__ == "__main__":
    main()
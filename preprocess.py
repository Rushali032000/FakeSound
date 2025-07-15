import os
import json
import soundfile as sf
import torch

# Configuration
dataset_type = "LA"  # or "PA"
base_path = r"C:\Users\Rushali Shah\Desktop\LA\ASVspoof2019_LA_train"  # Adjust to your path
input_wav_dir = r"C:\Users\Rushali Shah\Desktop\LA\ASVspoof2019_LA_train\wav"  # After converting .flac to .wav
output_wav_dir = r"C:\Users\Rushali Shah\Desktop\LA\converted_wav_10s"  # Preprocessed 10s files
protocol_file = r"C:\Users\Rushali Shah\Desktop\LA\ASVspoof2019_LA_cm_protocols\ASVspoof2019.LA.cm.train.trn.txt.{dataset_type}.cm.train.tr{'l' if dataset_type == 'LA' else 'n'}.txt"
output_json = r"C:\Users\Rushali Shah\Desktop\LA\finalaudio.json"

sample_rate = 16000
target_length = 160000  # 10s at 16kHz




# Step 3: Create train.json
data = {"audios": []}
with open(protocol_file, "r") as f:
    for line in f:
        parts = line.strip().split()
        file_id = parts[1]
        label_str = parts[4]
        label = 0 if label_str == "bonafide" else 1
        onset_offset = "0_0" if label == 0 else "0_10"  # 10s duration
        filepath = os.path.join(output_wav_dir, f"{file_id}.wav")
        if os.path.exists(filepath):
            data["audios"].append({
                "filepath": filepath,
                "label": label,
                "onset_offset": onset_offset,
                "audio_id": file_id
            })

with open(output_json, "w") as f:
    json.dump(data, f, indent=4)

print(f"Created {output_json} with {len(data['audios'])} entries.")
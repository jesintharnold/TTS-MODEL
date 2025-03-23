import os
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np

class LJSpeechDataset(Dataset):
    def __init__(self, metadata_path, mel_dir, duration_dir,phoneme_dict,max_data=None,mel_min=None,mel_max=None):
        
        self.metadata_path = metadata_path
        self.mel_dir = mel_dir
        self.duration_dir = duration_dir
        self.phoeneme_dict=phoneme_dict
        self.mel_min = mel_min
        self.mel_max = mel_max
        self.duration_max = 100.0
        self.max_data = max_data
        self.data = self._load_metadata()

    def _load_metadata(self):
            data = []
            with open(self.metadata_path, 'r', encoding='utf-8') as f:
                for i, line in enumerate(f):
                    if self.max_data is not None and i >= self.max_data:
                        break
                    parts = line.strip().split("|")
                    audio_id = parts[0]
                    text = parts[1]
                    phonemes = torch.tensor(self._phoneme_to_indices_(phoneme_string=parts[2]),dtype=torch.long)
                    mel_path = f"{self.mel_dir}/LJSpeech-mel-{audio_id}.npy" 
                    duration_path = f"{self.duration_dir}/LJSpeech-duration-{audio_id}.npy"
                    mel = torch.tensor(np.load(mel_path), dtype=torch.float32)
                    durations = torch.tensor(np.load(duration_path), dtype=torch.float32)
                    durations = durations / self.duration_max
                    data.append({
                        'text': text,
                        'phonemes': phonemes,
                        'mel': mel,
                        'duration': durations
                    })

            return data
    
    def _phoneme_to_indices_(self,phoneme_string):
        phonemes = phoneme_string.split()
        return [self.phoeneme_dict.get(p,self.phoeneme_dict["UNK"]) for p in phonemes]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def melspectogram_max_min(dataset):
    mel_spectograms = []
    for item in dataset:
        mel_spectograms.append(item["mel"].numpy())
    
    mel_spectograms = np.concatenate(mel_spectograms,axis=0)
    mel_min = np.min(mel_spectograms)
    mel_max = np.max(mel_spectograms)

    print(f"Mel Min: {mel_min}")
    print(f"Mel Max: {mel_max}")
    import json
    stats = {"mel_min": float(mel_min), "mel_max": float(mel_max)}
    with open("mel_min_max.json", "w") as f:
        json.dump(stats, f)

    print("Mel min and max saved to mel_min_max.json")
    
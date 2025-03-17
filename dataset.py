import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np

class LJSpeechDataset(Dataset):
    def __init__(self, metadata_path, mel_dir, duration_dir,phoneme_dict,max_data=None,mel_min=None,mel_max=None):
        self.metadata = self._load_metadata(metadata_path,max_data)
        self.mel_dir = mel_dir
        self.duration_dir = duration_dir
        self.phoeneme_dict=phoneme_dict
        self.mel_min = mel_min
        self.mel_max = mel_max

    def _load_metadata(self, metadata_path,max_datapoints):
        data = []
        with open(metadata_path, 'r', encoding='utf-8') as file:
            for line in file:
                parts = line.strip().split('|')
                file_id = parts[0]
                speaker = parts[1]
                phonemes = parts[2].strip('{}')
                text = parts[3]
                data.append((file_id, speaker, phonemes, text))
                if max_datapoints is not None and len(data) >= max_datapoints:
                    break
        print("Total length :",len(data))
        return data
    
    def _phoneme_to_indices_(self,phoneme_string):
        arr=[]
        phonemes = phoneme_string.split()
        return [self.phoeneme_dict.get(p,self.phoeneme_dict["UNK"]) for p in phonemes]

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        file_id, speaker, phonemes, text = self.metadata[idx]
        mel_path = f"{self.mel_dir}/LJSpeech-mel-{file_id}.npy"
        duration_path = f"{self.duration_dir}/LJSpeech-duration-{file_id}.npy"
        mel = torch.tensor(np.load(mel_path), dtype=torch.float32)
        
        if self.mel_min is not None and self.mel_max is not None:
            mel = (mel - self.mel_min) / (self.mel_max - self.mel_min)
        duration = torch.tensor(np.load(duration_path), dtype=torch.float32)
        phonemes = torch.tensor(self._phoneme_to_indices_(phoneme_string=phonemes),dtype=torch.long)
        return {
            'mel': mel,
            'duration': duration,
            'phonemes': phonemes,
            'text': text
        }


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
    
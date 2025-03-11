import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np

class LJSpeechDataset(Dataset):
    def __init__(self, metadata_path, mel_dir, duration_dir,phoneme_dict):
        self.metadata = self._load_metadata(metadata_path)
        self.mel_dir = mel_dir
        self.duration_dir = duration_dir
        self.phoeneme_dict=phoneme_dict

    def _load_metadata(self, metadata_path):
        data = []
        with open(metadata_path, 'r', encoding='utf-8') as file:
            for line in file:
                parts = line.strip().split('|')
                file_id = parts[0]
                speaker = parts[1]
                phonemes = parts[2].strip('{}')
                text = parts[3]
                data.append((file_id, speaker, phonemes, text))
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
        duration = torch.tensor(np.load(duration_path), dtype=torch.int64)
        phonemes = torch.tensor(self._phoneme_to_indices_(phoneme_string=phonemes),dtype=torch.long)
        return {
            'mel': mel,
            'duration': duration,
            'phonemes': phonemes,
            'text': text
        }


import json
import os
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np

class LJSpeechDataset(Dataset):
    def __init__(self, metadata_path, mel_dir, duration_dir, pitch_dir, energy_dir, phoneme_dict, max_data=None, 
                 mel_min=None, mel_max=None, pitch_min=None, pitch_max=None, energy_min=None, energy_max=None):
        
        self.metadata_path = metadata_path
        self.mel_dir = mel_dir
        self.duration_dir = duration_dir
        self.pitch_dir = pitch_dir
        self.energy_dir = energy_dir
        self.phoeneme_dict=phoneme_dict
        self.mel_min = mel_min
        self.mel_max = mel_max
        self.pitch_min = pitch_min
        self.pitch_max = pitch_max
        self.energy_min = energy_min
        self.energy_max = energy_max
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
                    mel = torch.tensor(np.load(mel_path), dtype=torch.float32)
                    if self.mel_min is not None and self.mel_max is not None:
                        mel = (mel - self.mel_min) / (self.mel_max - self.mel_min)

                    duration_path = f"{self.duration_dir}/LJSpeech-duration-{audio_id}.npy"
                    durations = torch.tensor(np.load(duration_path), dtype=torch.float32)
                    durations = durations / self.duration_max

                    pitch_path = f"{self.pitch_dir}/LJSpeech-pitch-{audio_id}.npy"
                    pitch = torch.tensor(np.load(pitch_path), dtype=torch.float32)
                    expanded_pitch = []
                    for j in range(len(durations)):
                        repeat_count = int(durations[j].item() * self.duration_max)
                        expanded_pitch.append(pitch[j].repeat(repeat_count))
                    expanded_pitch = torch.cat(expanded_pitch, dim=0)

                    if expanded_pitch.shape[0] > mel.shape[0]:
                        expanded_pitch = expanded_pitch[:mel.shape[0]]
                    elif expanded_pitch.shape[0] < mel.shape[0]:
                        expanded_pitch = torch.cat([expanded_pitch, torch.zeros(mel.shape[0] - expanded_pitch.shape[0], device=expanded_pitch.device)], dim=0)
                    if self.pitch_min is not None and self.pitch_max is not None:
                        expanded_pitch = (expanded_pitch - self.pitch_min) / (self.pitch_max - self.pitch_min)

                    energy_path = f"{self.energy_dir}/LJSpeech-energy-{audio_id}.npy"
                    energy = torch.tensor(np.load(energy_path), dtype=torch.float32)
                    expanded_energy = []
                    for j in range(len(durations)):
                        repeat_count = int(durations[j].item() * self.duration_max)
                        expanded_energy.append(energy[j].repeat(repeat_count))
                    expanded_energy = torch.cat(expanded_energy, dim=0)
                    
                    if expanded_energy.shape[0] > mel.shape[0]:
                        expanded_energy = expanded_energy[:mel.shape[0]]
                    elif expanded_energy.shape[0] < mel.shape[0]:
                        expanded_energy = torch.cat([expanded_energy, torch.zeros(mel.shape[0] - expanded_energy.shape[0], device=expanded_energy.device)], dim=0)
                    if self.energy_min is not None and self.energy_max is not None:
                        expanded_energy = (expanded_energy - self.energy_min) / (self.energy_max - self.energy_min)

                    # print(f"Sample {audio_id}: Mel length: {mel.shape[0]}, Expanded Pitch length: {expanded_pitch.shape[0]}, Expanded Energy length: {expanded_energy.shape[0]}")

                    data.append({
                        'text': text,
                        'phonemes': phonemes,
                        'mel': mel,
                        'duration': durations,
                        'pitch': expanded_pitch,
                        'energy': expanded_energy
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
    mel_min = float('inf')
    mel_max = float('-inf')
    pitch_min = float('inf')
    pitch_max = float('-inf')
    energy_min = float('inf')
    energy_max = float('-inf')
    
    for item in dataset:
        mel = item["mel"].numpy()
        mel_min = min(mel_min, np.min(mel))
        mel_max = max(mel_max, np.max(mel))
    
        pitch = item["pitch"].numpy()
        pitch_min = min(pitch_min, np.min(pitch))
        pitch_max = max(pitch_max, np.max(pitch))
    
        energy = item["energy"].numpy()
        energy_min = min(energy_min, np.min(energy))
        energy_max = max(energy_max, np.max(energy))

    print(f"Mel Min: {mel_min}, Mel Max: {mel_max}")
    print(f"Pitch Min: {pitch_min}, Pitch Max: {pitch_max}")
    print(f"Energy Min: {energy_min}, Energy Max: {energy_max}")

    mel_stats = {"mel_min": float(mel_min), "mel_max": float(mel_max)}
    with open("mel_min_max.json", "w") as f:
        json.dump(mel_stats, f)
    print("Mel min and max saved to mel_min_max.json")

    pitch_stats = {"pitch_min": float(pitch_min), "pitch_max": float(pitch_max)}
    with open("pitch_min_max.json", "w") as f:
        json.dump(pitch_stats, f)
    print("Pitch min and max saved to pitch_min_max.json")

    energy_stats = {"energy_min": float(energy_min), "energy_max": float(energy_max)}
    with open("energy_min_max.json", "w") as f:
        json.dump(energy_stats, f)
    print("Energy min and max saved to energy_min_max.json")
    
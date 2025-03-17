import json
import librosa.display
import torch
from env import AttrDict
from train import phoneme_map
from g2p_en import G2p
from train import TransformerTTS
from hifiganmodels import Generator as HIFIGAN
import soundfile as sf
import librosa
import nltk
import matplotlib.pyplot as plt
import numpy as np


try:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    global h
    with open("./LJ_V1/config.json", "r") as f:
        json_config = json.loads(f.read())
    h = AttrDict(json_config)
    hifigan = HIFIGAN(h)
    hifiganmodel=torch.load("./LJ_V1/generator_v1",map_location=device)
    hifigan.load_state_dict(hifiganmodel["generator"])
    hifigan = hifigan.to(device=device)
    hifigan.eval()
    print("HIFIGAN model is loaded")
    with torch.no_grad():
        predicted_mel_spectogram=np.load('./LJSPEECH/mel/LJSpeech-mel-LJ001-0001.npy')
        predicted_mel_spectogram = torch.tensor(predicted_mel_spectogram, dtype=torch.float32).to(device=device)
        predicted_mel_spectogram = predicted_mel_spectogram.permute(1,0)
        print("Predicted mel-spectrogram shape:", predicted_mel_spectogram.shape)
        # predicted_mel_spectogram = (predicted_mel_spectogram - predicted_mel_spectogram.min()) / (predicted_mel_spectogram.max() - predicted_mel_spectogram.min())
        plt.figure(figsize=(10, 4))
        librosa.display.specshow(predicted_mel_spectogram.cpu().numpy(), sr=22050, hop_length=256, x_axis='time', y_axis='mel')
        plt.colorbar(format='%+2.0f dB')
        plt.title('original Mel-Spectrogram')
        plt.savefig("./checkpoints/original002.png")
        plt.close()
        print("Mel-spectrogram saved to ./checkpoints/original002.png")
    with torch.no_grad():
        audio = hifigan(predicted_mel_spectogram).squeeze(0).cpu().numpy()
    audio = audio / np.max(np.abs(audio))
    outputfile = "./checkpoints/audio-original-002.wav"
    sf.write(outputfile,audio,samplerate=22050)
    print("Audio saved")
    
except Exception as e:
     print(f"Error during inference: {e}")
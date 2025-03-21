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
# nltk.download('averaged_perceptron_tagger')
# nltk.download('cmudict')
g2p=G2p()

h=None

print(phoneme_map)


def convert_to_phonemes(text, phonememap, device):
    phonemes = g2p(text=text)
    phonemes = ['P', 'R', 'IH1', 'N', 'T', 'IH0', 'NG', 'sp', 'IH1', 'N', 'DH', 'IY0', 'OW1', 'N', 'L', 'IY0', 'S', 'EH1', 'N', 'S', 'W', 'IH1', 'DH', 'sp', 'W', 'IH1', 'CH', 'W', 'IY1', 'AA1', 'R', 'AE1', 'T', 'P', 'R', 'EH1', 'Z', 'AH0', 'N', 'T', 'K', 'AH0', 'N', 'S', 'ER1', 'N', 'D', 'sp', 'D', 'IH1', 'F', 'ER0', 'Z', 'sp', 'F', 'R', 'AH1', 'M', 'M', 'OW1', 'S', 'T', 'IH1', 'F', 'N', 'AA1', 'T', 'F', 'R', 'AH1', 'M', 'AO1', 'L', 'DH', 'IY0', 'AA1', 'R', 'T', 'S', 'AH0', 'N', 'D', 'K', 'R', 'AE1', 'F', 'T', 'S', 'R', 'EH2', 'P', 'R', 'IH0', 'Z', 'EH1', 'N', 'T', 'IH0', 'D', 'IH1', 'N', 'DH', 'IY0', 'EH2', 'K', 'S', 'AH0', 'B', 'IH1', 'SH', 'AH0', 'N']
    phonemes_indices = [phonememap.get(p, phonememap["UNK"]) for p in phonemes]
    print("Extracted phonemes:", phonemes)
    print("Extracted phonemes indices:", phonemes_indices)
    return torch.tensor(phonemes_indices, dtype=torch.long).unsqueeze(0).to(device=device)

def inferenceModel():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    global h
    with open("./LJ_V1/config.json", "r") as f:
        json_config = json.loads(f.read())
    h = AttrDict(json_config)
    model = TransformerTTS(vocab_size=100,embedding_dim=256,hidden_dim=512,n_heads=8,n_layers=4,output_dim=80)
    checkpoint = torch.load("./checkpoints/model_epoch_49.pt",map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model=model.to(device=device)
    model.eval()
    print("TTS model is loaded")
    hifigan = HIFIGAN(h)
    hifiganmodel=torch.load("./LJ_V1/generator_v1",map_location=device)
    hifigan.load_state_dict(hifiganmodel["generator"])
    hifigan = hifigan.to(device=device)
    hifigan.eval()
    print("HIFIGAN model is loaded")

    with open("./mel_min_max.json",'r') as f:
        melconfig = json.load(f)
        mel_min=melconfig["mel_min"]
        mel_max=melconfig["mel_max"]

    while True:
        text = str(input("Enter the sentence for conversion (or type 'exit' to quit): "))
        if text.lower() == "exit":
            print("Exiting...")
            break
        else:
            try:
                phoneme_indices = convert_to_phonemes(text=text,phonememap=phoneme_map,device=device)
                with torch.no_grad():
                    predicted_mel_spectogram,predicted_durations=model(phoneme_indices)
                    predicted_durations = torch.clamp(predicted_durations, min=1.0)
                    predicted_durations = torch.round(predicted_durations).int()
                    print("Predicted Durations :",predicted_durations)
                    
                    # First check if the output is already in the normalized range [0,1]
                    print(f"Pre-denorm Mel Range: Min={predicted_mel_spectogram.min().item():.2f}, Max={predicted_mel_spectogram.max().item():.2f}")
                    predicted_mel_spectogram = (predicted_mel_spectogram - predicted_mel_spectogram.min()) / (predicted_mel_spectogram.max() - predicted_mel_spectogram.min())
                    print(f"Post-norm Mel Range: Min={predicted_mel_spectogram.min().item():.2f}, Max={predicted_mel_spectogram.max().item():.2f}")

                    # Apply denormalization
                    predicted_mel_spectogram = predicted_mel_spectogram * (mel_max - mel_min) + mel_min

                    print(f"Post-denorm Mel Range: Min={predicted_mel_spectogram.min().item():.2f}, Max={predicted_mel_spectogram.max().item():.2f}")
                    print(f"Expected Mel Range: Min={mel_min:.2f}, Max={mel_max:.2f}")
                    
                    predicted_mel_spectogram = predicted_mel_spectogram.squeeze(0)
                    predicted_mel_spectogram = predicted_mel_spectogram.permute(1,0)
                    print("Predicted mel-spectrogram shape:", predicted_mel_spectogram.shape)


                    plt.figure(figsize=(10, 4))
                    librosa.display.specshow(predicted_mel_spectogram.cpu().numpy(), sr=22050, hop_length=256, x_axis='time', y_axis='mel')
                    plt.colorbar(format='%+2.0f dB')
                    plt.title('Inference Mel-Spectrogram')
                    plt.savefig("./checkpoints/inference.png")
                    plt.close()
                    print("Mel-spectrogram saved to ./checkpoints/inference.png")

                with torch.no_grad():
                    audio = hifigan(predicted_mel_spectogram).squeeze(0).cpu().numpy()
                audio = audio / np.max(np.abs(audio))
                outputfile = "./checkpoints/audio.wav"
                sf.write(outputfile,audio,samplerate=22050)
                print("Audio saved")
                
            except Exception as e:
                 print(f"Error during inference: {e}")  
            
if __name__ == "__main__":
    inferenceModel()
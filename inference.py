# inference.py
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

g2p = G2p()

h = None

print("Phoneme map:", phoneme_map)

def convert_to_phonemes(text, phonememap, device):
    phonemes = g2p(text)
    #phonemes = [p.strip() for p in phonemes if p.strip() and p not in [".", ",", "!", "?"]]
    phonemes = ['P', 'R', 'IH1', 'N', 'T', 'IH0', 'NG', 'sp', 'IH1', 'N', 'DH', 'IY0', 'OW1', 'N', 'L', 'IY0', 'S', 'EH1', 'N', 'S', 'W', 'IH1', 'DH', 'sp', 'W', 'IH1', 'CH', 'W', 'IY1', 'AA1', 'R', 'AE1', 'T', 'P', 'R', 'EH1', 'Z', 'AH0', 'N', 'T', 'K', 'AH0', 'N', 'S', 'ER1', 'N', 'D', 'sp', 'D', 'IH1', 'F', 'ER0', 'Z', 'sp', 'F', 'R', 'AH1', 'M', 'M', 'OW1', 'S', 'T', 'IH1', 'F', 'N', 'AA1', 'T', 'F', 'R', 'AH1', 'M', 'AO1', 'L', 'DH', 'IY0', 'AA1', 'R', 'T', 'S', 'AH0', 'N', 'D', 'K', 'R', 'AE1', 'F', 'T', 'S', 'R', 'EH2', 'P', 'R', 'IH0', 'Z', 'EH1', 'N', 'T', 'IH0', 'D', 'IH1', 'N', 'DH', 'IY0', 'EH2', 'K', 'S', 'AH0', 'B', 'IH1', 'SH', 'AH0', 'N']
    phonemes_indices = [phonememap.get(p, phonememap["UNK"]) for p in phonemes]
    print("Extracted phonemes:", phonemes)
    print("Extracted phonemes indices:", phonemes_indices)
    phoneme_tensor = torch.tensor(phonemes_indices, dtype=torch.long).unsqueeze(0).to(device=device)
    src_lens = torch.tensor([len(phonemes_indices)], dtype=torch.long).to(device=device)
    return phoneme_tensor, src_lens

vocab_size = len(phoneme_map)
embedding_dim = 256
hidden_dim = 256
n_heads = 8
n_layers = 6
batch_size = 8
output_dim = 80

def expand_predictions(predictions, durations, mel_len, duration_max=100.0):
    """Expand phoneme-level predictions (pitch, energy) to frame-level using durations."""
    batch_size = durations.shape[0]
    expanded_predictions = []
    
    for i in range(batch_size):
        expanded = []
        for j in range(len(durations[i])):
            repeat_count = int(durations[i, j].item() * duration_max)
            expanded.append(predictions[i, j].repeat(repeat_count))
        expanded = torch.cat(expanded, dim=0)
        # Truncate or pad to match mel_len
        if expanded.shape[0] > mel_len[i]:
            expanded = expanded[:mel_len[i]]
        elif expanded.shape[0] < mel_len[i]:
            expanded = torch.cat([expanded, torch.zeros(mel_len[i] - expanded.shape[0], device=expanded.device)], dim=0)
        expanded_predictions.append(expanded)
    
    return torch.stack(expanded_predictions)

def inferenceModel():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    global h
    with open("./LJ_V1/config.json", "r") as f:
        json_config = json.loads(f.read())
    h = AttrDict(json_config)

    model = TransformerTTS(
        vocab_size=vocab_size,
        embedding_dim=embedding_dim,
        hidden_dim=hidden_dim,
        n_heads=n_heads,
        n_layers=n_layers,
        output_dim=output_dim
    )
    model.duration_max = 100.0
    checkpoint = torch.load("./checkpoints/model_epoch_19.pt", map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device=device)
    model.eval()
    print("TTS model is loaded")

    hifigan = HIFIGAN(h)
    hifiganmodel = torch.load("./LJ_V1/generator_v1", map_location=device)
    hifigan.load_state_dict(hifiganmodel["generator"])
    hifigan = hifigan.to(device=device)
    hifigan.eval()
    print("HiFi-GAN model is loaded")

    with open("./mel_min_max.json", 'r') as f:
        melconfig = json.load(f)
        mel_min = torch.tensor(melconfig["mel_min"], dtype=torch.float32).to(device)
        mel_max = torch.tensor(melconfig["mel_max"], dtype=torch.float32).to(device)

    with open("./pitch_min_max.json", 'r') as f:
        pitchconfig = json.load(f)
        pitch_min = torch.tensor(pitchconfig["pitch_min"], dtype=torch.float32).to(device)
        pitch_max = torch.tensor(pitchconfig["pitch_max"], dtype=torch.float32).to(device)

    with open("./energy_min_max.json", 'r') as f:
        energyconfig = json.load(f)
        energy_min = torch.tensor(energyconfig["energy_min"], dtype=torch.float32).to(device)
        energy_max = torch.tensor(energyconfig["energy_max"], dtype=torch.float32).to(device)

    while True:
        text = str(input("Enter the sentence for conversion (or type 'exit' to quit): "))
        if text.lower() == "exit":
            print("Exiting...")
            break
        else:
            try:
                phoneme_indices, src_lens = convert_to_phonemes(text=text, phonememap=phoneme_map, device=device)

                with torch.no_grad():
                    predicted_mel_spectrogram, predictions, mel_len, mel_mask = model(
                        phoneme_indices, src_lens, spectogram=None, durations=None, pitch=None, energy=None
                    )

                    # Extract predicted durations, pitch, and energy
                    log_duration_prediction = predictions["duration"]
                    pitch_prediction = predictions["pitch"]
                    energy_prediction = predictions["energy"]

                    # Compute rounded durations (same as in VarianceAdapter)
                    duration_rounded = torch.clamp(
                        (torch.round(torch.exp(log_duration_prediction) - 1)),
                        min=0,
                    )

                    # Expand pitch and energy to frame-level
                    expanded_pitch = expand_predictions(pitch_prediction, duration_rounded, mel_len, duration_max=model.duration_max)
                    expanded_energy = expand_predictions(energy_prediction, duration_rounded, mel_len, duration_max=model.duration_max)

                    # Denormalize pitch and energy for visualization
                    expanded_pitch = expanded_pitch * (pitch_max - pitch_min) + pitch_min
                    expanded_energy = expanded_energy * (energy_max - energy_min) + energy_min

                    print("Predicted Log Durations:", log_duration_prediction)
                    print("Predicted Durations (rounded):", duration_rounded)
                    print("Expanded Pitch (denormalized):", expanded_pitch)
                    print("Expanded Energy (denormalized):", expanded_energy)
                    print("Mel Lengths:", mel_len)

                    # Denormalize the mel-spectrogram
                    print(f"Pre-denorm Mel Range: Min={predicted_mel_spectrogram.min().item():.2f}, Max={predicted_mel_spectrogram.max().item():.2f}")
                    predicted_mel_spectrogram = predicted_mel_spectrogram * (mel_max - mel_min) + mel_min
                    print(f"Post-denorm Mel Range: Min={predicted_mel_spectrogram.min().item():.2f}, Max={predicted_mel_spectrogram.max().item():.2f}")
                    print(f"Expected Mel Range: Min={mel_min.item():.2f}, Max={mel_max.item():.2f}")

                    predicted_mel_spectrogram = predicted_mel_spectrogram.squeeze(0)  # [T, 80]
                    predicted_mel_spectrogram = predicted_mel_spectrogram.permute(1, 0)  # [80, T]
                    print("Predicted mel-spectrogram shape:", predicted_mel_spectrogram.shape)

                    plt.figure(figsize=(10, 4))
                    librosa.display.specshow(
                        predicted_mel_spectrogram.cpu().numpy(),
                        sr=22050,
                        hop_length=256,
                        x_axis='time',
                        y_axis='mel'
                    )
                    plt.colorbar(format='%+2.0f dB')
                    plt.title('Inference Mel-Spectrogram')
                    plt.savefig("./checkpoints/inference.png")
                    plt.close()
                    print("Mel-spectrogram saved to ./checkpoints/inference.png")

                    audio = hifigan(predicted_mel_spectrogram).squeeze(0).cpu().numpy()
                    audio = audio / np.max(np.abs(audio))
                    outputfile = "./checkpoints/audio.wav"
                    sf.write(outputfile, audio, samplerate=22050)
                    print(f"Audio saved to {outputfile}")

            except Exception as e:
                print(f"Error during inference: {e}")
                import traceback
                traceback.print_exc()

if __name__ == "__main__":
    inferenceModel()
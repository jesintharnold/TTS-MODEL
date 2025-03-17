from model import TransformerTTS
import torch
import torch.nn as nn
import os
import json
import matplotlib.pyplot as plt
from dataset import LJSpeechDataset,melspectogram_max_min
from torch.utils.data import DataLoader
import numpy as np
from torch.nn.utils.rnn import pad_sequence



spectogram_loss=nn.MSELoss()
duration_loss=nn.MSELoss()
best_val_loss = 100000000000

class TTStrain:
    def __init__(self,model,device,train_loader,val_loader,lr=1e-4):
        self.model=model.to(device)
        self.train_data=train_loader
        self.val_data=val_loader
        self.device=device
        self.spectogram_loss=nn.MSELoss()
        self.duration_loss=nn.MSELoss()
        self.optimizer=torch.optim.Adam(self.model.parameters(),lr=lr)
        self.latest_checkpoint = None
        self.train_losses = []
        self.val_losses = []

    def train_epoch(self):
        total_loss=0.0
        for batch_idx,batch in enumerate(self.train_data):
            text = batch["phonemes"].to(self.device)
            spectogram = batch["mel"].to(self.device)
            durations = batch["duration"].to(self.device)
            self.optimizer.zero_grad()
            predicted_spectrogram, predicted_durations = self.model(text, spectogram, durations)
            spectogram_loss = self.spectogram_loss(predicted_spectrogram,spectogram)
            duration_loss = self.duration_loss(predicted_durations,durations)
            
            loss = spectogram_loss+0.1*duration_loss
            loss.backward()
            self.optimizer.step()
            total_loss+=loss.item()
            if (batch_idx + 1) % 10 == 0:
                print(f"Batch {batch_idx + 1}/{len(self.train_data)}, Loss: {loss.item()}")
        return total_loss / len(self.train_data)
    
    def validate(self):
        self.model.eval()
        total_loss = 0.0
        with torch.no_grad():
            for batch in self.val_data:
                text = batch["phonemes"].to(self.device)
                spectogram = batch["mel"].to(self.device)
                durations = batch["duration"].to(self.device)
                predicted_spectrogram, predicted_durations = self.model(text, spectogram, durations)
                spectrogram_loss = self.spectogram_loss(predicted_spectrogram, spectogram)
                duration_loss = self.duration_loss(predicted_durations, durations)
                loss = spectrogram_loss + 0.1*duration_loss
                total_loss += loss.item()
        return total_loss / len(self.val_data)

    def save_checkpoint(self, epoch, save_dir="checkpoints"):
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        if self.latest_checkpoint is not None:
            try:
                os.remove(self.latest_checkpoint)
                print(f"Removed previous checkpoint: {self.latest_checkpoint}")
            except Exception as e:
                print(f"Error removing checkpoint {self.latest_checkpoint}: {e}")
    
        checkpoint_path = os.path.join(save_dir, f"model_epoch_{epoch}.pt")
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict()
        }, checkpoint_path)
        print(f"Checkpoint saved at {checkpoint_path}")
        self.latest_checkpoint = checkpoint_path

    def plot_losses(self, save_dir="checkpoints"):
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        plt.figure(figsize=(10, 6))
        plt.plot(self.train_losses, label='Training Loss')
        plt.plot(self.val_losses, label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training and Validation Losses')
        plt.legend()
        plt.grid(True)
        plot_path = os.path.join(save_dir, 'loss_plot.png')
        plt.savefig(plot_path)
        plt.close()
        print(f"Loss plot saved at {plot_path}")

    def train(self,epoch=10, save_dir="checkpoints"):
        for i in range(epoch):
            train_loss=self.train_epoch()
            val_loss = self.validate()
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            print(f"Epoch - {i} , Train loss - {train_loss} , Val loss - {val_loss}")
            global best_val_loss
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                self.save_checkpoint(i, save_dir)
            self.plot_losses(save_dir)
            
        







metadata_train_path="./LJSPEECH/train.txt"
metadata_val_path="./LJSPEECH/val.txt"
mel_dir="./LJSPEECH/mel"
duration_dir="./LJSPEECH/duration"

def create_phonemes_dict():
   phonemes=set()
   with open(metadata_train_path,'r',encoding='utf-8') as file:
        for line in file:
            _phonemes_=line.strip().split("|")[2].strip('{}')
            for phoneme in _phonemes_.split():
                phonemes.add(phoneme)
   arr=sorted(phonemes)
   phoneme_map={ val:key for key,val in enumerate(arr)}
   phoneme_map["PAD"] = len(arr)
   phoneme_map["UNK"] = len(arr)+1
   return phoneme_map

phoneme_map=create_phonemes_dict()   
def collatefn(batch):
    mels = []
    durations = []
    phonemes = []
    texts = []
    for item in batch:
        mels.append(item['mel'])
        durations.append(item['duration'])
        phonemes.append(item['phonemes'])
        texts.append(item['text'])
    mels_padded = pad_sequence(mels, batch_first=True, padding_value=0)
    durations_padded = pad_sequence(durations, batch_first=True, padding_value=0)
    phonemes_padded = pad_sequence(phonemes, batch_first=True, padding_value=phoneme_map["PAD"])
    return {
        'mel': mels_padded,
        'duration': durations_padded,
        'phonemes': phonemes_padded,
        'text': texts
    }

if __name__ == "__main__":
    
    vocab_size = 200
    embedding_dim = 512
    hidden_dim = 512
    n_heads = 8
    n_layers = 8
    output_dim = 80
    batch_size = 32
    num_epochs = 50
    max_data=200
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TransformerTTS(vocab_size=100,embedding_dim=256,hidden_dim=512,n_heads=8,n_layers=4,output_dim=80)
    optimizer= torch.optim.Adam(model.parameters(),lr=1e-4)

    with open("./mel_min_max.json",'r') as f:
        melconfig = json.load(f)
        mel_min=melconfig["mel_min"]
        mel_max=melconfig["mel_max"]


    train_dataset=LJSpeechDataset(metadata_path=metadata_train_path,mel_dir=mel_dir,duration_dir=duration_dir,phoneme_dict=phoneme_map,max_data=max_data,mel_min=mel_min,mel_max=mel_max)
    train_loader=DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        collate_fn=collatefn
    )


    for batch in train_dataset:
        print(batch)
        break


    val_dataset=LJSpeechDataset(metadata_path=metadata_val_path,mel_dir=mel_dir,duration_dir=duration_dir,phoneme_dict=phoneme_map,max_data=max_data//2,mel_min=mel_min,mel_max=mel_max)
    val_loader=DataLoader(
        dataset=val_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        collate_fn=collatefn
    )
    trainer = TTStrain(model=model,device=device,train_loader=train_loader,val_loader=val_loader,lr=1e-4)
    trainer.train(epoch=num_epochs)
    
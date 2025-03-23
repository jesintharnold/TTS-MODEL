from model import TransformerTTS,get_mask_from_lengths
import torch
import torch.nn as nn
import os
import json
import matplotlib.pyplot as plt
from dataset import LJSpeechDataset,melspectogram_max_min
from torch.utils.data import DataLoader
import numpy as np
from torch.nn.utils.rnn import pad_sequence
from torch.utils.tensorboard import SummaryWriter
import datetime

best_val_loss = 100000000000

class TTStrain:
    def __init__(self,model,device,train_loader,val_loader,lr=1e-4):
        self.model=model.to(device)
        self.train_data=train_loader
        self.val_data=val_loader
        self.device=device
        
        self.optimizer=torch.optim.Adam(self.model.parameters(),lr=lr)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=10, gamma=0.5)
        self.latest_checkpoint = None
        self.train_losses = []
        self.val_losses = []

        self.duration_loss=nn.MSELoss(reduction='none')
        self.mse_loss = nn.MSELoss(reduction='none')
        self.l1_loss = nn.L1Loss(reduction='none')

        current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        log_dir = os.path.join("./tensorboard-exp", f"experiment_{current_time}")
        self.writer = SummaryWriter(log_dir=log_dir)
        self.global_step = 0

    def train_epoch(self):
        self.model.train()
        total_loss=0.0
        for batch_idx,batch in enumerate(self.train_data):
            text = batch["phonemes"].to(self.device)
            spectogram = batch["mel"].to(self.device)
            durations = batch["duration"].to(self.device)
            src_lens = batch["src_lens"].to(self.device)
            self.optimizer.zero_grad()
            predicted_spectrogram, predicted_durations, mel_len, mel_mask = self.model(text, src_lens, spectogram, durations)

            if batch_idx == 0:
                print(f"Predicted durations (first sample): {predicted_durations[0].cpu().detach().numpy()}")
                print(f"Ground truth durations (first sample): {durations[0].cpu().numpy()}")
            
            mel_mask_bool = ~mel_mask
            mel_mask_bool = mel_mask_bool.unsqueeze(-1)

            mse_loss = self.mse_loss(predicted_spectrogram, spectogram)
            l1_loss = self.l1_loss(predicted_spectrogram, spectogram)
            mse_loss = mse_loss.masked_select(mel_mask_bool).mean()
            l1_loss = l1_loss.masked_select(mel_mask_bool).mean()
            spectogram_loss = 0.5 * mse_loss + 0.5 * l1_loss

            src_mask = get_mask_from_lengths(src_lens, text.shape[1]).to(self.device)
            duration_mask = ~src_mask  # True for non-padded positions
            duration_loss = self.duration_loss(predicted_durations, durations)
            duration_loss = duration_loss.masked_select(duration_mask).mean()            

            loss = spectogram_loss + 5.0 * duration_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            total_loss+=loss.item()


            self.writer.add_scalar("Loss/Train/Batch_Total", loss.item(), self.global_step)
            self.writer.add_scalar("Loss/Train/Batch_Spectrogram", spectogram_loss.item(), self.global_step)
            self.writer.add_scalar("Loss/Train/Batch_Duration", duration_loss.item(), self.global_step)
            self.writer.add_scalar("Learning_Rate", self.optimizer.param_groups[0]['lr'], self.global_step)
            self.global_step += 1

            if (batch_idx + 1) % 10 == 0:
                print(f"Batch {batch_idx + 1}/{len(self.train_data)}, Loss: {loss.item()}")
        return total_loss / len(self.train_data)
    
    def validate(self):
        self.model.eval()
        total_loss = 0.0
        with torch.no_grad():
            for batch_idx, batch in enumerate(self.val_data):
                text = batch["phonemes"].to(self.device)
                spectogram = batch["mel"].to(self.device)
                durations = batch["duration"].to(self.device)
                src_lens = batch["src_lens"].to(self.device)
                predicted_spectrogram, predicted_durations, mel_len, mel_mask = self.model(text, src_lens, spectogram, durations)
                
                mel_mask_bool = ~mel_mask
                mel_mask_bool = mel_mask_bool.unsqueeze(-1)

                mse_loss = self.mse_loss(predicted_spectrogram, spectogram)
                l1_loss = self.l1_loss(predicted_spectrogram, spectogram)
                mse_loss = mse_loss.masked_select(mel_mask_bool).mean()
                l1_loss = l1_loss.masked_select(mel_mask_bool).mean()
                spectogram_loss = 0.5 * mse_loss + 0.5 * l1_loss
                
                src_mask = get_mask_from_lengths(src_lens, text.shape[1]).to(self.device)
                duration_mask = ~src_mask
                duration_loss = self.duration_loss(predicted_durations, durations)
                duration_loss = duration_loss.masked_select(duration_mask).mean()

                loss = spectogram_loss + 5.0 * duration_loss
                total_loss += loss.item()


                val_step = self.global_step + batch_idx
                self.writer.add_scalar("Loss/Val/Batch_Total", loss.item(), val_step)
                self.writer.add_scalar("Loss/Val/Batch_Spectrogram", spectogram_loss.item(), val_step)
                self.writer.add_scalar("Loss/Val/Batch_Duration", duration_loss.item(), val_step)

                if batch_idx == 0:
                    self._log_visualizations(predicted_spectrogram, spectogram, predicted_durations, durations, mel_mask, val_step)

        return total_loss / len(self.val_data)

    def _log_visualizations(self, predicted_spectrogram, gt_spectrogram, predicted_durations, gt_durations, mel_mask, step):
        # Log mel-spectrograms (first sample in the batch)
        pred_mel = predicted_spectrogram[0].cpu().detach().numpy()  # [max_mel_len, output_dim]
        gt_mel = gt_spectrogram[0].cpu().detach().numpy()  # [max_mel_len, output_dim]
        mel_mask_sample = ~mel_mask[0].cpu().detach().numpy()  # [max_mel_len]

        # Trim the mel-spectrograms based on the mask
        valid_len = mel_mask_sample.sum()
        pred_mel = pred_mel[:valid_len, :]
        gt_mel = gt_mel[:valid_len, :]

        # Create figures for mel-spectrograms
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
        ax1.imshow(gt_mel.T, aspect='auto', origin='lower')
        ax1.set_title("Ground Truth Mel-Spectrogram")
        ax1.set_xlabel("Time Frames")
        ax1.set_ylabel("Mel Bins")
        ax2.imshow(pred_mel.T, aspect='auto', origin='lower')
        ax2.set_title("Predicted Mel-Spectrogram")
        ax2.set_xlabel("Time Frames")
        ax2.set_ylabel("Mel Bins")
        plt.tight_layout()
        self.writer.add_figure("Mel-Spectrogram", fig, step)
        plt.close(fig)

        pred_dur = predicted_durations[0].cpu().detach().numpy()
        gt_dur = gt_durations[0].cpu().detach().numpy()
        seq_len = len(gt_dur)

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 4))
        ax1.plot(range(seq_len), gt_dur, label="Ground Truth")
        ax1.set_title("Ground Truth Durations")
        ax1.set_xlabel("Phoneme Index")
        ax1.set_ylabel("Duration")
        ax1.legend()
        ax2.plot(range(seq_len), pred_dur, label="Predicted", color='orange')
        ax2.set_title("Predicted Durations")
        ax2.set_xlabel("Phoneme Index")
        ax2.set_ylabel("Duration")
        ax2.legend()
        plt.tight_layout()
        self.writer.add_figure("Durations", fig, step)
        plt.close(fig)


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

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(self.train_losses, label='Training Loss')
        ax.plot(self.val_losses, label='Validation Loss')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.set_title('Training and Validation Losses')
        ax.legend()
        ax.grid(True)
        self.writer.add_figure("Loss/Train_vs_Val", fig, len(self.train_losses) - 1)
        plt.close(fig)

    def train(self, epoch=10, save_dir="checkpoints", patience=20):
        best_val_loss = float('inf')
        patience_counter = 0
        for i in range(epoch):
            train_loss = self.train_epoch()
            val_loss = self.validate()
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            print(f"Epoch - {i}, Train loss - {train_loss}, Val loss - {val_loss}")


            self.writer.add_scalar("Loss/Train/Epoch", train_loss, i)
            self.writer.add_scalar("Loss/Val/Epoch", val_loss, i)

            
            if i % 5 == 0:
                for name, param in self.model.named_parameters():
                    self.writer.add_histogram(f"Parameters/{name}", param, i)
                    if param.grad is not None:
                        self.writer.add_histogram(f"Gradients/{name}", param.grad, i)
            
            self.scheduler.step()
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                self.save_checkpoint(i, save_dir)
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"Early stopping at epoch {i} due to no improvement in validation loss.")
                    break
            self.plot_losses(save_dir)
            
        self.writer.close()







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
    src_lens = torch.tensor([len(item['phonemes']) for item in batch], dtype=torch.long)
    return {
        'mel': mels_padded,
        'duration': durations_padded,
        'phonemes': phonemes_padded,
        'text': texts,
        'src_lens': src_lens
    }

if __name__ == "__main__":
    
    vocab_size = len(phoneme_map)
    embedding_dim = 256
    hidden_dim = 256
    n_heads = 8
    n_layers = 6
    batch_size = 8
    num_epochs = 20
    train_max_data=400
    val_max_data=50
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Training the data in device - ",device)

    model = TransformerTTS(vocab_size=vocab_size,embedding_dim=embedding_dim,hidden_dim=hidden_dim,n_heads=n_heads,n_layers=n_layers,output_dim=80)
    optimizer= torch.optim.Adam(model.parameters(),lr=1e-4)

    with open("./mel_min_max.json",'r') as f:
        melconfig = json.load(f)
        mel_min=melconfig["mel_min"]
        mel_max=melconfig["mel_max"]

    train_dataset=LJSpeechDataset(metadata_path=metadata_train_path,mel_dir=mel_dir,duration_dir=duration_dir,phoneme_dict=phoneme_map,max_data=train_max_data,mel_min=mel_min,mel_max=mel_max)
    train_loader=DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        collate_fn=collatefn
    )
    val_dataset=LJSpeechDataset(metadata_path=metadata_val_path,mel_dir=mel_dir,duration_dir=duration_dir,phoneme_dict=phoneme_map,max_data=val_max_data,mel_min=mel_min,mel_max=mel_max)
    val_loader=DataLoader(
        dataset=val_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        collate_fn=collatefn,
        drop_last=True
    )
    trainer = TTStrain(model=model,device=device,train_loader=train_loader,val_loader=val_loader,lr=1e-4)
    trainer.train(epoch=num_epochs)
    
from model import TransformerTTS
import torch
import torch.nn as nn
import os
model = TransformerTTS(vocab_size=100,embedding_dim=256,hidden_dim=512,n_heads=8,n_layers=4,output_dim=80)
optimizer= torch.optim.Adam(model.parameters(),lr=1e-4)
spectogram_loss=nn.MSELoss()
duration_loss=nn.MSELoss()

class TTStrain:
    def __init__(self,model,device,train_loader,val_loader,lr=1e-4):
        self.model=model.to(device)
        self.train=train_loader
        self.val=val_loader
        self.device=device
        self.spectogram_loss=nn.MSELoss()
        self.duration_loss=nn.MSELoss()
        self.optimizer=torch.optim.Adam(self.model.parameters(),lr=lr)

    def train_epoch(self):
        total_loss=0.0
        for batch_idx,(text,spectogram,durations) in enumerate(self.train):
            text = text.to(self.device)
            spectogram = spectogram.to(self.device)
            durations = durations.to(self.device)
            self.optimizer.zero_grad()
            predicted_spectrogram, predicted_durations = self.model(text, spectogram, durations)
            spectogram_loss = self.spectogram_loss(predicted_spectrogram,spectogram)
            duration_loss = self.duration_loss(predicted_durations,durations)
            loss = spectogram_loss+duration_loss
            loss.backward()
            self.optimizer.step()
            total_loss+=loss.item()
            if (batch_idx + 1) % 10 == 0:
                print(f"Batch {batch_idx + 1}/{len(self.train)}, Loss: {loss.item()}")
        return total_loss / len(self.train)
    
    def validate(self):
        self.model.eval()
        total_loss = 0.0
        with torch.no_grad():
            for text, spectrogram, durations in self.val:
                text = text.to(self.device)
                spectrogram = spectrogram.to(self.device)
                durations = durations.to(self.device)
                predicted_spectrogram, predicted_durations = self.model(text, spectrogram, durations)
                spectrogram_loss = self.spectogram_loss(predicted_spectrogram, spectrogram)
                duration_loss = self.duration_loss(predicted_durations, durations)
                loss = spectrogram_loss + duration_loss
                total_loss += loss.item()
        return total_loss / len(self.val)


    def save_checkpoint(self, epoch, save_dir="checkpoints"):
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        checkpoint_path = os.path.join(save_dir, f"model_epoch_{epoch}.pt")
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict()
        }, checkpoint_path)
        print(f"Checkpoint saved at {checkpoint_path}")

    def load_checkpoint(self, checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print(f"Checkpoint loaded from {checkpoint_path}")

    def train(self,epoch=10, save_dir="checkpoints"):
        for i in epoch:
            train_loss=self.train_epoch()
            val_loss = self.validate()
            print(f"Train Loss: {train_loss}")
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                self.save_checkpoint(epoch, save_dir)

    def infer(self, text):
        self.model.eval()
        with torch.no_grad():
            text = torch.tensor(text).unsqueeze(0).to(self.device)
            predicted_spectrogram, _ = self.model(text)
            predicted_spectrogram = predicted_spectrogram.squeeze(0).cpu().numpy()
            return predicted_spectrogram
        

if __name__ == "__main__":
    vocab_size = 100
    embedding_dim = 256
    hidden_dim = 512
    n_heads = 8
    n_layers = 4
    output_dim = 80
    batch_size = 32
    num_epochs = 20
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

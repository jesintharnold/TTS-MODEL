import torch
import torch.nn as nn
import torch.nn.functional as F


# embedding ---> encoder ---> duration predictor --->lenth regulator ---> decoder ---> final output


class DurationPredictor(nn.Module):
    def __init__(self,embedding_dim,hidden_dim):
        super().__init__()
        self.fc1=nn.Linear(embedding_dim,hidden_dim)
        self.act=nn.ReLU()
        self.fc2=nn.Linear(hidden_dim,1)
    def forward(self,x):
        batch_size,seq_len,embedding_dim=x.shape
        x=x.view(batch_size*seq_len,embedding_dim)
        x=self.fc1(x)
        x=self.act(x)
        x=self.fc2(x)
        x=x.view(batch_size,seq_len,1)
        return x


class LengthRegulator(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self,x,durations):
        batch_size, seq_len, embedding_dim = x.shape
        expanded_x=[]
        for i in range(batch_size):
            seq=[]
            for j in range(seq_len):
                seq.append(x[i,j].repeat(int(durations[i,j].item()),1))
            seq=torch.cat(seq,dim=0)
            expanded_x.append(seq)
        expanded_x=torch.stack(expanded_x,dim=0)
        return expanded_x

class TransformerTTS(nn.Module):
    def __init__(self,vocab_size,embedding_dim,hidden_dim,n_heads,n_layers,output_dim):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size,embedding_dim=embedding_dim)    # batches x seq_len x embedding_dim
        self.encoder=nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=embedding_dim,nhead=n_heads,dim_feedforward=hidden_dim),
            num_layers=n_layers
        ) # batches x seq_len x embedding_dim ----> batches*seq_len x 1 ---> batches x seq_len
        self.duration_predictor= DurationPredictor(embedding_dim=embedding_dim,hidden_dim=hidden_dim) # batches*seq_len x embedding_dim x 1
        self.length_regulator = LengthRegulator()
        self.decoder=nn.TransformerDecoder(
            nn.TransformerDecoderLayer(d_model=embedding_dim,nhead=n_heads,dim_feedforward=hidden_dim),
            num_layers=n_layers
        )

        self.fc=nn.Linear(embedding_dim,output_dim)
    
    def forward(self,text,spectogram=None,durations=None):
        embed=self.embedding(text)
        encoder=self.encoder(embed)
        durationpredictor=self.duration_predictor(encoder)

        if durations is not None:
            reg_output = self.length_regulator(encoder,durations.squeeze(-1))
        else:
            reg_output = self.length_regulator(encoder,durationpredictor.squeeze(-1))
        
        if spectogram is not None:
            decoder_output = self.decoder(reg_output,spectogram)
        else:
            decoder_output = self.decoder(reg_output,reg_output)
        
        output = self.fc(decoder_output)
        return output,durationpredictor

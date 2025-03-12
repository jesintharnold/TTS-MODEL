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
        x=x.view(batch_size,seq_len)
        return x


class LengthRegulator(nn.Module):
    def __init__(self,input_dim=None, projection_dim=None):
        super().__init__()
        self.projection = None
        if input_dim is not None and projection_dim is not None:
            self.projection = nn.Linear(input_dim, projection_dim)

    def forward(self,x,durations,target=None):
        batch_size, seq_len, embedding_dim = x.shape
        expanded_x=[]
        max_seq_len=0
        for i in range(batch_size):
            seq=[]
            for j in range(seq_len):
                seq.append(x[i,j].repeat(int(durations[i,j].item()),1))
            seq=torch.cat(seq,dim=0)
            expanded_x.append(seq)

            if seq.size(0) > max_seq_len:
                max_seq_len=seq.size(0)
        for i in range(batch_size):
            seq_len_index = expanded_x[i].size(0)

            if seq_len_index < max_seq_len:
                padding = torch.zeros((max_seq_len-seq_len_index,embedding_dim),device=x.device)
                expanded_x[i] = torch.cat([expanded_x[i], padding], dim=0)
        expanded_x=torch.stack(expanded_x,dim=0)
        if target is not None and self.projection is not None:
            target = self.projection(target)
        return expanded_x,target

class TransformerTTS(nn.Module):
    def __init__(self,vocab_size,embedding_dim,hidden_dim,n_heads,n_layers,output_dim):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size,embedding_dim=embedding_dim)    # batches x seq_len x embedding_dim
        self.encoder=nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=embedding_dim,nhead=n_heads,dim_feedforward=hidden_dim),
            num_layers=n_layers
        ) # batches x seq_len x embedding_dim ----> batches*seq_len x 1 ---> batches x seq_len
        self.duration_predictor= DurationPredictor(embedding_dim=embedding_dim,hidden_dim=hidden_dim) # batches*seq_len x embedding_dim x 1
        self.length_regulator = LengthRegulator(input_dim=output_dim,projection_dim=embedding_dim)
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
            reg_output, projected_spectogram = self.length_regulator(encoder, durations.squeeze(-1), spectogram)
        else:
            reg_output,projected_spectogram = self.length_regulator(encoder,durationpredictor.squeeze(-1),spectogram)
        
        if spectogram is not None:
            decoder_output = self.decoder(projected_spectogram,reg_output)
        else:
            decoder_output = self.decoder(reg_output,reg_output)
        
        output = self.fc(decoder_output)
        return output,durationpredictor

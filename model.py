import torch
import torch.nn as nn
import torch.nn.functional as F


# embedding ---> encoder ---> duration predictor --->lenth regulator ---> decoder ---> final output

class PositionalEncoding(nn.Module):
    def __init__(self, embed_dim, max_len=1000):
        super().__init__()
        self.embed_dim = embed_dim
        self.max_len = max_len
        pe = torch.zeros(max_len, embed_dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2).float() * (-torch.log(torch.tensor(10000.0)) / embed_dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, seq_len):
        return self.pe[:, :seq_len, :]

class RelativePositionalEncoding(nn.Module):
    def __init__(self, embed_dim,num_heads,max_len=1000):
        super().__init__()
        self.embed_dim=embed_dim
        self.max_len=max_len
        self.head_dim = embed_dim // num_heads
        self.relative_positional_embedding=nn.Parameter(
            torch.zeros(2*max_len-1,self.head_dim)  # 2*L-1 , head_dim
        ) 
        nn.init.trunc_normal_(self.relative_positional_embedding, std=0.02)

    def forward(self,seq_len):
        positions = torch.arange(seq_len, dtype=torch.long, device=self.relative_positional_embedding.device).unsqueeze(0)
        positions = positions - positions.transpose(0,1)
        positions+=self.max_len-1
        positions = torch.clamp(positions, 0, 2 * self.max_len - 2)
        rel_pos_encoding=self.relative_positional_embedding[positions]
        return rel_pos_encoding

class RelativeMultiHeadAttention(nn.Module):
    def __init__(self, embed_dim,num_heads):
        super().__init__()
        self.embed_dim=embed_dim
        self.num_heads=num_heads
        self.head_dim=embed_dim//num_heads
        self.scale = self.head_dim ** -0.5
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.rel_pos_encoding = RelativePositionalEncoding(embed_dim,num_heads=num_heads)

    def forward(self,query,key,value,mask=None):
        B,T,C = query.shape
        q = self.q_proj(query).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(key).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(value).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        attn_scores = torch.matmul(q, k.transpose(-2, -1))
        # print("Attention score - 1 shape : ",attn_scores.shape)

        rpe_q=query.permute(1,0,2).contiguous().view(T,B*self.num_heads,self.head_dim)
        rel_pos_embeddings = self.rel_pos_encoding(query.shape[1]).to(q.device) # we are passing the T - sequence length
        rpe=torch.matmul(rpe_q,rel_pos_embeddings.transpose(1,2)).transpose(0,1)  # output --> (batch*heads) seq seq
        rpe = rpe.contiguous().view(B,self.num_heads,query.shape[1],query.shape[1])  # batch heads seq seq
        attn = (attn_scores + rpe)/self.scale

        # print("RPE shape : ",rel_pos_embeddings.shape)
        # print("RPE multi-output shape : ",rpe.shape)
        # print("Attention score - 2 shape : ",attn.shape)
        if mask is not None:
            mask = mask.unsqueeze(1)
            attn = attn.masked_fill(mask, float('-inf'))

        attn_weights = F.softmax(attn, dim=-1)
        output = torch.matmul(attn_weights, v)
        output = output.transpose(1, 2).contiguous().view(B, T, C)
        output = self.out_proj(output)
        return output

class TransformerEncoderLayer(nn.Module):
    def __init__(self, embed_dim, num_heads, hidden_dim,dropout=0.1):
        super().__init__()
        self.self_attn = RelativeMultiHeadAttention(embed_dim, num_heads)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, embed_dim)
        )
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)
    def forward(self, x, src_mask=None):
        attn_output = self.self_attn(x, x, x,src_mask)
        x = x + self.dropout(attn_output)
        x = self.norm1(x)
        ffn_output = self.ffn(x)
        x = x + self.dropout(ffn_output)
        x = self.norm2(x)
        return x
    
class TransformerDecoderLayer(nn.Module):
    def __init__(self, embed_dim, num_heads, hidden_dim,dropout=0.1):
        super().__init__()
        self.self_attn = RelativeMultiHeadAttention(embed_dim, num_heads)
        self.cross_attn = RelativeMultiHeadAttention(embed_dim, num_heads)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, embed_dim)
        )
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.norm3 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, encoder_output,mask=None):
        attn_output = self.self_attn(x, x, x, mask)
        x = x + self.dropout(attn_output)
        x = self.norm1(x)
        cross_attn_output = self.cross_attn(x, encoder_output, encoder_output, mask)
        x = x + self.dropout(cross_attn_output)
        x = self.norm2(x)
        ffn_output = self.ffn(x)
        x = x + self.dropout(ffn_output)
        x = self.norm3(x)

        return x

def get_mask_from_lengths(lengths, max_len=None):
    if max_len is None:
        max_len = lengths.max().item()
    ids = torch.arange(0, max_len, device=lengths.device).unsqueeze(0)
    mask = (ids >= lengths.unsqueeze(1))
    return mask

class DurationPredictor(nn.Module):
    def __init__(self, embedding_dim, hidden_dim):
        super().__init__()
        self.conv_layer = nn.Sequential(
            nn.Conv1d(embedding_dim, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(0.1),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(0.1),
        )
        self.linear_layer = nn.Linear(hidden_dim, 1)
        self.softplus = nn.Softplus()
        self.scale = nn.Parameter(torch.tensor(1.0))

    def forward(self, x, mask=None):
        x = x.transpose(1, 2)
        x = self.conv_layer(x) 
        x = x.transpose(1, 2)
        x = self.linear_layer(x)
        x = self.softplus(x)
        x = x * self.scale
        x = x.squeeze(-1)
        if mask is not None:
            x = x.masked_fill(mask, 0.0)
        
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
        mel_len = []

        for i in range(batch_size):
            seq=[]
            for j in range(seq_len):
                seq.append(x[i,j].repeat(int(durations[i,j].item()),1))
            seq=torch.cat(seq,dim=0)
            expanded_x.append(seq)
            mel_len.append(seq.size(0))

            if seq.size(0) > max_seq_len:
                max_seq_len=seq.size(0)

        for i in range(batch_size):
            seq_len_index = expanded_x[i].size(0)
            if seq_len_index < max_seq_len:
                padding = torch.zeros((max_seq_len-seq_len_index,embedding_dim),device=x.device)
                expanded_x[i] = torch.cat([expanded_x[i], padding], dim=0)

        expanded_x=torch.stack(expanded_x,dim=0)

        mel_len = torch.LongTensor(mel_len).to(x.device)

        mel_mask = get_mask_from_lengths(mel_len, max_seq_len).to(x.device)

        if target is not None and self.projection is not None:
            target = self.projection(target)
        return expanded_x,target,mel_len, mel_mask

class PitchPredictor(nn.Module):
    def __init__(self, embedding_dim, hidden_dim):
        super().__init__()
        self.conv_layer = nn.Sequential(
            nn.Conv1d(embedding_dim, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(0.1),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(0.1),
        )
        self.linear_layer = nn.Linear(hidden_dim, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, mask=None):
        # print("PitchPredictor input shape:", x.shape)
        x = x.transpose(1, 2)
        x = self.conv_layer(x) 
        x = x.transpose(1, 2)
        x = self.linear_layer(x)
        x = self.sigmoid(x)
        x = x.squeeze(-1)
        if mask is not None:
            x = x.masked_fill(mask, 0.0)
        # print("PitchPredictor output shape:", x.shape)
        return x

class EnergyPredictor(nn.Module):
    def __init__(self, embedding_dim, hidden_dim):
        super().__init__()
        self.conv_layer = nn.Sequential(
            nn.Conv1d(embedding_dim, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(0.1),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(0.1),
        )
        self.linear_layer = nn.Linear(hidden_dim, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, mask=None):
        x = x.transpose(1, 2)
        x = self.conv_layer(x) 
        x = x.transpose(1, 2)
        x = self.linear_layer(x)
        x = self.sigmoid(x)
        x = x.squeeze(-1)
        if mask is not None:
            x = x.masked_fill(mask, 0.0)
        return x

class VarianceAdapter(nn.Module):
    def __init__(self,embedding_dim,hidden_dim,mel_output_dim):
        super().__init__()
        self.duration_predictor= DurationPredictor(embedding_dim=embedding_dim,hidden_dim=hidden_dim)
        self.length_regulator = LengthRegulator(input_dim=mel_output_dim,projection_dim=embedding_dim)

        self.pitch_predictor = PitchPredictor(embedding_dim=embedding_dim,hidden_dim=hidden_dim)
        self.energy_predictor = EnergyPredictor(embedding_dim=embedding_dim,hidden_dim=hidden_dim)

        self.pitch_projection = nn.Linear(1, embedding_dim)
        self.energy_projection = nn.Linear(1, embedding_dim)

        self.duration_max = 100.0
        self.pitch_min = 0.0
        self.pitch_max = 1.0
        self.energy_min = 0.0
        self.energy_max = 1.0
    
    def set_min_max(self, predictor, min_val, max_val):
        if predictor == "pitch":
            self.pitch_min = min_val
            self.pitch_max = max_val
        elif predictor == "energy":
            self.energy_min = min_val
            self.energy_max = max_val
    
    def forward(self,encoder_output,src_mask,durations=None,pitch=None,energy=None,spectogram=None):

        # DURATION
        durationpredictor=self.duration_predictor(encoder_output, src_mask)
        if durations is None:
            print("Raw durationpredictor values:", durationpredictor.cpu().detach().numpy())
            durations = torch.clamp(durationpredictor, min=0.01, max=1.0) 
            durations = durations * self.duration_max
            durations = torch.clamp(durations, min=1.0, max=100.0)
            durations = torch.round(durations).int()
        else:
            durations = durations * self.duration_max
        durations = durations.squeeze(-1)  

        # LENGTH REGULATOR
        reg_output, projected_spectogram, mel_len, mel_mask = self.length_regulator(encoder_output, durations, spectogram)

        layer_reg_out = reg_output if spectogram is None else projected_spectogram

        # PITCH PREDICTOR
        pitchpredictor = self.pitch_predictor(layer_reg_out,mel_mask)
        if pitch is None:
            pitch = torch.clamp(pitchpredictor, min=0.01, max=1.0)
            pitch = pitch * (self.pitch_max - self.pitch_min) + self.pitch_min
        else:
            pitch = pitch * (self.pitch_max - self.pitch_min) + self.pitch_min
        pitch_expanded = pitch.unsqueeze(-1)
        # print("Layer Regulator output : ",layer_reg_out.shape)
        # print("Expanded pitch prediction :",self.pitch_projection(pitch_expanded).shape)
        pitch_output_expanded = layer_reg_out + self.pitch_projection(pitch_expanded)

        #ENERGY PREDICTOR
        energypredictor = self.energy_predictor(pitch_output_expanded,mel_mask)
        if energy is None:
            energy = torch.clamp(energypredictor, min=0.01, max=1.0)
            energy = energy * (self.energy_max - self.energy_min) + self.energy_min
        else:
            energy = energy * (self.energy_max - self.energy_min) + self.energy_min
        
        energy_expanded = energy.unsqueeze(-1)
        energy_output_expanded = pitch_output_expanded + self.energy_projection(energy_expanded)

        predictions = {
            "duration":durationpredictor,
            "pitch": pitchpredictor,
            "energy": energypredictor
        }

        return energy_output_expanded,predictions,mel_len,mel_mask

class PostNet(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=5):
        super().__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, padding=kernel_size//2),
            nn.BatchNorm1d(out_channels),
            nn.Tanh(),
            nn.Dropout(0.1),
            nn.Conv1d(out_channels, out_channels, kernel_size=kernel_size, padding=kernel_size//2),
            nn.BatchNorm1d(out_channels),
            nn.Tanh(),
            nn.Dropout(0.1),
            nn.Conv1d(out_channels, out_channels, kernel_size=kernel_size, padding=kernel_size//2),
        )

    def forward(self, x):
        return self.conv_layers(x)

class TransformerTTS(nn.Module):
    def __init__(self,vocab_size,embedding_dim,hidden_dim,n_heads,n_layers,output_dim):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size,embedding_dim=embedding_dim)

        #Implement Positional encoding here bro
        self.positional_encoding = PositionalEncoding(embed_dim=embedding_dim, max_len=1000)

        self.encoder = nn.ModuleList([TransformerEncoderLayer(embedding_dim, n_heads, hidden_dim) for _ in range(n_layers)])
        self.decoder = nn.ModuleList([TransformerDecoderLayer(embedding_dim, n_heads, hidden_dim) for _ in range(n_layers)])

        self.variance_adapter = VarianceAdapter(embedding_dim=embedding_dim,hidden_dim=hidden_dim,mel_output_dim=output_dim)

        self.fc=nn.Linear(embedding_dim,output_dim)
        self.duration_max = 100.0
    
    def forward(self, text, src_lens, spectogram=None, durations=None, pitch=None, energy=None):
        embed=self.embedding(text)

        pos_encoding = self.positional_encoding(embed.shape[1]).to(embed.device)
        pos_encoding = pos_encoding.expand(embed.shape[0], -1, -1)
        embed = embed + pos_encoding

        T = text.shape[1]
        src_mask = get_mask_from_lengths(src_lens, T).to(embed.device)

        encoder_output=embed

        for layer in self.encoder:
            encoder_output = layer(encoder_output,src_mask)

        variance_adapter_output, predictions, mel_len, mel_mask = self.variance_adapter(encoder_output,src_mask,durations,pitch,energy,spectogram)

        max_mel_len = variance_adapter_output.shape[1]
        pos_encoding_dec = self.positional_encoding(max_mel_len).to(variance_adapter_output.device)
        pos_encoding_dec = pos_encoding_dec.expand(variance_adapter_output.shape[0], -1, -1)
        decoder_input = variance_adapter_output + pos_encoding_dec

        for layer in self.decoder:
            decoder_output = layer(decoder_input, variance_adapter_output, mel_mask)

        output = self.fc(decoder_output)
        return output, predictions, mel_len, mel_mask




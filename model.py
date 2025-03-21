import torch
import torch.nn as nn
import torch.nn.functional as F


# embedding ---> encoder ---> duration predictor --->lenth regulator ---> decoder ---> final output



class RelativePositionalEncoding(nn.Module):
    def __init__(self, embed_dim,num_heads,max_len=1000):
        super().__init__()
        self.embed_dim=embed_dim
        self.max_len=max_len
        self.relative_positional_embedding=nn.Parameter(
            torch.zeros(2*max_len-1,embed_dim // num_heads)  # 2*L-1 , head_dim
        ) 
        nn.init.trunc_normal_(self.relative_positional_embedding, std=0.02)

    def forward(self,seq_len):
        positions = torch.arange(seq_len,dtype=torch.long).unsqueeze(0)
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

    def forward(self,query,key,value):
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
    def forward(self, x):
        attn_output = self.self_attn(x, x, x)
        x = x + attn_output
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

    def forward(self, x, encoder_output):
        attn_output = self.self_attn(x, x, x)
        x = x + attn_output
        x = self.norm1(x)
        cross_attn_output = self.cross_attn(x, encoder_output, encoder_output)
        x = x + cross_attn_output
        x = self.norm2(x)
        ffn_output = self.ffn(x)
        x = x + self.dropout(ffn_output)
        x = self.norm3(x)

        return x

class DurationPredictor(nn.Module):
    def __init__(self,embedding_dim,hidden_dim):
        super().__init__()
        self.fc1=nn.Linear(embedding_dim,hidden_dim)
        self.act=nn.ReLU()
        self.fc2=nn.Linear(hidden_dim,1)
        self.softplus = nn.Softplus()
    def forward(self,x):
        batch_size,seq_len,embedding_dim=x.shape
        x=x.view(batch_size*seq_len,embedding_dim)
        x=self.fc1(x)
        x=self.act(x)
        x=self.fc2(x)
        x = self.softplus(x)
        x=x.view(batch_size,seq_len)
        return x

class LengthRegulator(nn.Module):
    def __init__(self, input_dim=None, projection_dim=None):
        super().__init__()
        self.projection = None
        if input_dim is not None and projection_dim is not None:
            self.projection = nn.Linear(input_dim, projection_dim)

    def forward(self, x, durations, target=None):
        batch_size, seq_len, embedding_dim = x.shape
        durations = durations.long()  # (batch_size, seq_len)
        durations = torch.clamp(durations, min=1)
        expanded_x_list = []
        for i in range(batch_size):
            expanded = torch.repeat_interleave(x[i], durations[i], dim=0)
            expanded_x_list.append(expanded)
        expanded_x = torch.nn.utils.rnn.pad_sequence(expanded_x_list, batch_first=True, padding_value=0.0)
        if target is not None and self.projection is not None:
            target = self.projection(target)
            target_max_len = expanded_x.shape[1]
            if target.shape[1] < target_max_len:
                padding = torch.zeros((batch_size, target_max_len - target.shape[1], embedding_dim), device=x.device)
                target = torch.cat([target, padding], dim=1)
            elif target.shape[1] > target_max_len:
                target = target[:, :target_max_len, :]
        return expanded_x, target

class TransformerTTS(nn.Module):
    def __init__(self,vocab_size,embedding_dim,hidden_dim,n_heads,n_layers,output_dim):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size,embedding_dim=embedding_dim)    # batches x seq_len x embedding_dim
       
        self.encoder = nn.ModuleList([TransformerEncoderLayer(embedding_dim, n_heads, hidden_dim) for _ in range(n_layers)]) # batches x seq_len x embedding_dim ----> batches*seq_len x 1 ---> batches x seq_len
        self.decoder = nn.ModuleList([TransformerDecoderLayer(embedding_dim, n_heads, hidden_dim) for _ in range(n_layers)])
       
        self.duration_predictor= DurationPredictor(embedding_dim=embedding_dim,hidden_dim=hidden_dim) # batches*seq_len x embedding_dim x 1
        self.length_regulator = LengthRegulator(input_dim=output_dim,projection_dim=embedding_dim)
        
        self.fc=nn.Linear(embedding_dim,output_dim)
    
    def forward(self,text,spectogram=None,durations=None):
        embed=self.embedding(text)
        encoder_output=embed
        for layer in self.encoder:  # Stack encoder layers
            encoder_output = layer(encoder_output)
        durationpredictor=self.duration_predictor(encoder_output)
        if durations is None:
            durations = torch.clamp(durationpredictor, min=1.0)
            durations = torch.round(durations).int()
        durations = durations.squeeze(-1)    
        reg_output, projected_spectogram = self.length_regulator(encoder_output, durations.squeeze(-1), spectogram)
        decoder_output = reg_output if spectogram is None else projected_spectogram
        for layer in self.decoder:
            decoder_output = layer(decoder_output,reg_output)
        output = self.fc(decoder_output)
        return output,durationpredictor




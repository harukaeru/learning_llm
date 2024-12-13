import torch
import torch.nn as nn
import math

###############################################################
# スケーリングド・ドットプロダクト・アテンション
###############################################################
def scaled_dot_product_attention(Q, K, V, mask=None):
    # Q, K, V: [batch_size, heads, seq_len, dim_head]
    d_k = Q.size(-1)
    scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)  # [B, heads, seq_len, seq_len]

    if mask is not None:
        # mask: [batch_size, 1, 1, seq_len]など
        scores = scores.masked_fill(mask == 0, float('-inf'))

    attn = torch.softmax(scores, dim=-1)  # [B, heads, seq_len, seq_len]
    out = torch.matmul(attn, V)  # [B, heads, seq_len, dim_head]
    return out, attn

###############################################################
# Multi-Head Attention
###############################################################
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.W_Q = nn.Linear(d_model, d_model)
        self.W_K = nn.Linear(d_model, d_model)
        self.W_V = nn.Linear(d_model, d_model)
        self.out = nn.Linear(d_model, d_model)

    def forward(self, Q, K, V, mask=None):
        # Q, K, V: [batch_size, seq_len, d_model]
        B, seq_len, _ = Q.size()

        # Linear projections
        Q = self.W_Q(Q).view(B, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        K = self.W_K(K).view(B, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        V = self.W_V(V).view(B, seq_len, self.num_heads, self.d_k).transpose(1, 2)

        # Scaled Dot-Product Attention
        out, attn = scaled_dot_product_attention(Q, K, V, mask)

        # Concat heads
        out = out.transpose(1, 2).contiguous().view(B, seq_len, self.d_model)
        out = self.out(out)  # [B, seq_len, d_model]

        return out, attn

###############################################################
# Position-wise Feed Forward Network
###############################################################
class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.fc2(self.dropout(torch.relu(self.fc1(x))))
        return x

###############################################################
# Positional Encoding
###############################################################
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        self.pe = pe.unsqueeze(0)  # [1, max_len, d_model]

    def forward(self, x):
        # x: [B, seq_len, d_model]
        seq_len = x.size(1)
        x = x + self.pe[:, :seq_len].to(x.device)
        return x

###############################################################
# Encoder Layer
###############################################################
class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super(EncoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        # Self-attention
        attn_out, _ = self.self_attn(x, x, x, mask)
        x = x + self.dropout(attn_out)
        x = self.norm1(x)

        # Feed Forward
        ff_out = self.feed_forward(x)
        x = x + self.dropout(ff_out)
        x = self.norm2(x)
        return x

###############################################################
# Decoder Layer
###############################################################
class DecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super(DecoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.cross_attn = MultiHeadAttention(d_model, num_heads)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, enc_out, tgt_mask=None, memory_mask=None):
        # Masked Self-Attention
        attn_out, _ = self.self_attn(x, x, x, tgt_mask)
        x = x + self.dropout(attn_out)
        x = self.norm1(x)

        # Cross-Attention with encoder output
        attn_out, _ = self.cross_attn(x, enc_out, enc_out, memory_mask)
        x = x + self.dropout(attn_out)
        x = self.norm2(x)

        # Feed Forward
        ff_out = self.feed_forward(x)
        x = x + self.dropout(ff_out)
        x = self.norm3(x)
        return x

###############################################################
# Encoder 全体
###############################################################
class Encoder(nn.Module):
    def __init__(self, src_vocab_size, d_model, num_heads, d_ff, num_layers, max_len=5000, dropout=0.1):
        super(Encoder, self).__init__()
        self.embedding = nn.Embedding(src_vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model, max_len)
        self.layers = nn.ModuleList([EncoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)])
        self.dropout = nn.Dropout(dropout)

    def forward(self, src, src_mask=None):
        # src: [B, src_len]
        x = self.embedding(src)  # [B, src_len, d_model]
        x = self.pos_encoding(x)
        x = self.dropout(x)

        for layer in self.layers:
            x = layer(x, src_mask)
        return x

###############################################################
# Decoder 全体
###############################################################
class Decoder(nn.Module):
    def __init__(self, tgt_vocab_size, d_model, num_heads, d_ff, num_layers, max_len=5000, dropout=0.1):
        super(Decoder, self).__init__()
        self.embedding = nn.Embedding(tgt_vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model, max_len)
        self.layers = nn.ModuleList([DecoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)])
        self.dropout = nn.Dropout(dropout)

    def forward(self, tgt, enc_out, tgt_mask=None, memory_mask=None):
        # tgt: [B, tgt_len]
        x = self.embedding(tgt)  # [B, tgt_len, d_model]
        x = self.pos_encoding(x)
        x = self.dropout(x)

        for layer in self.layers:
            x = layer(x, enc_out, tgt_mask, memory_mask)
        return x

###############################################################
# Transformer Model
###############################################################
class Transformer(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, d_model=512, num_heads=8, d_ff=2048, num_layers=6, dropout=0.1, max_len=5000):
        super(Transformer, self).__init__()
        self.encoder = Encoder(src_vocab_size, d_model, num_heads, d_ff, num_layers, max_len, dropout)
        self.decoder = Decoder(tgt_vocab_size, d_model, num_heads, d_ff, num_layers, max_len, dropout)
        self.fc_out = nn.Linear(d_model, tgt_vocab_size)

    def forward(self, src, tgt, src_mask=None, tgt_mask=None, memory_mask=None):
        enc_out = self.encoder(src, src_mask)
        dec_out = self.decoder(tgt, enc_out, tgt_mask, memory_mask)
        out = self.fc_out(dec_out)  # [B, tgt_len, tgt_vocab_size]
        return out

###############################################################
# 簡易的な動作テスト
###############################################################
if __name__ == "__main__":
    # パラメータ
    src_vocab_size = 1000
    tgt_vocab_size = 1000
    batch_size = 2
    src_len = 10
    tgt_len = 10

    model = Transformer(src_vocab_size, tgt_vocab_size, d_model=128, num_heads=4, d_ff=256, num_layers=2)
    src = torch.randint(0, src_vocab_size, (batch_size, src_len))
    tgt = torch.randint(0, tgt_vocab_size, (batch_size, tgt_len))

    # 通常、tgt_maskは未来方向のトークンをマスクするために用いる。
    # ここでは簡略化のため未使用。
    out = model(src, tgt)
    print(out.size())  # [2, 10, 1000]


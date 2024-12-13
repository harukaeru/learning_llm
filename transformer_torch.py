import torch
import torch.nn as nn
import math
from torch.utils.data import Dataset, DataLoader

###############################################################
# 前回定義したモデルを利用するためのクラスを再掲
###############################################################
def scaled_dot_product_attention(Q, K, V, mask=None):
    d_k = Q.size(-1)
    scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)

    if mask is not None:
        scores = scores.masked_fill(mask == float('-inf'), float('-inf'))

    attn = torch.softmax(scores, dim=-1)
    out = torch.matmul(attn, V)
    return out, attn

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
        # Q, K, V: [B, seq_len, d_model] (seq_lenはそれぞれ異なる可能性がある)
        B, seq_len_q, _ = Q.size()
        B, seq_len_k, _ = K.size()
        B, seq_len_v, _ = V.size()

        # Linear projections
        Q = self.W_Q(Q).view(B, seq_len_q, self.num_heads, self.d_k).transpose(1, 2)  # [B, heads, seq_len_q, d_k]
        K = self.W_K(K).view(B, seq_len_k, self.num_heads, self.d_k).transpose(1, 2)  # [B, heads, seq_len_k, d_k]
        V = self.W_V(V).view(B, seq_len_v, self.num_heads, self.d_k).transpose(1, 2)  # [B, heads, seq_len_v, d_k]

        # Scaled Dot-Product Attention
        out, attn = scaled_dot_product_attention(Q, K, V, mask)  # out: [B, heads, seq_len_q, d_k]

        # headsとd_kを結合して元のd_modelに戻す
        out = out.transpose(1, 2).contiguous().view(B, seq_len_q, self.d_model)  # [B, seq_len_q, d_model]

        out = self.out(out)  # [B, seq_len_q, d_model]
        return out, attn

class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.fc2(self.dropout(torch.relu(self.fc1(x))))
        return x

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.pe = pe.unsqueeze(0)

    def forward(self, x):
        seq_len = x.size(1)
        x = x + self.pe[:, :seq_len].to(x.device)
        return x

class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super(EncoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        attn_out, _ = self.self_attn(x, x, x, mask)
        x = x + self.dropout(attn_out)
        x = self.norm1(x)

        ff_out = self.feed_forward(x)
        x = x + self.dropout(ff_out)
        x = self.norm2(x)
        return x

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
        attn_out, _ = self.self_attn(x, x, x, tgt_mask)
        x = x + self.dropout(attn_out)
        x = self.norm1(x)

        attn_out, _ = self.cross_attn(x, enc_out, enc_out, memory_mask)
        x = x + self.dropout(attn_out)
        x = self.norm2(x)

        ff_out = self.feed_forward(x)
        x = x + self.dropout(ff_out)
        x = self.norm3(x)
        return x

class Encoder(nn.Module):
    def __init__(self, src_vocab_size, d_model, num_heads, d_ff, num_layers, max_len=5000, dropout=0.1):
        super(Encoder, self).__init__()
        self.embedding = nn.Embedding(src_vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model, max_len)
        self.layers = nn.ModuleList([EncoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)])
        self.dropout = nn.Dropout(dropout)

    def forward(self, src, src_mask=None):
        x = self.embedding(src)
        x = self.pos_encoding(x)
        x = self.dropout(x)
        for layer in self.layers:
            x = layer(x, src_mask)
        return x

class Decoder(nn.Module):
    def __init__(self, tgt_vocab_size, d_model, num_heads, d_ff, num_layers, max_len=5000, dropout=0.1):
        super(Decoder, self).__init__()
        self.embedding = nn.Embedding(tgt_vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model, max_len)
        self.layers = nn.ModuleList([DecoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)])
        self.dropout = nn.Dropout(dropout)

    def forward(self, tgt, enc_out, tgt_mask=None, memory_mask=None):
        x = self.embedding(tgt)
        x = self.pos_encoding(x)
        x = self.dropout(x)
        for layer in self.layers:
            x = layer(x, enc_out, tgt_mask, memory_mask)
        return x

class Transformer(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, d_model=128, num_heads=4, d_ff=256, num_layers=2, dropout=0.1, max_len=5000):
        super(Transformer, self).__init__()
        self.encoder = Encoder(src_vocab_size, d_model, num_heads, d_ff, num_layers, max_len, dropout)
        self.decoder = Decoder(tgt_vocab_size, d_model, num_heads, d_ff, num_layers, max_len, dropout)
        self.fc_out = nn.Linear(d_model, tgt_vocab_size)

    def forward(self, src, tgt, src_mask=None, tgt_mask=None, memory_mask=None):
        enc_out = self.encoder(src, src_mask)
        dec_out = self.decoder(tgt, enc_out, tgt_mask, memory_mask)
        out = self.fc_out(dec_out)
        return out

###############################################################
# 因果的マスクを作成する関数
###############################################################
def generate_subsequent_mask(sz):
    # 下三角行列で後のトークンをマスク
    mask = (torch.triu(torch.ones((sz, sz))) == 1).transpose(0,1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, 0.0)
    return mask

###############################################################
# 簡易トイデータセットの作成
# ここでは "i like apples" "you like oranges" "we all love apples"
# といった短い文を繰り返す。
###############################################################
sentences = [
    "i like apples",
    "you like oranges",
    "we all love apples",
    "they hate bananas",
    "i love oranges"
]

# 単語辞書作成
words = set()
for s in sentences:
    words.update(s.split())
word_list = ["<pad>", "<sos>", "<eos>"] + sorted(list(words))
word2idx = {w: i for i, w in enumerate(word_list)}
idx2word = {i: w for w, i in word2idx.items()}

pad_idx = word2idx["<pad>"]
sos_idx = word2idx["<sos>"]
eos_idx = word2idx["<eos>"]

def encode_sentence(s):
    # <sos>, <eos>を付与
    tokens = [sos_idx] + [word2idx[w] for w in s.split()] + [eos_idx]
    return tokens

encoded_sentences = [encode_sentence(s) for s in sentences]

# すべて同じ長さになるようにpadding（最大長に合わせる）
max_len = max(len(es) for es in encoded_sentences)
for es in encoded_sentences:
    while len(es) < max_len:
        es.append(pad_idx)

# Tensor化
data = torch.tensor(encoded_sentences) # [num_sentences, max_len]

class LM_Dataset(Dataset):
    def __init__(self, data):
        # data: [num_samples, seq_len]
        self.data = data
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        # 一文ずつ取得
        return self.data[idx]

dataset = LM_Dataset(data)
dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

###############################################################
# モデルと学習ループ
###############################################################
src_vocab_size = len(word_list)
tgt_vocab_size = len(word_list)

model = Transformer(src_vocab_size, tgt_vocab_size, d_model=64, num_heads=2, d_ff=128, num_layers=2)
criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

num_epochs = 10

for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    for batch in dataloader:
        # batch: [B, seq_len]
        src = batch
        tgt = batch

        # 言語モデル学習のため、入力と出力を一トークンずらす
        # tgt_in: <sos>から最後-1まで, tgt_out: 最初+1から最後まで
        tgt_in = tgt[:, :-1]
        tgt_out = tgt[:, 1:]

        seq_len = tgt_in.size(1)
        tgt_mask = generate_subsequent_mask(seq_len)  # [seq_len, seq_len]
        tgt_mask = tgt_mask.unsqueeze(0) # broadcasting用 [1, seq_len, seq_len]

        logits = model(src, tgt_in, tgt_mask=tgt_mask)
        # logits: [B, seq_len, vocab_size]
        loss = criterion(logits.transpose(1, 2), tgt_out)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {total_loss/len(dataloader):.4f}")

###############################################################
# 推論(テキスト生成)
# 与えた先頭トークンから続くトークンをモデルが予測する
###############################################################
model.eval()

def greedy_decode(model, src, max_len=10):
    # src: [1, src_len]
    # 最初の入力トークンとして<sos>のみを入れる (言語モデルタスク)
    ys = torch.tensor([[sos_idx]]).to(src.device) # [1,1]

    for i in range(max_len):
        sz = ys.size(1)
        tgt_mask = generate_subsequent_mask(sz).unsqueeze(0)
        out = model(src, ys, tgt_mask=tgt_mask) # [1, sz, vocab_size]
        prob = out[:, -1, :]  # 最後のステップの出力
        next_word = torch.argmax(prob, dim=-1).item()
        ys = torch.cat([ys, torch.tensor([[next_word]])], dim=1)
        if next_word == eos_idx:
            break
    return ys[0].tolist()

# テスト用に "i like" という文章を始点に続く語を予測させる
input_sentence = "i like"
input_tokens = encode_sentence(input_sentence)
input_tokens = input_tokens[:-1] # <eos>はまだいれない
input_tensor = torch.tensor(input_tokens).unsqueeze(0)  # [1, seq_len]

with torch.no_grad():
    output_tokens = greedy_decode(model, input_tensor, max_len=10)
    # output_tokensには<sos>, ..., <eos>が含まれる
    generated_sentence = [idx2word[t] for t in output_tokens]
    print("Generated:", " ".join(generated_sentence))

# 出力例：Generated: <sos> i like apples <eos>
# ※ 学習データが少なすぎるので有用な文法や創造的な生成は期待できないが、
#   一応次のトークンをある程度予測できるようになる。

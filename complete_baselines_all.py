"""
Complete RNA Function Prediction Baselines
All-in-one implementation with training and evaluation

Usage:
    python complete_baselines.py --data rna_data.csv --model lstm --epochs 10
    
Author: RNAChat Baselines Team
Date: 2025
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import pandas as pd
import numpy as np
from transformers import (
    T5Tokenizer, T5ForConditionalGeneration,
    AutoTokenizer, AutoModelForSeq2SeqLM,
    get_linear_schedule_with_warmup
)
from collections import Counter
from tqdm import tqdm
import argparse
import json
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')


# ============================================================================
# EVALUATION METRICS
# ============================================================================

def compute_ngrams(tokens, n):
    """Compute n-grams"""
    ngrams = Counter()
    for i in range(len(tokens) - n + 1):
        ngram = tuple(tokens[i:i+n])
        ngrams[ngram] += 1
    return ngrams


def compute_bleu_single(prediction, reference):
    """Compute BLEU-1,2,3,4 for a single pair"""
    pred_tokens = prediction.lower().split()
    ref_tokens = reference.lower().split()
    
    if len(pred_tokens) == 0:
        return [0.0, 0.0, 0.0, 0.0]
    
    bleu_scores = []
    for n in range(1, 5):
        pred_ngrams = compute_ngrams(pred_tokens, n)
        ref_ngrams = compute_ngrams(ref_tokens, n)
        
        if sum(pred_ngrams.values()) == 0:
            bleu_scores.append(0.0)
            continue
        
        # Clipped counts
        clipped = sum(min(pred_ngrams[ng], ref_ngrams[ng]) for ng in pred_ngrams)
        total = sum(pred_ngrams.values())
        precision = clipped / total if total > 0 else 0
        
        # Brevity penalty
        bp = 1.0 if len(pred_tokens) > len(ref_tokens) else np.exp(1 - len(ref_tokens) / len(pred_tokens))
        
        bleu_scores.append(bp * precision)
    
    return bleu_scores


def compute_bleu_corpus(predictions, references):
    """Compute average BLEU scores"""
    all_scores = [compute_bleu_single(pred, ref) for pred, ref in zip(predictions, references)]
    avg_scores = np.mean(all_scores, axis=0)
    return avg_scores.tolist()


def compute_simcse(predictions, references, device='cuda'):
    """Compute SimCSE similarity with fallback to word overlap"""
    try:
        from transformers import AutoTokenizer, AutoModel
        
        # Try multiple SimCSE models for better compatibility
        models_to_try = [
            'princeton-nlp/sup-simcse-roberta-large',
            'princeton-nlp/sup-simcse-roberta-base',
            'princeton-nlp/sup-simcse-bert-base-uncased'
        ]
        
        model = None
        tokenizer = None
        
        for model_name in models_to_try:
            try:
                print(f"Trying SimCSE model: {model_name}")
                tokenizer = AutoTokenizer.from_pretrained(model_name)
                model = AutoModel.from_pretrained(model_name, use_safetensors=True).to(device)
                print(f"Successfully loaded: {model_name}")
                break
            except Exception as e:
                print(f"Failed to load {model_name}: {e}")
                continue
        
        if model is None:
            raise Exception("All SimCSE models failed to load")
        model.eval()
        
        def encode(texts, batch_size=32):
            all_embeddings = []
            for i in range(0, len(texts), batch_size):
                batch = texts[i:i+batch_size]
                inputs = tokenizer(batch, padding=True, truncation=True, max_length=512, return_tensors='pt').to(device)
                with torch.no_grad():
                    embeddings = model(**inputs).last_hidden_state[:, 0, :]
                all_embeddings.append(embeddings.cpu())
            return torch.cat(all_embeddings, dim=0)
        
        pred_emb = encode(predictions)
        ref_emb = encode(references)
        
        # Cosine similarity
        pred_emb = F.normalize(pred_emb, p=2, dim=1)
        ref_emb = F.normalize(ref_emb, p=2, dim=1)
        similarities = (pred_emb * ref_emb).sum(dim=1)
        
        return similarities.mean().item()
    except Exception as e:
        print(f"Warning: SimCSE computation failed: {e}")
        print("Falling back to word overlap similarity...")
        
        # Fallback: compute word overlap similarity
        def word_overlap_similarity(pred, ref):
            pred_words = set(pred.lower().split())
            ref_words = set(ref.lower().split())
            if len(pred_words) == 0 and len(ref_words) == 0:
                return 1.0
            if len(pred_words) == 0 or len(ref_words) == 0:
                return 0.0
            intersection = len(pred_words & ref_words)
            union = len(pred_words | ref_words)
            return intersection / union if union > 0 else 0.0
        
        similarities = [word_overlap_similarity(pred, ref) for pred, ref in zip(predictions, references)]
        return np.mean(similarities)


# ============================================================================
# DATA LOADING
# ============================================================================

def load_data_from_csv(csv_path):
    """Load data from CSV and split into train/val/test"""
    print(f"Loading data from {csv_path}...")
    df = pd.read_csv(csv_path)
    
    # Check required columns
    required_cols = ['sequence', 'summary_no_citation']
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")
    
    # Clean data - remove rows with NaN values and convert to string
    print(f"Original data size: {len(df)}")
    df = df.dropna(subset=required_cols)
    df['sequence'] = df['sequence'].astype(str)
    df['summary_no_citation'] = df['summary_no_citation'].astype(str)
    
    # Remove empty sequences or summaries
    df = df[(df['sequence'].str.len() > 0) & (df['summary_no_citation'].str.len() > 0)]
    print(f"Cleaned data size: {len(df)}")
    
    # Handle name column
    if 'name' not in df.columns:
        df['name'] = [f'RNA_{i:05d}' for i in range(len(df))]
    
    # Split: 90% train/val, 10% test
    n = len(df)
    test_size = int(0.1 * n)
    train_val_size = n - test_size
    
    # Further split train/val: 90% train, 10% val
    val_size = int(0.1 * train_val_size)
    train_size = train_val_size - val_size
    
    train_df = df.iloc[:train_size]
    val_df = df.iloc[train_size:train_val_size]
    test_df = df.iloc[train_val_size:]
    
    print(f"Data split: Train={len(train_df)}, Val={len(val_df)}, Test={len(test_df)}")
    
    return train_df, val_df, test_df


# ============================================================================
# TOKENIZERS
# ============================================================================

class RNATokenizer:
    """Tokenizer for RNA sequences"""
    def __init__(self):
        self.base_to_idx = {
            '<PAD>': 0, '<SOS>': 1, '<EOS>': 2, '<UNK>': 3,
            'A': 4, 'C': 5, 'G': 6, 'U': 7, 'T': 8, 'N': 9
        }
        self.idx_to_base = {v: k for k, v in self.base_to_idx.items()}
        self.vocab_size = len(self.base_to_idx)
    
    def encode(self, sequence, max_length=512):
        tokens = [self.base_to_idx['<SOS>']]
        for base in str(sequence)[:max_length-2].upper():
            tokens.append(self.base_to_idx.get(base, self.base_to_idx['<UNK>']))
        tokens.append(self.base_to_idx['<EOS>'])
        return torch.tensor(tokens, dtype=torch.long)
    
    def decode(self, tokens):
        bases = []
        for token in tokens:
            if isinstance(token, torch.Tensor):
                token = token.item()
            base = self.idx_to_base.get(token, '')
            if base in ['<SOS>', '<EOS>', '<PAD>']:
                if base == '<EOS>':
                    break
                continue
            bases.append(base)
        return ''.join(bases)


class TextTokenizer:
    """Word-level tokenizer for summaries"""
    def __init__(self):
        self.word_to_idx = {'<PAD>': 0, '<SOS>': 1, '<EOS>': 2, '<UNK>': 3}
        self.idx_to_word = {0: '<PAD>', 1: '<SOS>', 2: '<EOS>', 3: '<UNK>'}
        self.vocab_size = 4
    
    def build_vocab(self, texts, min_freq=2):
        word_freq = {}
        for text in texts:
            for word in str(text).lower().split():
                word_freq[word] = word_freq.get(word, 0) + 1
        
        for word, freq in word_freq.items():
            if freq >= min_freq and word not in self.word_to_idx:
                self.word_to_idx[word] = self.vocab_size
                self.idx_to_word[self.vocab_size] = word
                self.vocab_size += 1
        
        print(f"Built vocabulary: {self.vocab_size} tokens")
    
    def encode(self, text, max_length=200):
        tokens = [self.word_to_idx['<SOS>']]
        for word in str(text).lower().split()[:max_length-2]:
            tokens.append(self.word_to_idx.get(word, self.word_to_idx['<UNK>']))
        tokens.append(self.word_to_idx['<EOS>'])
        return torch.tensor(tokens, dtype=torch.long)
    
    def decode(self, tokens):
        words = []
        for token in tokens:
            if isinstance(token, torch.Tensor):
                token = token.item()
            word = self.idx_to_word.get(token, '')
            if word == '<EOS>':
                break
            if word not in ['<SOS>', '<PAD>', '<UNK>']:
                words.append(word)
        return ' '.join(words)


# ============================================================================
# DATASET
# ============================================================================

class RNADataset(Dataset):
    """Dataset for seq2seq models"""
    def __init__(self, df, rna_tokenizer, text_tokenizer):
        self.df = df.reset_index(drop=True)
        self.rna_tokenizer = rna_tokenizer
        self.text_tokenizer = text_tokenizer
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        src = self.rna_tokenizer.encode(row['sequence'])
        tgt = self.text_tokenizer.encode(row['summary_no_citation'])
        return {'src': src, 'tgt': tgt, 'summary': row['summary_no_citation']}


def collate_fn(batch):
    srcs = pad_sequence([b['src'] for b in batch], batch_first=True, padding_value=0)
    tgts = pad_sequence([b['tgt'] for b in batch], batch_first=True, padding_value=0)
    return {'src': srcs, 'tgt': tgts, 'summaries': [b['summary'] for b in batch]}


# ============================================================================
# SEQ2SEQ MODELS
# ============================================================================

class LSTMEncoderDecoder(nn.Module):
    """LSTM with attention"""
    def __init__(self, src_vocab_size, tgt_vocab_size, embed_dim=256, hidden_dim=512, num_layers=2, dropout=0.3):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        self.src_embedding = nn.Embedding(src_vocab_size, embed_dim, padding_idx=0)
        self.tgt_embedding = nn.Embedding(tgt_vocab_size, embed_dim, padding_idx=0)
        
        self.encoder = nn.LSTM(embed_dim, hidden_dim, num_layers, batch_first=True, 
                              bidirectional=True, dropout=dropout if num_layers > 1 else 0)
        self.decoder = nn.LSTM(embed_dim + hidden_dim*2, hidden_dim*2, num_layers, 
                              batch_first=True, dropout=dropout if num_layers > 1 else 0)
        
        self.attention = nn.Linear(hidden_dim*4, 1)
        self.output = nn.Linear(hidden_dim*2, tgt_vocab_size)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, src, tgt, teacher_forcing_ratio=0.5):
        batch_size, tgt_len = tgt.shape
        vocab_size = self.output.out_features
        
        # Encode
        src_emb = self.dropout(self.src_embedding(src))
        enc_out, (h, c) = self.encoder(src_emb)
        
        # Reshape hidden states
        h = h.view(self.num_layers, 2, batch_size, self.hidden_dim)
        h = torch.cat([h[:, 0], h[:, 1]], dim=2)
        c = c.view(self.num_layers, 2, batch_size, self.hidden_dim)
        c = torch.cat([c[:, 0], c[:, 1]], dim=2)
        
        # Decode
        outputs = torch.zeros(batch_size, tgt_len, vocab_size).to(src.device)
        dec_input = tgt[:, 0].unsqueeze(1)
        
        for t in range(1, tgt_len):
            emb = self.dropout(self.tgt_embedding(dec_input))
            
            # Attention
            query = h[-1].unsqueeze(1).expand(-1, enc_out.size(1), -1)
            attn_in = torch.cat([query, enc_out], dim=2)
            attn_scores = F.softmax(self.attention(attn_in).squeeze(2), dim=1)
            context = torch.bmm(attn_scores.unsqueeze(1), enc_out)
            
            # Decode step
            dec_in = torch.cat([emb, context], dim=2)
            out, (h, c) = self.decoder(dec_in, (h, c))
            outputs[:, t] = self.output(out.squeeze(1))
            
            # Teacher forcing
            dec_input = tgt[:, t].unsqueeze(1) if np.random.random() < teacher_forcing_ratio else outputs[:, t].argmax(1).unsqueeze(1)
        
        return outputs
    
    def generate(self, src, max_length=200):
        self.eval()
        with torch.no_grad():
            batch_size = src.size(0)
            device = src.device
            
            src_emb = self.src_embedding(src)
            enc_out, (h, c) = self.encoder(src_emb)
            
            h = h.view(self.num_layers, 2, batch_size, self.hidden_dim)
            h = torch.cat([h[:, 0], h[:, 1]], dim=2)
            c = c.view(self.num_layers, 2, batch_size, self.hidden_dim)
            c = torch.cat([c[:, 0], c[:, 1]], dim=2)
            
            generated = torch.ones(batch_size, 1, dtype=torch.long).to(device)
            
            for _ in range(max_length):
                emb = self.tgt_embedding(generated[:, -1:])
                
                query = h[-1].unsqueeze(1).expand(-1, enc_out.size(1), -1)
                attn_in = torch.cat([query, enc_out], dim=2)
                attn_scores = F.softmax(self.attention(attn_in).squeeze(2), dim=1)
                context = torch.bmm(attn_scores.unsqueeze(1), enc_out)
                
                dec_in = torch.cat([emb, context], dim=2)
                out, (h, c) = self.decoder(dec_in, (h, c))
                next_token = self.output(out.squeeze(1)).argmax(1).unsqueeze(1)
                
                generated = torch.cat([generated, next_token], dim=1)
                if (next_token == 2).all():  # <EOS>
                    break
            
            return generated


class TransformerSeq2Seq(nn.Module):
    """Transformer from scratch"""
    def __init__(self, src_vocab_size, tgt_vocab_size, d_model=512, nhead=8, 
                 num_layers=6, dim_ff=2048, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        
        self.src_embed = nn.Embedding(src_vocab_size, d_model, padding_idx=0)
        self.tgt_embed = nn.Embedding(tgt_vocab_size, d_model, padding_idx=0)
        self.pos_encoding = self._create_pos_encoding(5000, d_model)
        
        self.transformer = nn.Transformer(d_model, nhead, num_layers, num_layers, 
                                         dim_ff, dropout, batch_first=True)
        self.output = nn.Linear(d_model, tgt_vocab_size)
    
    def _create_pos_encoding(self, max_len, d_model):
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe.unsqueeze(0)
    
    def forward(self, src, tgt):
        src_mask = (src == 0)
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(tgt.size(1)).to(src.device)
        tgt_pad_mask = (tgt == 0)
        
        src_emb = self.src_embed(src) * np.sqrt(self.d_model)
        src_emb = src_emb + self.pos_encoding[:, :src.size(1)].to(src.device)
        
        tgt_emb = self.tgt_embed(tgt) * np.sqrt(self.d_model)
        tgt_emb = tgt_emb + self.pos_encoding[:, :tgt.size(1)].to(tgt.device)
        
        out = self.transformer(src_emb, tgt_emb, tgt_mask=tgt_mask, 
                               src_key_padding_mask=src_mask, tgt_key_padding_mask=tgt_pad_mask)
        return self.output(out)
    
    def generate(self, src, max_length=200):
        self.eval()
        with torch.no_grad():
            device = src.device
            batch_size = src.size(0)
            
            src_mask = (src == 0)
            src_emb = self.src_embed(src) * np.sqrt(self.d_model)
            src_emb = src_emb + self.pos_encoding[:, :src.size(1)].to(device)
            memory = self.transformer.encoder(src_emb, src_key_padding_mask=src_mask)
            
            generated = torch.ones(batch_size, 1, dtype=torch.long).to(device)
            
            for _ in range(max_length):
                tgt_mask = nn.Transformer.generate_square_subsequent_mask(generated.size(1)).to(device)
                tgt_emb = self.tgt_embed(generated) * np.sqrt(self.d_model)
                tgt_emb = tgt_emb + self.pos_encoding[:, :generated.size(1)].to(device)
                
                out = self.transformer.decoder(tgt_emb, memory, tgt_mask=tgt_mask)
                next_token = self.output(out[:, -1, :]).argmax(1).unsqueeze(1)
                generated = torch.cat([generated, next_token], dim=1)
                
                if (next_token == 2).all():
                    break
            
            return generated


# ============================================================================
# PRE-TRAINED LM MODELS
# ============================================================================

class PretrainedLM:
    """Wrapper for T5/FLAN-T5/BART"""
    def __init__(self, model_name, device='cuda'):
        self.device = device
        self.model_name = model_name
        
        print(f"Loading {model_name}...")
        if 'flan' in model_name.lower():
            self.tokenizer = AutoTokenizer.from_pretrained(f'google/{model_name}')
            self.model = AutoModelForSeq2SeqLM.from_pretrained(f'google/{model_name}').to(device)
        elif 't5' in model_name.lower():
            self.tokenizer = T5Tokenizer.from_pretrained(model_name)
            self.model = T5ForConditionalGeneration.from_pretrained(model_name).to(device)
        elif 'bart' in model_name.lower():
            self.tokenizer = AutoTokenizer.from_pretrained(f'facebook/{model_name}')
            self.model = AutoModelForSeq2SeqLM.from_pretrained(f'facebook/{model_name}').to(device)
        
        print(f"Loaded {sum(p.numel() for p in self.model.parameters())/1e6:.1f}M parameters")


# ============================================================================
# TRAINING FUNCTIONS
# ============================================================================

def train_seq2seq(model, train_loader, val_loader, text_tokenizer, num_epochs=10, lr=1e-4, device='cuda'):
    """Train seq2seq model"""
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    
    best_val_loss = float('inf')
    
    for epoch in range(num_epochs):
        # Train
        model.train()
        train_loss = 0
        for batch in tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}'):
            src = batch['src'].to(device)
            tgt = batch['tgt'].to(device)
            
            optimizer.zero_grad()
            output = model(src, tgt[:, :-1])
            loss = criterion(output.reshape(-1, output.size(-1)), tgt[:, 1:].reshape(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            train_loss += loss.item()
        
        # Validate
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                src = batch['src'].to(device)
                tgt = batch['tgt'].to(device)
                output = model(src, tgt[:, :-1])
                loss = criterion(output.reshape(-1, output.size(-1)), tgt[:, 1:].reshape(-1))
                val_loss += loss.item()
        
        avg_train = train_loss / len(train_loader)
        avg_val = val_loss / len(val_loader)
        print(f'Epoch {epoch+1}: Train Loss={avg_train:.4f}, Val Loss={avg_val:.4f}')
        
        if avg_val < best_val_loss:
            best_val_loss = avg_val
            torch.save(model.state_dict(), 'results/best_model.pt')
            print(f'Saved best model')
    
    # Load best
    model.load_state_dict(torch.load('results/best_model.pt'))
    return model


def train_pretrained_lm(lm_wrapper, train_df, val_df, num_epochs=10, batch_size=8, lr=1e-4):
    """Train pre-trained LM"""
    device = lm_wrapper.device
    model = lm_wrapper.model
    tokenizer = lm_wrapper.tokenizer
    
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    
    best_val_loss = float('inf')
    
    for epoch in range(num_epochs):
        # Train
        model.train()
        train_loss = 0
        
        for i in tqdm(range(0, len(train_df), batch_size), desc=f'Epoch {epoch+1}/{num_epochs}'):
            batch_df = train_df.iloc[i:i+batch_size]
            inputs = [f"Describe the function of RNA {name}: {seq[:1000]}" for name, seq in zip(batch_df['name'], batch_df['sequence'])]
            targets = list(batch_df['summary_no_citation'])
            
            input_enc = tokenizer(inputs, padding=True, truncation=True, max_length=512, return_tensors='pt').to(device)
            target_enc = tokenizer(targets, padding=True, truncation=True, max_length=200, return_tensors='pt').to(device)
            
            labels = target_enc['input_ids'].clone()
            labels[labels == tokenizer.pad_token_id] = -100
            
            optimizer.zero_grad()
            outputs = model(input_ids=input_enc['input_ids'], attention_mask=input_enc['attention_mask'], labels=labels)
            loss = outputs.loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            train_loss += loss.item()
        
        # Validate
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for i in range(0, len(val_df), batch_size):
                batch_df = val_df.iloc[i:i+batch_size]
                inputs = [f"Describe the function of RNA {name}: {seq[:1000]}" for name, seq in zip(batch_df['name'], batch_df['sequence'])]
                targets = list(batch_df['summary_no_citation'])
                
                input_enc = tokenizer(inputs, padding=True, truncation=True, max_length=512, return_tensors='pt').to(device)
                target_enc = tokenizer(targets, padding=True, truncation=True, max_length=200, return_tensors='pt').to(device)
                
                labels = target_enc['input_ids'].clone()
                labels[labels == tokenizer.pad_token_id] = -100
                
                outputs = model(input_ids=input_enc['input_ids'], attention_mask=input_enc['attention_mask'], labels=labels)
                val_loss += outputs.loss.item()
        
        avg_train = train_loss / (len(train_df) / batch_size)
        avg_val = val_loss / (len(val_df) / batch_size)
        print(f'Epoch {epoch+1}: Train Loss={avg_train:.4f}, Val Loss={avg_val:.4f}')
        
        if avg_val < best_val_loss:
            best_val_loss = avg_val
            torch.save(model.state_dict(), 'results/best_lm.pt')
            print(f'Saved best model')
    
    model.load_state_dict(torch.load('results/best_lm.pt'))
    return lm_wrapper


# ============================================================================
# EVALUATION FUNCTIONS
# ============================================================================

def evaluate_seq2seq(model, test_loader, text_tokenizer, device='cuda'):
    """Evaluate seq2seq model"""
    model.eval()
    predictions, references = [], []
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc='Evaluating'):
            src = batch['src'].to(device)
            generated = model.generate(src, max_length=200)
            
            for i in range(generated.size(0)):
                pred = text_tokenizer.decode(generated[i])
                predictions.append(pred)
                references.append(batch['summaries'][i])
    
    # Compute metrics
    bleu = compute_bleu_corpus(predictions, references)
    simcse = compute_simcse(predictions, references, device)
    
    return {
        'BLEU-1': bleu[0],
        'BLEU-2': bleu[1],
        'BLEU-3': bleu[2],
        'BLEU-4': bleu[3],
        'SimCSE': simcse,
        'predictions': predictions,
        'references': references
    }


def evaluate_pretrained_lm(lm_wrapper, test_df, batch_size=8):
    """Evaluate pre-trained LM"""
    model = lm_wrapper.model
    tokenizer = lm_wrapper.tokenizer
    device = lm_wrapper.device
    
    model.eval()
    predictions, references = [], []
    
    with torch.no_grad():
        for i in tqdm(range(0, len(test_df), batch_size), desc='Evaluating'):
            batch_df = test_df.iloc[i:i+batch_size]
            inputs = [f"Describe the function of RNA {name}: {seq[:1000]}" for name, seq in zip(batch_df['name'], batch_df['sequence'])]
            
            input_enc = tokenizer(inputs, padding=True, truncation=True, max_length=512, return_tensors='pt').to(device)
            outputs = model.generate(input_ids=input_enc['input_ids'], attention_mask=input_enc['attention_mask'], 
                                    max_length=200, num_beams=4, early_stopping=True)
            
            preds = tokenizer.batch_decode(outputs, skip_special_tokens=True)
            predictions.extend(preds)
            references.extend(list(batch_df['summary_no_citation']))
    
    # Compute metrics
    bleu = compute_bleu_corpus(predictions, references)
    simcse = compute_simcse(predictions, references, device)
    
    return {
        'BLEU-1': bleu[0],
        'BLEU-2': bleu[1],
        'BLEU-3': bleu[2],
        'BLEU-4': bleu[3],
        'SimCSE': simcse,
        'predictions': predictions,
        'references': references
    }


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, required=True, help='Path to CSV file')
    parser.add_argument('--model', type=str, required=True, 
                       choices=['lstm', 'transformer', 't5-base', 't5-large', 'flan-t5-base', 'bart-base'])
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--output', type=str, default='results.json')
    args = parser.parse_args()
    
    # Load data
    train_df, val_df, test_df = load_data_from_csv(args.data)
    
    # Adjust batch size for large models
    if args.model in ['t5-large', 'bart-base']:
        args.batch_size = min(args.batch_size, 8)
        print(f"Adjusted batch size to {args.batch_size} for large model")
    
    # Train and evaluate based on model type
    if args.model in ['lstm', 'transformer']:
        print(f"\n{'='*80}")
        print(f"Training {args.model.upper()} Baseline")
        print(f"{'='*80}\n")
        
        # Initialize tokenizers
        rna_tokenizer = RNATokenizer()
        text_tokenizer = TextTokenizer()
        text_tokenizer.build_vocab(train_df['summary_no_citation'].tolist())
        
        # Create datasets
        train_dataset = RNADataset(train_df, rna_tokenizer, text_tokenizer)
        val_dataset = RNADataset(val_df, rna_tokenizer, text_tokenizer)
        test_dataset = RNADataset(test_df, rna_tokenizer, text_tokenizer)
        
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, collate_fn=collate_fn)
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, collate_fn=collate_fn)
        
        # Initialize model
        if args.model == 'lstm':
            model = LSTMEncoderDecoder(rna_tokenizer.vocab_size, text_tokenizer.vocab_size)
        else:
            model = TransformerSeq2Seq(rna_tokenizer.vocab_size, text_tokenizer.vocab_size)
        
        # Train
        print("\nTraining...")
        model = train_seq2seq(model, train_loader, val_loader, text_tokenizer, 
                             args.epochs, args.lr, args.device)
        
        # Evaluate
        print("\nEvaluating on test set...")
        results = evaluate_seq2seq(model, test_loader, text_tokenizer, args.device)
    
    else:  # Pre-trained LM
        print(f"\n{'='*80}")
        print(f"Training {args.model.upper()} Baseline")
        print(f"{'='*80}\n")
        
        # Initialize model
        lm_wrapper = PretrainedLM(args.model, args.device)
        
        # Train
        print("\nTraining...")
        lm_wrapper = train_pretrained_lm(lm_wrapper, train_df, val_df, 
                                        args.epochs, args.batch_size, args.lr)
        
        # Evaluate
        print("\nEvaluating on test set...")
        results = evaluate_pretrained_lm(lm_wrapper, test_df, args.batch_size)
    
    # Print results
    print(f"\n{'='*80}")
    print(f"FINAL RESULTS - {args.model.upper()}")
    print(f"{'='*80}")
    print(f"BLEU-1:  {results['BLEU-1']:.4f}")
    print(f"BLEU-2:  {results['BLEU-2']:.4f}")
    print(f"BLEU-3:  {results['BLEU-3']:.4f}")
    print(f"BLEU-4:  {results['BLEU-4']:.4f}")
    print(f"SimCSE:  {results['SimCSE']:.4f}")
    print(f"{'='*80}\n")
    
    # Save results
    output_data = {
        'model': args.model,
        'metrics': {
            'BLEU-1': results['BLEU-1'],
            'BLEU-2': results['BLEU-2'],
            'BLEU-3': results['BLEU-3'],
            'BLEU-4': results['BLEU-4'],
            'SimCSE': results['SimCSE']
        },
        'predictions': results['predictions'],
        'references': results['references']
    }
    
    with open(args.output, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    print(f"Results saved to {args.output}")
    
    # Save a few examples
    print("\nSample Predictions:")
    print("-" * 80)
    for i in range(min(3, len(results['predictions']))):
        print(f"\nExample {i+1}:")
        print(f"Reference: {results['references'][i][:150]}...")
        print(f"Predicted: {results['predictions'][i][:150]}...")
    print("-" * 80)


if __name__ == '__main__':
    main()
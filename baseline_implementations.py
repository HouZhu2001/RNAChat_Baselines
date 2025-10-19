"""
RNAChat Baseline Models - Complete Implementation
Author: Research Team
Date: 2025

This file contains all baseline implementations for comparison with RNAChat.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
from transformers import (
    T5Tokenizer, T5ForConditionalGeneration,
    BartTokenizer, BartForConditionalGeneration,
    AutoTokenizer, AutoModelForSeq2SeqLM
)
import editdistance
from typing import List, Dict, Tuple
import json

# ============================================================================
# DATA UTILITIES
# ============================================================================

class RNADataset(Dataset):
    """Unified dataset for all baselines"""
    def __init__(self, data_path, split='train'):
        with open(data_path, 'r') as f:
            self.data = json.load(f)
        self.split = split
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        return {
            'sequence': item['sequence'],
            'name': item['name'],
            'function': item['function'],
            'rna_type': item.get('rna_type', 'unknown')
        }

def extract_kmers(sequence: str, k: int = 3) -> List[str]:
    """Extract k-mer features from RNA sequence"""
    kmers = []
    for i in range(len(sequence) - k + 1):
        kmers.append(sequence[i:i+k])
    return kmers

def sequence_to_kmers_string(sequence: str, k_values: List[int] = [3, 4, 5]) -> str:
    """Convert sequence to space-separated k-mers for TF-IDF"""
    all_kmers = []
    for k in k_values:
        all_kmers.extend(extract_kmers(sequence, k))
    return ' '.join(all_kmers)


# ============================================================================
# 1. TRADITIONAL ML BASELINES
# ============================================================================

class TFIDFRandomForest:
    """TF-IDF + Random Forest baseline"""
    def __init__(self, n_estimators=100, max_depth=20):
        self.vectorizer = TfidfVectorizer(max_features=5000)
        self.classifier = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=42
        )
        self.function_templates = {}
        
    def fit(self, sequences: List[str], functions: List[str]):
        # Convert sequences to k-mer strings
        kmer_docs = [sequence_to_kmers_string(seq) for seq in sequences]
        
        # TF-IDF vectorization
        X = self.vectorizer.fit_transform(kmer_docs)
        
        # Create function categories (simple clustering of similar functions)
        self.function_to_label = {}
        self.label_to_functions = {}
        for i, func in enumerate(set(functions)):
            self.function_to_label[func] = i
            self.label_to_functions[i] = func
            
        y = [self.function_to_label[f] for f in functions]
        
        # Train classifier
        self.classifier.fit(X, y)
        
    def predict(self, sequences: List[str]) -> List[str]:
        kmer_docs = [sequence_to_kmers_string(seq) for seq in sequences]
        X = self.vectorizer.transform(kmer_docs)
        predictions = self.classifier.predict(X)
        
        # Convert predictions back to function descriptions
        return [self.label_to_functions[pred] for pred in predictions]


class TFIDFSVM:
    """TF-IDF + SVM baseline"""
    def __init__(self, C=1.0, kernel='rbf'):
        self.vectorizer = TfidfVectorizer(max_features=5000)
        self.classifier = SVC(C=C, kernel=kernel, random_state=42)
        self.function_to_label = {}
        self.label_to_functions = {}
        
    def fit(self, sequences: List[str], functions: List[str]):
        kmer_docs = [sequence_to_kmers_string(seq) for seq in sequences]
        X = self.vectorizer.fit_transform(kmer_docs)
        
        for i, func in enumerate(set(functions)):
            self.function_to_label[func] = i
            self.label_to_functions[i] = func
            
        y = [self.function_to_label[f] for f in functions]
        self.classifier.fit(X, y)
        
    def predict(self, sequences: List[str]) -> List[str]:
        kmer_docs = [sequence_to_kmers_string(seq) for seq in sequences]
        X = self.vectorizer.transform(kmer_docs)
        predictions = self.classifier.predict(X)
        return [self.label_to_functions[pred] for pred in predictions]


# ============================================================================
# 2. SEQUENCE-ONLY DEEP LEARNING BASELINES
# ============================================================================

class RNATokenizer:
    """Simple RNA sequence tokenizer"""
    def __init__(self):
        self.vocab = {
            '<PAD>': 0, '<SOS>': 1, '<EOS>': 2, '<UNK>': 3,
            'A': 4, 'C': 5, 'G': 6, 'U': 7, 'T': 8, 'N': 9
        }
        self.idx_to_token = {v: k for k, v in self.vocab.items()}
        
    def encode(self, sequence: str, max_len: int = 512) -> torch.Tensor:
        tokens = [self.vocab['<SOS>']]
        for base in sequence[:max_len-2]:
            tokens.append(self.vocab.get(base.upper(), self.vocab['<UNK>']))
        tokens.append(self.vocab['<EOS>'])
        return torch.tensor(tokens)
    
    def decode(self, tokens: torch.Tensor) -> str:
        return ''.join([self.idx_to_token.get(t.item(), '') for t in tokens])


class LSTMEncoderDecoder(nn.Module):
    """LSTM Encoder-Decoder with Attention"""
    def __init__(self, vocab_size=10, embed_dim=256, hidden_dim=512, 
                 num_layers=2, dropout=0.3):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        
        self.encoder = nn.LSTM(
            embed_dim, hidden_dim, num_layers,
            batch_first=True, bidirectional=True, dropout=dropout
        )
        
        self.decoder = nn.LSTM(
            embed_dim, hidden_dim * 2, num_layers,
            batch_first=True, dropout=dropout
        )
        
        # Attention mechanism
        self.attention = nn.Linear(hidden_dim * 4, 1)
        self.output_projection = nn.Linear(hidden_dim * 2, vocab_size)
        
    def forward(self, src, tgt, teacher_forcing_ratio=0.5):
        # Encode
        src_embedded = self.embedding(src)
        encoder_outputs, (hidden, cell) = self.encoder(src_embedded)
        
        # Decode with attention
        batch_size = src.size(0)
        tgt_len = tgt.size(1)
        vocab_size = self.output_projection.out_features
        
        outputs = torch.zeros(batch_size, tgt_len, vocab_size).to(src.device)
        decoder_input = tgt[:, 0].unsqueeze(1)
        
        for t in range(1, tgt_len):
            embedded = self.embedding(decoder_input)
            decoder_output, (hidden, cell) = self.decoder(embedded, (hidden, cell))
            
            # Attention
            attention_scores = self.attention(
                torch.cat([decoder_output.expand_as(encoder_outputs), 
                          encoder_outputs], dim=2)
            )
            attention_weights = F.softmax(attention_scores, dim=1)
            context = torch.sum(attention_weights * encoder_outputs, dim=1, keepdim=True)
            
            # Output
            output = self.output_projection(context)
            outputs[:, t] = output.squeeze(1)
            
            # Teacher forcing
            use_teacher_forcing = np.random.random() < teacher_forcing_ratio
            decoder_input = tgt[:, t].unsqueeze(1) if use_teacher_forcing else output.argmax(2)
            
        return outputs


class GRUEncoderDecoder(nn.Module):
    """GRU Encoder-Decoder (similar to LSTM but with GRU)"""
    def __init__(self, vocab_size=10, embed_dim=256, hidden_dim=512, 
                 num_layers=2, dropout=0.3):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        
        self.encoder = nn.GRU(
            embed_dim, hidden_dim, num_layers,
            batch_first=True, bidirectional=True, dropout=dropout
        )
        
        self.decoder = nn.GRU(
            embed_dim, hidden_dim * 2, num_layers,
            batch_first=True, dropout=dropout
        )
        
        self.attention = nn.Linear(hidden_dim * 4, 1)
        self.output_projection = nn.Linear(hidden_dim * 2, vocab_size)
        
    def forward(self, src, tgt, teacher_forcing_ratio=0.5):
        src_embedded = self.embedding(src)
        encoder_outputs, hidden = self.encoder(src_embedded)
        
        batch_size = src.size(0)
        tgt_len = tgt.size(1)
        vocab_size = self.output_projection.out_features
        
        outputs = torch.zeros(batch_size, tgt_len, vocab_size).to(src.device)
        decoder_input = tgt[:, 0].unsqueeze(1)
        
        for t in range(1, tgt_len):
            embedded = self.embedding(decoder_input)
            decoder_output, hidden = self.decoder(embedded, hidden)
            
            attention_scores = self.attention(
                torch.cat([decoder_output.expand_as(encoder_outputs), 
                          encoder_outputs], dim=2)
            )
            attention_weights = F.softmax(attention_scores, dim=1)
            context = torch.sum(attention_weights * encoder_outputs, dim=1, keepdim=True)
            
            output = self.output_projection(context)
            outputs[:, t] = output.squeeze(1)
            
            use_teacher_forcing = np.random.random() < teacher_forcing_ratio
            decoder_input = tgt[:, t].unsqueeze(1) if use_teacher_forcing else output.argmax(2)
            
        return outputs


class TransformerEncoderDecoder(nn.Module):
    """Transformer Encoder-Decoder from scratch"""
    def __init__(self, vocab_size=10, d_model=512, nhead=8, 
                 num_encoder_layers=6, num_decoder_layers=6, 
                 dim_feedforward=2048, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        
        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        
        self.output_projection = nn.Linear(d_model, vocab_size)
        
    def forward(self, src, tgt):
        src_embedded = self.embedding(src) * np.sqrt(self.d_model)
        src_embedded = self.pos_encoder(src_embedded)
        
        tgt_embedded = self.embedding(tgt) * np.sqrt(self.d_model)
        tgt_embedded = self.pos_encoder(tgt_embedded)
        
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(tgt.size(1)).to(src.device)
        
        output = self.transformer(src_embedded, tgt_embedded, tgt_mask=tgt_mask)
        return self.output_projection(output)


class PositionalEncoding(nn.Module):
    """Positional encoding for Transformer"""
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                            (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


class CNNLSTMHybrid(nn.Module):
    """CNN feature extractor + LSTM decoder"""
    def __init__(self, vocab_size=10, embed_dim=256, hidden_dim=512):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        
        # CNN encoder
        self.conv1 = nn.Conv1d(embed_dim, 256, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(256, 512, kernel_size=3, padding=1)
        self.conv3 = nn.Conv1d(512, 512, kernel_size=3, padding=1)
        self.pool = nn.MaxPool1d(2)
        
        # LSTM decoder
        self.decoder = nn.LSTM(embed_dim, 512, 2, batch_first=True)
        self.attention = nn.Linear(1024, 1)
        self.output_projection = nn.Linear(512, vocab_size)
        
    def forward(self, src, tgt):
        # CNN encoding
        src_embedded = self.embedding(src).transpose(1, 2)
        x = F.relu(self.conv1(src_embedded))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = F.relu(self.conv3(x))
        encoder_outputs = x.transpose(1, 2)
        
        # LSTM decoding
        tgt_embedded = self.embedding(tgt)
        decoder_output, _ = self.decoder(tgt_embedded)
        
        # Attention
        attention_scores = self.attention(
            torch.cat([decoder_output.unsqueeze(2).expand(-1, -1, encoder_outputs.size(1), -1),
                      encoder_outputs.unsqueeze(1).expand(-1, decoder_output.size(1), -1, -1)], 
                     dim=3)
        )
        attention_weights = F.softmax(attention_scores.squeeze(-1), dim=2)
        context = torch.bmm(attention_weights, encoder_outputs)
        
        output = self.output_projection(decoder_output)
        return output


# ============================================================================
# 3. PRE-TRAINED LANGUAGE MODEL BASELINES
# ============================================================================

class FineTunedT5:
    """Fine-tuned T5 baseline"""
    def __init__(self, model_name='t5-base', device='cuda'):
        self.tokenizer = T5Tokenizer.from_pretrained(model_name)
        self.model = T5ForConditionalGeneration.from_pretrained(model_name).to(device)
        self.device = device
        
    def train(self, train_loader, optimizer, num_epochs=10):
        self.model.train()
        for epoch in range(num_epochs):
            for batch in train_loader:
                sequences = batch['sequence']
                functions = batch['function']
                
                # Format input
                inputs = [f"describe RNA function: {seq}" for seq in sequences]
                
                # Tokenize
                input_ids = self.tokenizer(
                    inputs, return_tensors='pt', padding=True, truncation=True
                ).input_ids.to(self.device)
                
                labels = self.tokenizer(
                    functions, return_tensors='pt', padding=True, truncation=True
                ).input_ids.to(self.device)
                
                # Forward pass
                outputs = self.model(input_ids=input_ids, labels=labels)
                loss = outputs.loss
                
                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
    def predict(self, sequences: List[str], max_length=200) -> List[str]:
        self.model.eval()
        inputs = [f"describe RNA function: {seq}" for seq in sequences]
        
        input_ids = self.tokenizer(
            inputs, return_tensors='pt', padding=True, truncation=True
        ).input_ids.to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(input_ids, max_length=max_length)
            
        predictions = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
        return predictions


class FineTunedFLANT5:
    """Fine-tuned FLAN-T5 baseline (instruction-tuned)"""
    def __init__(self, model_name='google/flan-t5-base', device='cuda'):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(device)
        self.device = device
        
    def train(self, train_loader, optimizer, num_epochs=10):
        self.model.train()
        for epoch in range(num_epochs):
            for batch in train_loader:
                sequences = batch['sequence']
                functions = batch['function']
                
                inputs = [f"Describe the biological function of this RNA sequence: {seq}" 
                         for seq in sequences]
                
                input_ids = self.tokenizer(
                    inputs, return_tensors='pt', padding=True, truncation=True
                ).input_ids.to(self.device)
                
                labels = self.tokenizer(
                    functions, return_tensors='pt', padding=True, truncation=True
                ).input_ids.to(self.device)
                
                outputs = self.model(input_ids=input_ids, labels=labels)
                loss = outputs.loss
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
    def predict(self, sequences: List[str], max_length=200) -> List[str]:
        self.model.eval()
        inputs = [f"Describe the biological function of this RNA sequence: {seq}" 
                 for seq in sequences]
        
        input_ids = self.tokenizer(
            inputs, return_tensors='pt', padding=True, truncation=True
        ).input_ids.to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(input_ids, max_length=max_length)
            
        return self.tokenizer.batch_decode(outputs, skip_special_tokens=True)


class FineTunedBART:
    """Fine-tuned BART baseline"""
    def __init__(self, model_name='facebook/bart-base', device='cuda'):
        self.tokenizer = BartTokenizer.from_pretrained(model_name)
        self.model = BartForConditionalGeneration.from_pretrained(model_name).to(device)
        self.device = device
        
    def train(self, train_loader, optimizer, num_epochs=10):
        self.model.train()
        for epoch in range(num_epochs):
            for batch in train_loader:
                sequences = batch['sequence']
                functions = batch['function']
                
                input_ids = self.tokenizer(
                    sequences, return_tensors='pt', padding=True, truncation=True
                ).input_ids.to(self.device)
                
                labels = self.tokenizer(
                    functions, return_tensors='pt', padding=True, truncation=True
                ).input_ids.to(self.device)
                
                outputs = self.model(input_ids=input_ids, labels=labels)
                loss = outputs.loss
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
    def predict(self, sequences: List[str], max_length=200) -> List[str]:
        self.model.eval()
        input_ids = self.tokenizer(
            sequences, return_tensors='pt', padding=True, truncation=True
        ).input_ids.to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(input_ids, max_length=max_length)
            
        return self.tokenizer.batch_decode(outputs, skip_special_tokens=True)


# ============================================================================
# 4. ALTERNATIVE RNA ENCODER BASELINES
# ============================================================================

class RNAFMChat:
    """RNA-FM + Adaptor + Vicuna baseline"""
    def __init__(self, rnafm_model, vicuna_model, device='cuda'):
        """
        Similar to RNAChat but uses RNA-FM instead of RiNALMo
        
        Args:
            rnafm_model: Pre-trained RNA-FM encoder
            vicuna_model: Vicuna-13B model
        """
        self.encoder = rnafm_model.to(device)
        self.generator = vicuna_model.to(device)
        
        # Adaptor layer (RNA-FM output dim -> Vicuna input dim)
        # RNA-FM typically outputs 640-dim, map to 5120 for Vicuna
        self.adaptor = nn.Linear(640, 5120).to(device)
        self.device = device
        
    def encode_rna(self, sequences: List[str]) -> torch.Tensor:
        """Encode RNA sequences using RNA-FM"""
        # Tokenize and encode sequences
        with torch.no_grad():
            embeddings = self.encoder(sequences)  # [batch, seq_len, 640]
        return embeddings
    
    def forward(self, sequences, prompts):
        # Get RNA embeddings
        rna_embeddings = self.encode_rna(sequences)
        
        # Project to LLM space
        projected_embeddings = self.adaptor(rna_embeddings)
        
        # Generate with Vicuna
        outputs = self.generator.generate_with_embeddings(
            projected_embeddings, prompts
        )
        return outputs


class OneHotMLPChat:
    """One-Hot + MLP + Vicuna baseline"""
    def __init__(self, vicuna_model, device='cuda'):
        self.generator = vicuna_model.to(device)
        self.device = device
        
        # MLP encoder: one-hot (4 bases) -> 5120
        self.encoder = nn.Sequential(
            nn.Linear(4, 1024),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(1024, 2048),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(2048, 5120)
        ).to(device)
        
    def one_hot_encode(self, sequence: str, max_len=512) -> torch.Tensor:
        """Convert RNA sequence to one-hot encoding"""
        base_to_idx = {'A': 0, 'C': 1, 'G': 2, 'U': 3, 'T': 3}
        
        encoding = torch.zeros(max_len, 4)
        for i, base in enumerate(sequence[:max_len]):
            if base.upper() in base_to_idx:
                encoding[i, base_to_idx[base.upper()]] = 1
                
        return encoding
    
    def forward(self, sequences, prompts):
        # One-hot encode sequences
        batch_encodings = torch.stack([
            self.one_hot_encode(seq) for seq in sequences
        ]).to(self.device)
        
        # Pass through MLP
        embeddings = self.encoder(batch_encodings)
        
        # Generate with Vicuna
        outputs = self.generator.generate_with_embeddings(embeddings, prompts)
        return outputs


# ============================================================================
# 5. RETRIEVAL-BASED BASELINES
# ============================================================================

class KNNRetrieval:
    """k-Nearest Neighbors Retrieval baseline"""
    def __init__(self, k=5, similarity_metric='edit_distance'):
        self.k = k
        self.similarity_metric = similarity_metric
        self.train_sequences = []
        self.train_functions = []
        
    def fit(self, sequences: List[str], functions: List[str]):
        self.train_sequences = sequences
        self.train_functions = functions
        
    def _compute_similarity(self, seq1: str, seq2: str) -> float:
        """Compute similarity between two sequences"""
        if self.similarity_metric == 'edit_distance':
            dist = editdistance.eval(seq1, seq2)
            max_len = max(len(seq1), len(seq2))
            return 1 - (dist / max_len) if max_len > 0 else 0
        else:
            raise ValueError(f"Unknown metric: {self.similarity_metric}")
    
    def predict(self, sequences: List[str]) -> List[str]:
        predictions = []
        
        for query_seq in sequences:
            # Compute similarities to all training sequences
            similarities = [
                self._compute_similarity(query_seq, train_seq)
                for train_seq in self.train_sequences
            ]
            
            # Get k nearest neighbors
            top_k_indices = np.argsort(similarities)[-self.k:][::-1]
            top_k_functions = [self.train_functions[i] for i in top_k_indices]
            
            # Combine/average the functions (simple concatenation for now)
            # In practice, you might want to weight by similarity
            combined_function = " ".join(top_k_functions[:3])  # Top 3
            predictions.append(combined_function)
            
        return predictions


class RAGWithGPT4o:
    """Retrieval-Augmented Generation with GPT-4o"""
    def __init__(self, k=3, api_key=None):
        self.k = k
        self.retriever = KNNRetrieval(k=k)
        self.api_key = api_key
        
    def fit(self, sequences: List[str], functions: List[str]):
        self.retriever.fit(sequences, functions)
        
    def predict(self, sequences: List[str]) -> List[str]:
        import openai
        openai.api_key = self.api_key
        
        predictions = []
        for query_seq in sequences:
            # Retrieve similar examples
            similar_examples = self.retriever.predict([query_seq])[0]
            
            # Construct prompt
            prompt = f"""Given these similar RNA sequences and their functions:
{similar_examples}

Now, describe the function of this RNA sequence:
{query_seq}

Provide a detailed functional description:"""
            
            # Query GPT-4o
            response = openai.ChatCompletion.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=200
            )
            
            predictions.append(response.choices[0].message.content)
            
        return predictions


class RAGWithLLaMA2:
    """Retrieval-Augmented Generation with LLaMA-2"""
    def __init__(self, llama_model, k=3, device='cuda'):
        self.k = k
        self.retriever = KNNRetrieval(k=k)
        self.llama = llama_model.to(device)
        self.device = device
        
    def fit(self, sequences: List[str], functions: List[str]):
        self.retriever.fit(sequences, functions)
        
    def predict(self, sequences: List[str]) -> List[str]:
        predictions = []
        
        for query_seq in sequences:
            # Retrieve similar examples
            similar_examples = self.retriever.predict([query_seq])[0]
            
            # Construct prompt
            prompt = f"""Given these similar RNA sequences and their functions:
{similar_examples}

Now, describe the function of this RNA sequence:
{query_seq}

Provide a detailed functional description:"""
            
            # Generate with LLaMA-2
            inputs = self.llama.tokenizer(prompt, return_tensors='pt').to(self.device)
            outputs = self.llama.generate(**inputs, max_length=300)
            prediction = self.llama.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            predictions.append(prediction)
            
        return predictions


# ============================================================================
# 6. HYBRID BASELINE
# ============================================================================

class RiNALMoTemplate:
    """RiNALMo + MLP Classifier + Template-based Generation"""
    def __init__(self, rinalmo_model, num_classes=100, device='cuda'):
        self.encoder = rinalmo_model.to(device)
        self.classifier = nn.Sequential(
            nn.Linear(1280, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        ).to(device)
        self.device = device
        
        # Define templates for each class
        self.templates = {
            0: "This RNA molecule functions as {} and is involved in {}.",
            1: "The primary role of this RNA is {} with activity in {}.",
            # ... more templates
        }
        self.class_to_function = {}
        
    def fit(self, sequences: List[str], functions: List[str]):
        # Create function categories
        unique_functions = list(set(functions))
        self.class_to_function = {i: func for i, func in enumerate(unique_functions)}
        
        # Train classifier
        optimizer = torch.optim.Adam(self.classifier.parameters(), lr=1e-4)
        
        for epoch in range(10):
            for seq, func in zip(sequences, functions):
                # Get RiNALMo embedding
                with torch.no_grad():
                    embedding = self.encoder(seq).mean(dim=0)  # Average pooling
                
                # Classify
                logits = self.classifier(embedding.unsqueeze(0))
                label = list(self.class_to_function.values()).index(func)
                
                loss = F.cross_entropy(logits, torch.tensor([label]).to(self.device))
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
    
    def predict(self, sequences: List[str]) -> List[str]:
        predictions = []
        
        for seq in sequences:
            # Get embedding
            with torch.no_grad():
                embedding = self.encoder(seq).mean(dim=0)
                logits = self.classifier(embedding.unsqueeze(0))
                pred_class = logits.argmax(dim=1).item()
            
            # Generate text from template
            function = self.class_to_function[pred_class]
            predictions.append(function)
            
        return predictions


# ============================================================================
# TRAINING UTILITIES
# ============================================================================

class BaselineTrainer:
    """Unified trainer for all baselines"""
    def __init__(self, model, device='cuda'):
        self.model = model
        self.device = device
        
    def train(self, train_loader, val_loader, num_epochs=10, lr=1e-4):
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, num_epochs)
        
        best_val_loss = float('inf')
        
        for epoch in range(num_epochs):
            # Training
            self.model.train()
            train_loss = 0
            for batch in train_loader:
                sequences = batch['sequence']
                functions = batch['function']
                
                # Forward pass (model-specific)
                loss = self.model.compute_loss(sequences, functions)
                
                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
            
            # Validation
            val_loss = self.evaluate(val_loader)
            
            print(f"Epoch {epoch+1}/{num_epochs} - "
                  f"Train Loss: {train_loss/len(train_loader):.4f} - "
                  f"Val Loss: {val_loss:.4f}")
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(self.model.state_dict(), 'best_model.pt')
            
            scheduler.step()
    
    def evaluate(self, data_loader):
        self.model.eval()
        total_loss = 0
        
        with torch.no_grad():
            for batch in data_loader:
                sequences = batch['sequence']
                functions = batch['function']
                loss = self.model.compute_loss(sequences, functions)
                total_loss += loss.item()
        
        return total_loss / len(data_loader)


# ============================================================================
# EVALUATION UTILITIES
# ============================================================================

def evaluate_all_baselines(baselines: Dict, test_data: List[Dict]) -> Dict:
    """Evaluate all baselines on test data"""
    from evaluation_metrics import compute_bleu, compute_simcse
    
    results = {}
    
    for name, model in baselines.items():
        print(f"\nEvaluating {name}...")
        
        sequences = [item['sequence'] for item in test_data]
        ground_truth = [item['function'] for item in test_data]
        
        # Generate predictions
        predictions = model.predict(sequences)
        
        # Compute metrics
        bleu_scores = compute_bleu(predictions, ground_truth)
        simcse_score = compute_simcse(predictions, ground_truth)
        
        results[name] = {
            'BLEU-1': bleu_scores[0],
            'BLEU-2': bleu_scores[1],
            'BLEU-3': bleu_scores[2],
            'BLEU-4': bleu_scores[3],
            'SimCSE': simcse_score
        }
        
        print(f"{name} Results:")
        for metric, score in results[name].items():
            print(f"  {metric}: {score:.4f}")
    
    return results


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

def main():
    """Example usage of all baselines"""
    
    # Load data
    train_dataset = RNADataset('data/train.json')
    test_dataset = RNADataset('data/test.json')
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32)
    
    # Initialize all baselines
    baselines = {
        # Traditional ML
        'TF-IDF-RF': TFIDFRandomForest(),
        'TF-IDF-SVM': TFIDFSVM(),
        
        # Seq2Seq
        'LSTM-ED': LSTMEncoderDecoder(),
        'GRU-ED': GRUEncoderDecoder(),
        'Trans-ED': TransformerEncoderDecoder(),
        'CNN-LSTM': CNNLSTMHybrid(),
        
        # Pre-trained LM
        'FT-T5-Base': FineTunedT5('t5-base'),
        'FT-T5-Large': FineTunedT5('t5-large'),
        'FT-FLAN-T5': FineTunedFLANT5(),
        'FT-BART': FineTunedBART(),
        
        # Retrieval
        'kNN-Retrieval': KNNRetrieval(k=5),
    }
    
    # Train and evaluate all baselines
    results = evaluate_all_baselines(baselines, test_dataset)
    
    # Save results
    import json
    with open('baseline_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print("\nAll baselines evaluated successfully!")


if __name__ == '__main__':
    main()
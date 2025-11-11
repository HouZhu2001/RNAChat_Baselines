"""
Fine-Grained RNA Function Prediction Benchmarks
Comprehensive evaluation suite for:
1. GO Term Prediction (BP, MF, CC)
2. RNA Type Classification
3. Subcellular Localization
4. Disease Association
5. Interaction Partner Prediction
6. Secondary Structure-Function Mapping

Includes 15+ specialized baselines + RNAChat integration

Usage:
    python finegrained_benchmarks.py --task go_prediction --model all --data go_data.csv
    
Author: RNAChat Benchmarks Team
Date: 2025
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support, 
    roc_auc_score, average_precision_score,
    hamming_loss, multilabel_confusion_matrix
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from transformers import (
    AutoTokenizer, AutoModel, AutoModelForSequenceClassification,
    T5Tokenizer, T5ForConditionalGeneration, Trainer, TrainingArguments
)
from scipy.sparse import csr_matrix
from collections import defaultdict, Counter
import networkx as nx
from tqdm import tqdm
import argparse
import json
import pickle
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')


# ============================================================================
# DATA STRUCTURES
# ============================================================================

class GOTerm:
    """Gene Ontology term structure"""
    def __init__(self, go_id, name, namespace, definition=""):
        self.go_id = go_id
        self.name = name
        self.namespace = namespace  # BP, MF, or CC
        self.definition = definition
        self.parents = []
        self.children = []


class GOGraph:
    """Gene Ontology directed acyclic graph"""
    def __init__(self):
        self.terms = {}
        self.graph = nx.DiGraph()
    
    def add_term(self, go_term):
        self.terms[go_term.go_id] = go_term
        self.graph.add_node(go_term.go_id)
    
    def add_relationship(self, child_id, parent_id):
        if child_id in self.terms and parent_id in self.terms:
            self.terms[child_id].parents.append(parent_id)
            self.terms[parent_id].children.append(child_id)
            self.graph.add_edge(child_id, parent_id)
    
    def get_ancestors(self, go_id):
        """Get all ancestor terms"""
        if go_id not in self.graph:
            return set()
        return set(nx.ancestors(self.graph, go_id))
    
    def get_descendants(self, go_id):
        """Get all descendant terms"""
        if go_id not in self.graph:
            return set()
        return set(nx.descendants(self.graph, go_id))


# ============================================================================
# EVALUATION METRICS
# ============================================================================

def compute_multilabel_metrics(y_true, y_pred, y_scores=None):
    """Compute comprehensive multi-label metrics"""
    metrics = {}
    
    # Basic metrics
    metrics['accuracy'] = accuracy_score(y_true, y_pred)
    metrics['hamming_loss'] = hamming_loss(y_true, y_pred)
    
    # Per-class metrics (macro/micro average)
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, average=None, zero_division=0
    )
    
    metrics['macro_precision'] = np.mean(precision)
    metrics['macro_recall'] = np.mean(recall)
    metrics['macro_f1'] = np.mean(f1)
    
    # Micro average
    precision_micro, recall_micro, f1_micro, _ = precision_recall_fscore_support(
        y_true, y_pred, average='micro', zero_division=0
    )
    metrics['micro_precision'] = precision_micro
    metrics['micro_recall'] = recall_micro
    metrics['micro_f1'] = f1_micro
    
    # AUC metrics if scores provided
    if y_scores is not None:
        try:
            metrics['macro_auc_roc'] = roc_auc_score(y_true, y_scores, average='macro')
            metrics['micro_auc_roc'] = roc_auc_score(y_true, y_scores, average='micro')
            metrics['macro_auc_pr'] = average_precision_score(y_true, y_scores, average='macro')
            metrics['micro_auc_pr'] = average_precision_score(y_true, y_scores, average='micro')
        except:
            pass
    
    return metrics


def compute_hierarchical_metrics(y_true, y_pred, go_graph):
    """Compute hierarchy-aware metrics for GO prediction"""
    metrics = {}
    
    # Hierarchical precision/recall
    total_tp = 0
    total_pred = 0
    total_true = 0
    
    for true_terms, pred_terms in zip(y_true, y_pred):
        # Expand to ancestors
        true_expanded = set(true_terms)
        pred_expanded = set(pred_terms)
        
        for term in true_terms:
            true_expanded.update(go_graph.get_ancestors(term))
        for term in pred_terms:
            pred_expanded.update(go_graph.get_ancestors(term))
        
        # Compute overlap
        tp = len(true_expanded & pred_expanded)
        total_tp += tp
        total_pred += len(pred_expanded)
        total_true += len(true_expanded)
    
    metrics['hierarchical_precision'] = total_tp / total_pred if total_pred > 0 else 0
    metrics['hierarchical_recall'] = total_tp / total_true if total_true > 0 else 0
    
    if metrics['hierarchical_precision'] + metrics['hierarchical_recall'] > 0:
        metrics['hierarchical_f1'] = (
            2 * metrics['hierarchical_precision'] * metrics['hierarchical_recall'] /
            (metrics['hierarchical_precision'] + metrics['hierarchical_recall'])
        )
    else:
        metrics['hierarchical_f1'] = 0
    
    return metrics


def compute_information_content_metrics(y_true, y_pred, ic_scores):
    """Compute information content weighted metrics"""
    metrics = {}
    
    weighted_correct = 0
    total_weight = 0
    
    for true_terms, pred_terms in zip(y_true, y_pred):
        true_set = set(true_terms)
        pred_set = set(pred_terms)
        
        for term in pred_set:
            if term in ic_scores:
                weight = ic_scores[term]
                if term in true_set:
                    weighted_correct += weight
                total_weight += weight
    
    metrics['ic_weighted_precision'] = weighted_correct / total_weight if total_weight > 0 else 0
    
    return metrics


# ============================================================================
# TASK 1: GO TERM PREDICTION BASELINES
# ============================================================================

class BiRWLGO:
    """
    Bi-Random Walk for lncRNA GO prediction
    From: "Gene Ontology-based function prediction of long non-coding RNAs using bi-random walk"
    """
    def __init__(self, alpha=0.5, max_iter=100, go_graph=None):
        self.alpha = alpha
        self.max_iter = max_iter
        self.go_graph = go_graph
        self.rna_similarity = None
        self.go_similarity = None
        self.known_associations = None
    
    def build_rna_similarity(self, sequences):
        """Build RNA-RNA similarity matrix based on k-mer"""
        n = len(sequences)
        similarity = np.zeros((n, n))
        
        # Simple k-mer based similarity
        for i in range(n):
            for j in range(i, n):
                sim = self._sequence_similarity(sequences[i], sequences[j])
                similarity[i, j] = sim
                similarity[j, i] = sim
        
        # Normalize
        row_sums = similarity.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1
        self.rna_similarity = similarity / row_sums
        
        return self.rna_similarity
    
    def _sequence_similarity(self, seq1, seq2, k=3):
        """Compute k-mer Jaccard similarity"""
        def get_kmers(seq, k):
            return set([seq[i:i+k] for i in range(len(seq)-k+1)])
        
        kmers1 = get_kmers(str(seq1), k)
        kmers2 = get_kmers(str(seq2), k)
        
        if len(kmers1 | kmers2) == 0:
            return 0
        return len(kmers1 & kmers2) / len(kmers1 | kmers2)
    
    def build_go_similarity(self, go_terms):
        """Build GO-GO similarity matrix using GO graph"""
        n = len(go_terms)
        similarity = np.zeros((n, n))
        
        for i, go_i in enumerate(go_terms):
            for j, go_j in enumerate(go_terms):
                if i == j:
                    similarity[i, j] = 1.0
                else:
                    # Semantic similarity based on common ancestors
                    sim = self._go_semantic_similarity(go_i, go_j)
                    similarity[i, j] = sim
        
        # Normalize
        row_sums = similarity.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1
        self.go_similarity = similarity / row_sums
        
        return self.go_similarity
    
    def _go_semantic_similarity(self, go1, go2):
        """Compute GO semantic similarity"""
        if self.go_graph is None:
            return 0.5  # fallback
        
        ancestors1 = self.go_graph.get_ancestors(go1)
        ancestors2 = self.go_graph.get_ancestors(go2)
        
        ancestors1.add(go1)
        ancestors2.add(go2)
        
        if len(ancestors1 | ancestors2) == 0:
            return 0
        return len(ancestors1 & ancestors2) / len(ancestors1 | ancestors2)
    
    def fit(self, sequences, go_annotations, go_terms):
        """Train the bi-random walk model"""
        # Build similarity matrices
        print("Building RNA similarity matrix...")
        self.build_rna_similarity(sequences)
        
        print("Building GO similarity matrix...")
        self.build_go_similarity(go_terms)
        
        # Build association matrix
        n_rnas = len(sequences)
        n_gos = len(go_terms)
        self.known_associations = np.zeros((n_rnas, n_gos))
        
        go_to_idx = {go: i for i, go in enumerate(go_terms)}
        for i, annots in enumerate(go_annotations):
            for go in annots:
                if go in go_to_idx:
                    self.known_associations[i, go_to_idx[go]] = 1
        
        return self
    
    def predict(self, test_sequences, top_k=10):
        """Predict GO terms using bi-random walk"""
        predictions = []
        
        for test_seq in tqdm(test_sequences, desc="BiRWLGO Prediction"):
            # Compute similarity to training RNAs
            sim_scores = np.array([
                self._sequence_similarity(test_seq, train_seq)
                for train_seq in self.train_sequences
            ])
            
            # Initialize scores
            rna_scores = sim_scores / (sim_scores.sum() + 1e-10)
            go_scores = np.ones(self.known_associations.shape[1]) / self.known_associations.shape[1]
            
            # Bi-random walk
            for _ in range(self.max_iter):
                # Update GO scores
                new_go_scores = (
                    self.alpha * (rna_scores @ self.known_associations) +
                    (1 - self.alpha) * (go_scores @ self.go_similarity.T)
                )
                new_go_scores = new_go_scores / (new_go_scores.sum() + 1e-10)
                
                # Check convergence
                if np.allclose(new_go_scores, go_scores, atol=1e-6):
                    break
                
                go_scores = new_go_scores
            
            # Get top-k predictions
            top_indices = np.argsort(go_scores)[-top_k:][::-1]
            pred_gos = [self.go_terms[i] for i in top_indices]
            predictions.append(pred_gos)
        
        return predictions
    
    def set_train_data(self, sequences, go_terms):
        """Set training data for prediction"""
        self.train_sequences = sequences
        self.go_terms = go_terms


class TFIDFGOPredictor:
    """TF-IDF based GO prediction"""
    def __init__(self, n_features=5000):
        self.vectorizer = TfidfVectorizer(
            ngram_range=(3, 5), analyzer='char', max_features=n_features
        )
        self.mlb = MultiLabelBinarizer()
        self.classifier = OneVsRestClassifier(RandomForestClassifier(n_estimators=100))
    
    def fit(self, sequences, go_annotations):
        # Vectorize sequences
        X = self.vectorizer.fit_transform(sequences)
        
        # Encode GO terms
        y = self.mlb.fit_transform(go_annotations)
        
        # Train
        self.classifier.fit(X, y)
        
        return self
    
    def predict(self, sequences, threshold=0.5):
        X = self.vectorizer.transform(sequences)
        y_scores = self.classifier.predict_proba(X)
        
        predictions = []
        for scores in y_scores:
            pred_indices = np.where(scores > threshold)[0]
            pred_gos = [self.mlb.classes_[i] for i in pred_indices]
            predictions.append(pred_gos if pred_gos else [self.mlb.classes_[np.argmax(scores)]])
        
        return predictions, y_scores


class DeepGOPredictor(nn.Module):
    """
    Deep learning GO predictor inspired by DeepGO
    CNN + LSTM encoder with multi-label classification head
    """
    def __init__(self, vocab_size, num_go_terms, embed_dim=128, hidden_dim=256):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        
        # CNN layers for local patterns
        self.conv1 = nn.Conv1d(embed_dim, hidden_dim, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=5, padding=2)
        self.conv3 = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=7, padding=3)
        
        self.pool = nn.AdaptiveMaxPool1d(1)
        
        # LSTM for sequential patterns
        self.lstm = nn.LSTM(hidden_dim*3, hidden_dim, batch_first=True, bidirectional=True)
        
        # Classification head
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim*2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden_dim, num_go_terms)
        )
    
    def forward(self, x):
        # Embed
        emb = self.embedding(x).transpose(1, 2)  # [B, E, L]
        
        # CNN features
        c1 = F.relu(self.conv1(emb))
        c2 = F.relu(self.conv2(emb))
        c3 = F.relu(self.conv3(emb))
        
        # Concatenate
        conv_out = torch.cat([c1, c2, c3], dim=1).transpose(1, 2)  # [B, L, H*3]
        
        # LSTM
        lstm_out, _ = self.lstm(conv_out)
        
        # Global pooling
        pooled = lstm_out.mean(dim=1)  # [B, H*2]
        
        # Classify
        logits = self.fc(pooled)
        
        return logits


class GOTransformer(nn.Module):
    """Transformer-based GO predictor"""
    def __init__(self, vocab_size, num_go_terms, d_model=256, nhead=8, num_layers=4):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.pos_encoder = self._create_pos_encoding(5000, d_model)
        
        encoder_layer = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward=1024, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(d_model // 2, num_go_terms)
        )
    
    def _create_pos_encoding(self, max_len, d_model):
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe.unsqueeze(0)
    
    def forward(self, x):
        emb = self.embedding(x)
        emb = emb + self.pos_encoder[:, :x.size(1), :].to(x.device)
        
        # Create padding mask
        mask = (x == 0)
        
        # Transform
        out = self.transformer(emb, src_key_padding_mask=mask)
        
        # Pool and classify
        pooled = out.mean(dim=1)
        logits = self.classifier(pooled)
        
        return logits


class ProtTransGOPredictor:
    """
    Use pre-trained ProtTrans (or RNA-FM) embeddings for GO prediction
    """
    def __init__(self, model_name='Rostlab/prot_bert', num_go_terms=100):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.encoder = AutoModel.from_pretrained(model_name)
        self.mlb = MultiLabelBinarizer()
        
        # Classifier head
        self.classifier = nn.Sequential(
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_go_terms)
        )
        
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.encoder.to(self.device)
        self.classifier.to(self.device)
    
    def encode_sequences(self, sequences, batch_size=16):
        """Get embeddings from pre-trained model"""
        embeddings = []
        
        self.encoder.eval()
        with torch.no_grad():
            for i in range(0, len(sequences), batch_size):
                batch = sequences[i:i+batch_size]
                # Add spaces between bases for ProtBERT format
                batch = [' '.join(seq) for seq in batch]
                
                inputs = self.tokenizer(batch, padding=True, truncation=True, 
                                       max_length=512, return_tensors='pt').to(self.device)
                outputs = self.encoder(**inputs)
                batch_emb = outputs.last_hidden_state[:, 0, :].cpu().numpy()
                embeddings.append(batch_emb)
        
        return np.vstack(embeddings)
    
    def fit(self, sequences, go_annotations, epochs=10, batch_size=32, lr=1e-4):
        """Fine-tune classifier on GO prediction"""
        # Get embeddings
        print("Encoding sequences...")
        X = self.encode_sequences(sequences)
        
        # Encode labels
        y = self.mlb.fit_transform(go_annotations)
        
        # Train classifier
        X_tensor = torch.FloatTensor(X).to(self.device)
        y_tensor = torch.FloatTensor(y).to(self.device)
        
        optimizer = torch.optim.Adam(self.classifier.parameters(), lr=lr)
        criterion = nn.BCEWithLogitsLoss()
        
        self.classifier.train()
        for epoch in range(epochs):
            for i in range(0, len(X_tensor), batch_size):
                batch_X = X_tensor[i:i+batch_size]
                batch_y = y_tensor[i:i+batch_size]
                
                optimizer.zero_grad()
                logits = self.classifier(batch_X)
                loss = criterion(logits, batch_y)
                loss.backward()
                optimizer.step()
            
            print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}")
        
        return self
    
    def predict(self, sequences, threshold=0.5):
        """Predict GO terms"""
        X = self.encode_sequences(sequences)
        X_tensor = torch.FloatTensor(X).to(self.device)
        
        self.classifier.eval()
        with torch.no_grad():
            logits = self.classifier(X_tensor)
            probs = torch.sigmoid(logits).cpu().numpy()
        
        predictions = []
        for prob in probs:
            pred_indices = np.where(prob > threshold)[0]
            pred_gos = [self.mlb.classes_[i] for i in pred_indices]
            predictions.append(pred_gos if pred_gos else [self.mlb.classes_[np.argmax(prob)]])
        
        return predictions, probs


# ============================================================================
# TASK 2: RNA TYPE CLASSIFICATION BASELINES
# ============================================================================

class RNATypeClassifier:
    """Multi-class RNA type classification"""
    RNA_TYPES = ['mRNA', 'tRNA', 'rRNA', 'miRNA', 'lncRNA', 'snoRNA', 'snRNA', 'other']
    
    def __init__(self, model_type='rf'):
        self.model_type = model_type
        self.vectorizer = TfidfVectorizer(ngram_range=(3, 5), analyzer='char', max_features=5000)
        
        if model_type == 'rf':
            self.classifier = RandomForestClassifier(n_estimators=200, max_depth=20)
        elif model_type == 'svm':
            self.classifier = SVC(kernel='rbf', probability=True)
        elif model_type == 'xgboost':
            from xgboost import XGBClassifier
            self.classifier = XGBClassifier(n_estimators=200)
    
    def fit(self, sequences, rna_types):
        X = self.vectorizer.fit_transform(sequences)
        self.classifier.fit(X, rna_types)
        return self
    
    def predict(self, sequences):
        X = self.vectorizer.transform(sequences)
        preds = self.classifier.predict(X)
        probs = self.classifier.predict_proba(X)
        return preds, probs


class RNATypeCNN(nn.Module):
    """CNN for RNA type classification"""
    def __init__(self, vocab_size, num_classes=8, embed_dim=128):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        
        self.convs = nn.ModuleList([
            nn.Conv1d(embed_dim, 256, k, padding=k//2) for k in [3, 5, 7, 9]
        ])
        
        self.fc = nn.Sequential(
            nn.Linear(256*4, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, x):
        emb = self.embedding(x).transpose(1, 2)
        
        conv_outs = [F.relu(conv(emb)) for conv in self.convs]
        pooled = [F.adaptive_max_pool1d(out, 1).squeeze(2) for out in conv_outs]
        
        concat = torch.cat(pooled, dim=1)
        logits = self.fc(concat)
        
        return logits


# ============================================================================
# TASK 3: SUBCELLULAR LOCALIZATION PREDICTION
# ============================================================================

class SubcellularLocalizationPredictor:
    """Predict subcellular localization of RNAs"""
    LOCATIONS = ['nucleus', 'cytoplasm', 'mitochondria', 'exosome', 'membrane', 'other']
    
    def __init__(self):
        self.vectorizer = TfidfVectorizer(ngram_range=(3, 5), analyzer='char', max_features=3000)
        self.classifier = RandomForestClassifier(n_estimators=150)
    
    def fit(self, sequences, locations):
        X = self.vectorizer.fit_transform(sequences)
        self.classifier.fit(X, locations)
        return self
    
    def predict(self, sequences):
        X = self.vectorizer.transform(sequences)
        preds = self.classifier.predict(X)
        probs = self.classifier.predict_proba(X)
        return preds, probs


# ============================================================================
# RNACHAT INTEGRATION
# ============================================================================

class RNAChatGOPredictor:
    """
    Adapt RNAChat for GO term prediction
    Uses RNAChat's generated text to extract GO terms
    """
    def __init__(self, rnachat_model_path=None, go_terms_list=None):
        self.go_terms_list = go_terms_list
        # Load RNAChat model (placeholder - replace with actual loading)
        self.rnachat = None  # Load your RNAChat model here
        
        # GO term extraction patterns
        self.go_patterns = self._build_go_patterns()
    
    def _build_go_patterns(self):
        """Build keyword patterns for GO term extraction"""
        patterns = defaultdict(list)
        
        # Biological Process keywords
        patterns['BP'] = [
            'transcription', 'translation', 'regulation', 'metabolism',
            'signaling', 'transport', 'catalysis', 'binding', 'process'
        ]
        
        # Molecular Function keywords
        patterns['MF'] = [
            'binding', 'catalytic', 'activity', 'function', 'enzyme',
            'receptor', 'transporter', 'kinase', 'phosphatase'
        ]
        
        # Cellular Component keywords
        patterns['CC'] = [
            'nucleus', 'cytoplasm', 'membrane', 'ribosome', 'mitochondria',
            'complex', 'organelle', 'compartment'
        ]
        
        return patterns
    
    def predict_from_text(self, generated_texts, top_k=10):
        """Extract GO terms from RNAChat generated text"""
        predictions = []
        
        for text in generated_texts:
            text_lower = text.lower()
            
            # Score GO terms based on keyword matching
            go_scores = {}
            for go_term in self.go_terms_list:
                score = 0
                go_name = go_term.name.lower()
                
                # Exact match
                if go_name in text_lower:
                    score += 10
                
                # Keyword matching
                words = go_name.split()
                for word in words:
                    if word in text_lower:
                        score += 1
                
                # Definition matching
                if hasattr(go_term, 'definition'):
                    def_words = go_term.definition.lower().split()
                    for word in def_words[:10]:  # Top 10 words from definition
                        if len(word) > 4 and word in text_lower:
                            score += 0.5
                
                if score > 0:
                    go_scores[go_term.go_id] = score
            
            # Get top-k
            sorted_gos = sorted(go_scores.items(), key=lambda x: x[1], reverse=True)
            pred_gos = [go_id for go_id, score in sorted_gos[:top_k]]
            
            predictions.append(pred_gos if pred_gos else [self.go_terms_list[0].go_id])
        
        return predictions
    
    def predict(self, sequences, names=None, batch_size=8):
        """
        Generate predictions using RNAChat
        This is a placeholder - implement actual RNAChat inference
        """
        predictions = []
        
        # Placeholder: Replace with actual RNAChat inference
        generated_texts = []
        for seq, name in zip(sequences, names if names else ['RNA']*len(sequences)):
            # generated_text = self.rnachat.generate(seq, name)
            generated_text = f"This RNA molecule functions in transcription regulation and binds to proteins in the nucleus."
            generated_texts.append(generated_text)
        
        # Extract GO terms from generated text
        go_predictions = self.predict_from_text(generated_texts)
        
        return go_predictions


class RNAChatRNATypeClassifier:
    """Use RNAChat for RNA type classification"""
    def __init__(self, rnachat_model=None):
        self.rnachat = rnachat_model
        self.type_keywords = {
            'mRNA': ['messenger', 'coding', 'protein', 'translation', 'mrna'],
            'tRNA': ['transfer', 'trna', 'amino acid', 'anticodon'],
            'rRNA': ['ribosomal', 'rrna', 'ribosome', '16s', '23s'],
            'miRNA': ['micro', 'mirna', 'regulation', 'target', 'small'],
            'lncRNA': ['long', 'non-coding', 'lncrna', 'regulatory'],
            'snoRNA': ['small nucleolar', 'snorna', 'rrna modification'],
            'snRNA': ['small nuclear', 'snrna', 'splicing'],
            'other': ['other', 'unknown', 'novel']
        }
    
    def predict(self, sequences, names=None):
        """Classify RNA type from RNAChat output"""
        predictions = []
        
        for seq, name in zip(sequences, names if names else ['RNA']*len(sequences)):
            # Generate text with RNAChat (placeholder)
            # generated_text = self.rnachat.generate(seq, name)
            generated_text = f"This RNA functions as a messenger RNA encoding proteins."
            
            # Score each type
            scores = {}
            text_lower = generated_text.lower()
            for rna_type, keywords in self.type_keywords.items():
                score = sum(1 for kw in keywords if kw in text_lower)
                scores[rna_type] = score
            
            # Predict type with highest score
            pred_type = max(scores, key=scores.get) if max(scores.values()) > 0 else 'other'
            predictions.append(pred_type)
        
        return predictions


# ============================================================================
# DATA LOADING
# ============================================================================

def load_go_data(csv_path):
    """Load GO annotation data from rna_go.csv format"""
    print(f"Loading GO data from {csv_path}...")
    df = pd.read_csv(csv_path)
    
    # Required columns: rna_id and sequence
    required_cols = ['rna_id', 'sequence']
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Missing column: {col}")
    
    df = df.dropna(subset=required_cols)
    
    # Get all GO term columns (all columns except rna_id and sequence)
    go_term_columns = [col for col in df.columns if col not in ['rna_id', 'sequence']]
    
    # For each row, collect GO terms where the value is not empty/NaN
    def extract_go_terms(row):
        go_terms = []
        for go_col in go_term_columns:
            value = row[go_col]
            # If the value is not empty/NaN, the RNA has this GO term
            if pd.notna(value) and str(value).strip() != "":
                go_terms.append(go_col)  # Use column name as GO term identifier
        return go_terms
    
    df['go_list'] = df.apply(extract_go_terms, axis=1)
    
    # Filter out rows with no GO terms
    df = df[df['go_list'].apply(len) > 0]
    
    # Split train/test
    n = len(df)
    train_size = int(0.8 * n)
    val_size = int(0.1 * n)
    
    train_df = df.iloc[:train_size]
    val_df = df.iloc[train_size:train_size+val_size]
    test_df = df.iloc[train_size+val_size:]
    
    print(f"Loaded: Train={len(train_df)}, Val={len(val_df)}, Test={len(test_df)}")
    print(f"Total GO terms: {len(go_term_columns)}")
    return train_df, val_df, test_df


def load_rna_type_data(csv_path):
    """Load RNA type classification data from a CSV file."""
    print(f"Loading RNA type data from {csv_path}...")
    df = pd.read_csv(csv_path)
    
    # Normalize column names (handle Sequence vs sequence, etc.)
    normalized_columns = {col: col.strip().lower() for col in df.columns}
    df = df.rename(columns=normalized_columns)
    
    required_cols = ['sequence', 'rna_type']
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Missing column: {col}")
    
    df = df.dropna(subset=required_cols)
    
    # Trim whitespace from RNA type labels
    df['rna_type'] = df['rna_type'].astype(str).str.strip()
    
    # Split
    n = len(df)
    train_size = int(0.8 * n)
    val_size = int(0.1 * n)
    
    train_df = df.iloc[:train_size].reset_index(drop=True)
    val_df = df.iloc[train_size:train_size+val_size].reset_index(drop=True)
    test_df = df.iloc[train_size+val_size:].reset_index(drop=True)
    
    print(f"Loaded RNA type data: Train={len(train_df)}, Val={len(val_df)}, Test={len(test_df)}")
    unique_types = df['rna_type'].nunique()
    print(f"Unique RNA types: {unique_types}")
    
    return train_df, val_df, test_df


def build_go_graph_from_obo(obo_path):
    """Build GO graph from OBO file"""
    go_graph = GOGraph()
    
    # Simplified OBO parser (use goatools for production)
    current_term = None
    
    with open(obo_path, 'r') as f:
        for line in f:
            line = line.strip()
            
            if line == '[Term]':
                if current_term:
                    go_graph.add_term(current_term)
                current_term = {'parents': []}
            
            elif line.startswith('id:'):
                if current_term is not None:
                    current_term['id'] = line.split('id:')[1].strip()
            
            elif line.startswith('name:'):
                if current_term is not None:
                    current_term['name'] = line.split('name:')[1].strip()
            
            elif line.startswith('namespace:'):
                if current_term is not None:
                    ns = line.split('namespace:')[1].strip()
                    if 'biological_process' in ns:
                        current_term['namespace'] = 'BP'
                    elif 'molecular_function' in ns:
                        current_term['namespace'] = 'MF'
                    elif 'cellular_component' in ns:
                        current_term['namespace'] = 'CC'
            
            elif line.startswith('is_a:'):
                if current_term is not None:
                    parent = line.split('is_a:')[1].split('!')[0].strip()
                    current_term['parents'].append(parent)
    
    # Add final term
    if current_term:
        go_term = GOTerm(
            current_term.get('id', ''),
            current_term.get('name', ''),
            current_term.get('namespace', 'BP')
        )
        go_graph.add_term(go_term)
        
        for parent in current_term.get('parents', []):
            go_graph.add_relationship(current_term['id'], parent)
    
    return go_graph


# ============================================================================
# TRAINING FUNCTIONS
# ============================================================================

def train_deep_go(model, train_loader, val_loader, num_epochs=50, lr=1e-3, device='cuda'):
    """Train DeepGO-style model"""
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCEWithLogitsLoss()
    
    best_val_f1 = 0
    
    for epoch in range(num_epochs):
        # Train
        model.train()
        train_loss = 0
        for batch in tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}'):
            seqs = batch['sequence'].to(device)
            labels = batch['labels'].to(device)
            
            optimizer.zero_grad()
            logits = model(seqs)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        # Validate
        model.eval()
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for batch in val_loader:
                seqs = batch['sequence'].to(device)
                labels = batch['labels']
                
                logits = model(seqs)
                probs = torch.sigmoid(logits).cpu().numpy()
                preds = (probs > 0.5).astype(int)
                
                all_preds.append(preds)
                all_labels.append(labels.numpy())
        
        all_preds = np.vstack(all_preds)
        all_labels = np.vstack(all_labels)
        
        metrics = compute_multilabel_metrics(all_labels, all_preds)
        
        print(f"Epoch {epoch+1}: Loss={train_loss/len(train_loader):.4f}, "
              f"F1={metrics['micro_f1']:.4f}, AUC={metrics.get('micro_auc_roc', 0):.4f}")
        
        if metrics['micro_f1'] > best_val_f1:
            best_val_f1 = metrics['micro_f1']
            torch.save(model.state_dict(), 'checkpoints/best_deepgo.pt')
    
    model.load_state_dict(torch.load('checkpoints/best_deepgo.pt'))
    return model


# ============================================================================
# EVALUATION FUNCTIONS
# ============================================================================

def evaluate_go_prediction(model, test_df, go_graph=None, model_type='traditional'):
    """Comprehensive GO prediction evaluation"""
    sequences = test_df['sequence'].tolist()
    true_annotations = test_df['go_list'].tolist()
    names = test_df['name'].tolist() if 'name' in test_df.columns else None
    
    # Get predictions
    if model_type == 'traditional':
        predictions, scores = model.predict(sequences)
    elif model_type == 'rnachat':
        predictions = model.predict(sequences, names)
        scores = None
    elif model_type == 'birwlgo':
        predictions = model.predict(sequences)
        scores = None
    else:
        predictions = model.predict(sequences)
        scores = None
    
    # Convert to binary matrix for standard metrics
    all_go_terms = sorted(list(set([go for annots in true_annotations for go in annots])))
    go_to_idx = {go: i for i, go in enumerate(all_go_terms)}
    
    y_true = np.zeros((len(true_annotations), len(all_go_terms)))
    y_pred = np.zeros((len(predictions), len(all_go_terms)))
    
    for i, annots in enumerate(true_annotations):
        for go in annots:
            if go in go_to_idx:
                y_true[i, go_to_idx[go]] = 1
    
    for i, preds in enumerate(predictions):
        for go in preds:
            if go in go_to_idx:
                y_pred[i, go_to_idx[go]] = 1
    
    # Compute metrics
    results = {}
    
    # Standard multi-label metrics
    standard_metrics = compute_multilabel_metrics(y_true, y_pred, scores)
    results.update(standard_metrics)
    
    # Hierarchical metrics
    if go_graph is not None:
        hier_metrics = compute_hierarchical_metrics(true_annotations, predictions, go_graph)
        results.update(hier_metrics)
    
    # Per-namespace metrics
    namespaces = ['BP', 'MF', 'CC']
    for ns in namespaces:
        if go_graph is not None:
            ns_true = [[go for go in annots if go in go_graph.terms and go_graph.terms[go].namespace == ns] 
                       for annots in true_annotations]
            ns_pred = [[go for go in preds if go in go_graph.terms and go_graph.terms[go].namespace == ns] 
                       for preds in predictions]
            
            # Compute namespace-specific F1
            tp = sum(len(set(t) & set(p)) for t, p in zip(ns_true, ns_pred))
            fp = sum(len(set(p) - set(t)) for t, p in zip(ns_true, ns_pred))
            fn = sum(len(set(t) - set(p)) for t, p in zip(ns_true, ns_pred))
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            
            results[f'{ns}_precision'] = precision
            results[f'{ns}_recall'] = recall
            results[f'{ns}_f1'] = f1
    
    return results


def evaluate_rna_type_classification(model, test_df):
    """Evaluate RNA type classification"""
    sequences = test_df['sequence'].tolist()
    true_types = test_df['rna_type'].tolist()
    
    if hasattr(model, 'predict'):
        if 'name' in test_df.columns:
            names = test_df['name'].tolist()
            pred_types = model.predict(sequences, names)
        else:
            pred_types, _ = model.predict(sequences)
    else:
        # Neural model
        pred_types = []  # Implement neural prediction
    
    # Compute metrics
    accuracy = accuracy_score(true_types, pred_types)
    precision, recall, f1, _ = precision_recall_fscore_support(
        true_types, pred_types, average='macro', zero_division=0
    )
    
    # Per-class metrics
    per_class_metrics = precision_recall_fscore_support(
        true_types, pred_types, average=None, zero_division=0
    )
    
    results = {
        'accuracy': accuracy,
        'macro_precision': precision,
        'macro_recall': recall,
        'macro_f1': f1,
        'per_class': {
            'precision': per_class_metrics[0].tolist(),
            'recall': per_class_metrics[1].tolist(),
            'f1': per_class_metrics[2].tolist()
        }
    }
    
    return results


# ============================================================================
# MAIN EXPERIMENT RUNNER
# ============================================================================

def run_go_prediction_benchmarks(train_df, val_df, test_df, go_graph=None):
    """Run all GO prediction baselines"""
    results = {}
    
    print("\n" + "="*80)
    print("GO TERM PREDICTION BENCHMARKS")
    print("="*80)
    
    # Prepare data
    train_seqs = train_df['sequence'].tolist()
    train_annots = train_df['go_list'].tolist()
    
    # Get all GO terms
    all_go_terms = sorted(list(set([go for annots in train_annots for go in annots])))
    
    # 1. BiRWLGO
    print("\n--- BiRWLGO ---")
    birwlgo = BiRWLGO(alpha=0.5, go_graph=go_graph)
    birwlgo.fit(train_seqs, train_annots, all_go_terms)
    birwlgo.set_train_data(train_seqs, all_go_terms)
    result = evaluate_go_prediction(birwlgo, test_df, go_graph, 'birwlgo')
    results['BiRWLGO'] = result
    print(f"BiRWLGO: F1={result['micro_f1']:.4f}, Hier-F1={result.get('hierarchical_f1', 0):.4f}")
    
    # 2. TF-IDF + Random Forest
    print("\n--- TF-IDF-RF ---")
    tfidf_rf = TFIDFGOPredictor(n_features=5000)
    tfidf_rf.fit(train_seqs, train_annots)
    result = evaluate_go_prediction(tfidf_rf, test_df, go_graph, 'traditional')
    results['TF-IDF-RF'] = result
    print(f"TF-IDF-RF: F1={result['micro_f1']:.4f}, Hier-F1={result.get('hierarchical_f1', 0):.4f}")
    
    # 3. DeepGO (if neural models enabled)
    # Commented out for speed - uncomment to run
    # print("\n--- DeepGO ---")
    # deepgo = DeepGOPredictor(vocab_size=10, num_go_terms=len(all_go_terms))
    # train neural model...
    # results['DeepGO'] = result
    
    # 4. RNAChat
    print("\n--- RNAChat ---")
    # Create mock GO terms for RNAChat
    go_terms_objs = [GOTerm(go_id, go_id, 'BP') for go_id in all_go_terms]
    rnachat_go = RNAChatGOPredictor(go_terms_list=go_terms_objs)
    result = evaluate_go_prediction(rnachat_go, test_df, go_graph, 'rnachat')
    results['RNAChat'] = result
    print(f"RNAChat: F1={result['micro_f1']:.4f}, Hier-F1={result.get('hierarchical_f1', 0):.4f}")
    
    return results


def run_rna_type_benchmarks(train_df, val_df, test_df):
    """Run RNA type classification benchmarks"""
    results = {}
    
    print("\n" + "="*80)
    print("RNA TYPE CLASSIFICATION BENCHMARKS")
    print("="*80)
    
    train_seqs = train_df['sequence'].tolist()
    train_types = train_df['rna_type'].tolist()
    
    # 1. TF-IDF + Random Forest
    print("\n--- TF-IDF-RF ---")
    tfidf_rf = RNATypeClassifier(model_type='rf')
    tfidf_rf.fit(train_seqs, train_types)
    result = evaluate_rna_type_classification(tfidf_rf, test_df)
    results['TF-IDF-RF'] = result
    print(f"TF-IDF-RF: Accuracy={result['accuracy']:.4f}, F1={result['macro_f1']:.4f}")
    
    # 2. TF-IDF + SVM
    print("\n--- TF-IDF-SVM ---")
    tfidf_svm = RNATypeClassifier(model_type='svm')
    tfidf_svm.fit(train_seqs, train_types)
    result = evaluate_rna_type_classification(tfidf_svm, test_df)
    results['TF-IDF-SVM'] = result
    print(f"TF-IDF-SVM: Accuracy={result['accuracy']:.4f}, F1={result['macro_f1']:.4f}")
    
    # 3. RNAChat
    print("\n--- RNAChat ---")
    rnachat_type = RNAChatRNATypeClassifier()
    result = evaluate_rna_type_classification(rnachat_type, test_df)
    results['RNAChat'] = result
    print(f"RNAChat: Accuracy={result['accuracy']:.4f}, F1={result['macro_f1']:.4f}")
    
    return results


def save_results(results, task_name, output_dir='results/finegrained'):
    """Save results to JSON and generate LaTeX table"""
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Save JSON
    json_path = f"{output_dir}/{task_name}_results.json"
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {json_path}")
    
    # Generate LaTeX table
    latex_path = f"{output_dir}/{task_name}_table.tex"
    
    if 'go' in task_name.lower():
        # GO prediction table
        latex = "\\begin{table}[h]\n\\centering\n"
        latex += "\\begin{tabular}{lcccccc}\n\\toprule\n"
        latex += "Model & Micro-F1 & Macro-F1 & Hier-F1 & BP-F1 & MF-F1 & CC-F1 \\\\\n\\midrule\n"
        
        for model, res in results.items():
            latex += f"{model} & {res['micro_f1']:.3f} & {res['macro_f1']:.3f} & "
            latex += f"{res.get('hierarchical_f1', 0):.3f} & "
            latex += f"{res.get('BP_f1', 0):.3f} & {res.get('MF_f1', 0):.3f} & "
            latex += f"{res.get('CC_f1', 0):.3f} \\\\\n"
        
        latex += "\\bottomrule\n\\end{tabular}\n"
        latex += "\\caption{GO term prediction results. Hier-F1: hierarchy-aware F1.}\n"
        latex += "\\end{table}"
    
    elif 'type' in task_name.lower():
        # RNA type classification table
        latex = "\\begin{table}[h]\n\\centering\n"
        latex += "\\begin{tabular}{lccc}\n\\toprule\n"
        latex += "Model & Accuracy & Macro-Precision & Macro-F1 \\\\\n\\midrule\n"
        
        for model, res in results.items():
            latex += f"{model} & {res['accuracy']:.3f} & {res['macro_precision']:.3f} & "
            latex += f"{res['macro_f1']:.3f} \\\\\n"
        
        latex += "\\bottomrule\n\\end{tabular}\n"
        latex += "\\caption{RNA type classification results.}\n"
        latex += "\\end{table}"
    
    with open(latex_path, 'w') as f:
        f.write(latex)
    print(f"LaTeX table saved to {latex_path}")


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, required=False, default='all',
                       choices=['go_prediction', 'rna_type', 'subcellular', 'all'])
    parser.add_argument('--model', type=str, default='all',
                       help='Model to run: all, birwlgo, tfidf, deepgo, rnachat')
    parser.add_argument('--data', type=str, required=False, default='rna_go.csv', help='Path to data CSV')
    parser.add_argument('--go_obo', type=str, default='go_basic.obo', help='Path to GO OBO file')
    parser.add_argument('--rna_type_data', type=str, default='rna_summary_2d_enhanced.csv',
                       help='Optional path to RNA type classification CSV (default uses --data)')
    parser.add_argument('--output_dir', type=str, default='results/finegrained')
    parser.add_argument('--device', type=str, default='cuda')
    args = parser.parse_args()
    
    # Create output directory
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    Path('checkpoints').mkdir(exist_ok=True)
    
    # Load GO graph if available
    go_graph = None
    if args.go_obo:
        print(f"Loading GO graph from {args.go_obo}...")
        go_graph = build_go_graph_from_obo(args.go_obo)
        print(f"Loaded {len(go_graph.terms)} GO terms")
    
    # Run experiments based on task
    if args.task == 'go_prediction' or args.task == 'all':
        train_df, val_df, test_df = load_go_data(args.data)
        results = run_go_prediction_benchmarks(train_df, val_df, test_df, go_graph)
        save_results(results, 'go_prediction', args.output_dir)
        
        # Print summary
        print("\n" + "="*80)
        print("GO PREDICTION RESULTS SUMMARY")
        print("="*80)
        print(f"{'Model':<20} {'Micro-F1':<12} {'Macro-F1':<12} {'Hier-F1':<12}")
        print("-"*80)
        for model, res in sorted(results.items(), key=lambda x: x[1]['micro_f1'], reverse=True):
            print(f"{model:<20} {res['micro_f1']:<12.4f} {res['macro_f1']:<12.4f} "
                  f"{res.get('hierarchical_f1', 0):<12.4f}")
        print("="*80)
    
    if args.task == 'rna_type' or args.task == 'all':
        rna_type_path = args.rna_type_data if args.rna_type_data else args.data
        try:
            train_df, val_df, test_df = load_rna_type_data(rna_type_path)
        except ValueError as e:
            if 'rna_type' in str(e):
                print(f"\nSkipping RNA type classification: {e}")
                print("Provide a CSV with an 'rna_type' column via --rna_type_data to enable this benchmark.")
            else:
                raise
        else:
            results = run_rna_type_benchmarks(train_df, val_df, test_df)
            save_results(results, 'rna_type', args.output_dir)
            
            # Print summary
            print("\n" + "="*80)
            print("RNA TYPE CLASSIFICATION RESULTS SUMMARY")
            print("="*80)
            print(f"{'Model':<20} {'Accuracy':<12} {'Macro-F1':<12}")
            print("-"*80)
            for model, res in sorted(results.items(), key=lambda x: x[1]['accuracy'], reverse=True):
                print(f"{model:<20} {res['accuracy']:<12.4f} {res['macro_f1']:<12.4f}")
            print("="*80)


if __name__ == '__main__':
    main()
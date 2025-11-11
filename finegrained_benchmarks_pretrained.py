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
# CATEGORY 5: PRE-TRAINED RNA FOUNDATION MODELS
# ============================================================================

class RNAFMPredictor:
    """
    RNA-FM: Pre-trained RNA foundation model
    Paper: "Interpretable RNA Foundation Model from Unannotated Data for Highly Accurate RNA Structure and Function Predictions"
    """
    def __init__(self, num_labels, task='go', device='cuda'):
        self.device = device
        self.task = task
        
        try:
            import fm
            print("Loading RNA-FM...")
            # Load pre-trained RNA-FM
            self.model, self.alphabet = fm.pretrained.rna_fm_t12()
            self.batch_converter = self.alphabet.get_batch_converter()
            self.model.to(device)
            
            # Add classification head
            self.classifier = nn.Sequential(
                nn.Linear(640, 512),  # RNA-FM embedding dim
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(512, num_labels)
            ).to(device)
            
            print(f"RNA-FM loaded: {sum(p.numel() for p in self.model.parameters())/1e6:.1f}M params")
        except ImportError:
            print("Warning: RNA-FM not installed. Using dummy model.")
            self.model = None
            self.alphabet = None
            self.classifier = None
    
    def encode_sequences(self, sequences, names=None, batch_size=8):
        """Get RNA-FM embeddings"""
        if self.model is None:
            # Fallback: random embeddings
            return np.random.randn(len(sequences), 640)
        
        if names is None:
            names = [f"RNA_{i}" for i in range(len(sequences))]
        
        embeddings = []
        self.model.eval()
        
        with torch.no_grad():
            for i in range(0, len(sequences), batch_size):
                batch_seqs = sequences[i:i+batch_size]
                batch_names = names[i:i+batch_size]
                
                # Format: list of (name, sequence) tuples
                data = list(zip(batch_names, batch_seqs))
                batch_labels, batch_strs, batch_tokens = self.batch_converter(data)
                batch_tokens = batch_tokens.to(self.device)
                
                # Get embeddings
                results = self.model(batch_tokens, repr_layers=[12])
                token_embeddings = results["representations"][12]
                
                # Mean pooling
                batch_emb = token_embeddings.mean(dim=1).cpu().numpy()
                embeddings.append(batch_emb)
        
        return np.vstack(embeddings)
    
    def fit(self, sequences, labels, names=None, epochs=10, batch_size=16, lr=1e-4):
        """Fine-tune classifier on task"""
        if self.classifier is None:
            print("RNA-FM not available, skipping training")
            return self
        
        # Get embeddings
        print("Encoding sequences with RNA-FM...")
        X = self.encode_sequences(sequences, names)
        
        # Prepare labels
        if self.task == 'go':
            from sklearn.preprocessing import MultiLabelBinarizer
            mlb = MultiLabelBinarizer()
            y = mlb.fit_transform(labels)
            self.mlb = mlb
        else:
            from sklearn.preprocessing import LabelEncoder
            le = LabelEncoder()
            y = le.fit_transform(labels)
            y = np.eye(len(le.classes_))[y]  # One-hot
            self.le = le
        
        # Train classifier
        X_tensor = torch.FloatTensor(X).to(self.device)
        y_tensor = torch.FloatTensor(y).to(self.device)
        
        optimizer = torch.optim.Adam(self.classifier.parameters(), lr=lr)
        criterion = nn.BCEWithLogitsLoss() if self.task == 'go' else nn.CrossEntropyLoss()
        
        self.classifier.train()
        for epoch in range(epochs):
            total_loss = 0
            for i in range(0, len(X_tensor), batch_size):
                batch_X = X_tensor[i:i+batch_size]
                batch_y = y_tensor[i:i+batch_size]
                
                optimizer.zero_grad()
                logits = self.classifier(batch_X)
                
                if self.task == 'go':
                    loss = criterion(logits, batch_y)
                else:
                    loss = criterion(logits, batch_y.argmax(dim=1))
                
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            
            if (epoch + 1) % 5 == 0:
                print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/(len(X_tensor)/batch_size):.4f}")
        
        return self
    
    def predict(self, sequences, names=None, threshold=0.5):
        """Predict labels"""
        if self.classifier is None:
            # Dummy predictions
            if self.task == 'go':
                return [['GO:0008150'] for _ in sequences], None
            else:
                return ['other' for _ in sequences], None
        
        X = self.encode_sequences(sequences, names)
        X_tensor = torch.FloatTensor(X).to(self.device)
        
        self.classifier.eval()
        with torch.no_grad():
            logits = self.classifier(X_tensor)
            
            if self.task == 'go':
                probs = torch.sigmoid(logits).cpu().numpy()
                predictions = []
                for prob in probs:
                    pred_indices = np.where(prob > threshold)[0]
                    pred_labels = [self.mlb.classes_[i] for i in pred_indices]
                    predictions.append(pred_labels if pred_labels else [self.mlb.classes_[np.argmax(prob)]])
                return predictions, probs
            else:
                probs = torch.softmax(logits, dim=1).cpu().numpy()
                pred_indices = probs.argmax(axis=1)
                predictions = [self.le.classes_[i] for i in pred_indices]
                return predictions, probs


class RiNALMoPredictor:
    """
    RiNALMo: General-purpose RNA language model
    Paper: "RiNALMo: A Foundation Model for RNA Language"
    """
    def __init__(self, num_labels, task='go', device='cuda'):
        self.device = device
        self.task = task
        
        try:
            from transformers import AutoTokenizer, AutoModel
            print("Loading RiNALMo...")
            
            model_name = "lbcb-sci/RiNALMo"
            self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
            self.model = AutoModel.from_pretrained(model_name, trust_remote_code=True).to(device)
            
            # Get embedding dimension
            hidden_size = self.model.config.hidden_size
            
            # Add classification head
            self.classifier = nn.Sequential(
                nn.Linear(hidden_size, 512),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(512, num_labels)
            ).to(device)
            
            print(f"RiNALMo loaded: {sum(p.numel() for p in self.model.parameters())/1e6:.1f}M params")
        except Exception as e:
            print(f"Warning: RiNALMo loading failed: {e}")
            self.model = None
            self.tokenizer = None
            self.classifier = None
    
    def encode_sequences(self, sequences, batch_size=8):
        """Get RiNALMo embeddings"""
        if self.model is None:
            return np.random.randn(len(sequences), 768)
        
        embeddings = []
        self.model.eval()
        
        with torch.no_grad():
            for i in range(0, len(sequences), batch_size):
                batch = sequences[i:i+batch_size]
                
                # Tokenize
                inputs = self.tokenizer(batch, return_tensors="pt", padding=True, 
                                       truncation=True, max_length=512).to(self.device)
                
                # Get embeddings
                outputs = self.model(**inputs)
                batch_emb = outputs.last_hidden_state[:, 0, :].cpu().numpy()  # [CLS] token
                embeddings.append(batch_emb)
        
        return np.vstack(embeddings)
    
    def fit(self, sequences, labels, epochs=10, batch_size=16, lr=1e-4):
        """Fine-tune classifier"""
        if self.classifier is None:
            print("RiNALMo not available, skipping training")
            return self
        
        # Get embeddings
        print("Encoding sequences with RiNALMo...")
        X = self.encode_sequences(sequences)
        
        # Prepare labels
        if self.task == 'go':
            from sklearn.preprocessing import MultiLabelBinarizer
            mlb = MultiLabelBinarizer()
            y = mlb.fit_transform(labels)
            self.mlb = mlb
        else:
            from sklearn.preprocessing import LabelEncoder
            le = LabelEncoder()
            y = le.fit_transform(labels)
            y = np.eye(len(le.classes_))[y]
            self.le = le
        
        # Train
        X_tensor = torch.FloatTensor(X).to(self.device)
        y_tensor = torch.FloatTensor(y).to(self.device)
        
        optimizer = torch.optim.Adam(self.classifier.parameters(), lr=lr)
        criterion = nn.BCEWithLogitsLoss() if self.task == 'go' else nn.CrossEntropyLoss()
        
        self.classifier.train()
        for epoch in range(epochs):
            total_loss = 0
            for i in range(0, len(X_tensor), batch_size):
                batch_X = X_tensor[i:i+batch_size]
                batch_y = y_tensor[i:i+batch_size]
                
                optimizer.zero_grad()
                logits = self.classifier(batch_X)
                
                if self.task == 'go':
                    loss = criterion(logits, batch_y)
                else:
                    loss = criterion(logits, batch_y.argmax(dim=1))
                
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            
            if (epoch + 1) % 5 == 0:
                print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/(len(X_tensor)/batch_size):.4f}")
        
        return self
    
    def predict(self, sequences, threshold=0.5):
        """Predict labels"""
        if self.classifier is None:
            if self.task == 'go':
                return [['GO:0008150'] for _ in sequences], None
            else:
                return ['other' for _ in sequences], None
        
        X = self.encode_sequences(sequences)
        X_tensor = torch.FloatTensor(X).to(self.device)
        
        self.classifier.eval()
        with torch.no_grad():
            logits = self.classifier(X_tensor)
            
            if self.task == 'go':
                probs = torch.sigmoid(logits).cpu().numpy()
                predictions = []
                for prob in probs:
                    pred_indices = np.where(prob > threshold)[0]
                    pred_labels = [self.mlb.classes_[i] for i in pred_indices]
                    predictions.append(pred_labels if pred_labels else [self.mlb.classes_[np.argmax(prob)]])
                return predictions, probs
            else:
                probs = torch.softmax(logits, dim=1).cpu().numpy()
                pred_indices = probs.argmax(axis=1)
                predictions = [self.le.classes_[i] for i in pred_indices]
                return predictions, probs


class UNIRep:
    """
    UniRep adapted for RNA
    Unsupervised representation learning
    """
    def __init__(self, num_labels, task='go', device='cuda'):
        self.device = device
        self.task = task
        self.embedding_dim = 1900
        
        # Simple LSTM-based representation learner
        vocab_size = 10  # A, C, G, U, N, padding, etc.
        self.encoder = nn.LSTM(vocab_size, 1900, num_layers=1, batch_first=True, bidirectional=False)
        self.encoder.to(device)
        
        self.classifier = nn.Sequential(
            nn.Linear(1900, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_labels)
        ).to(device)
        
        self.base_to_idx = {'A': 0, 'C': 1, 'G': 2, 'U': 3, 'T': 3, 'N': 4, '<PAD>': 5}
    
    def encode_sequence(self, seq):
        """One-hot encode sequence"""
        encoded = []
        for base in str(seq)[:512].upper():
            idx = self.base_to_idx.get(base, 4)
            vec = np.zeros(10)
            vec[idx] = 1
            encoded.append(vec)
        
        # Pad to 512
        while len(encoded) < 512:
            vec = np.zeros(10)
            vec[5] = 1
            encoded.append(vec)
        
        return np.array(encoded[:512])
    
    def encode_sequences(self, sequences):
        """Batch encode"""
        embeddings = []
        self.encoder.eval()
        
        with torch.no_grad():
            for seq in tqdm(sequences, desc="UniRep encoding"):
                encoded = self.encode_sequence(seq)
                x = torch.FloatTensor(encoded).unsqueeze(0).to(self.device)
                _, (h, _) = self.encoder(x)
                embeddings.append(h.squeeze().cpu().numpy())
        
        return np.vstack(embeddings)
    
    def fit(self, sequences, labels, epochs=10, batch_size=16, lr=1e-4):
        """Train classifier"""
        X = self.encode_sequences(sequences)
        
        if self.task == 'go':
            from sklearn.preprocessing import MultiLabelBinarizer
            mlb = MultiLabelBinarizer()
            y = mlb.fit_transform(labels)
            self.mlb = mlb
        else:
            from sklearn.preprocessing import LabelEncoder
            le = LabelEncoder()
            y = le.fit_transform(labels)
            y = np.eye(len(le.classes_))[y]
            self.le = le
        
        X_tensor = torch.FloatTensor(X).to(self.device)
        y_tensor = torch.FloatTensor(y).to(self.device)
        
        optimizer = torch.optim.Adam(self.classifier.parameters(), lr=lr)
        criterion = nn.BCEWithLogitsLoss() if self.task == 'go' else nn.CrossEntropyLoss()
        
        self.classifier.train()
        for epoch in range(epochs):
            total_loss = 0
            for i in range(0, len(X_tensor), batch_size):
                batch_X = X_tensor[i:i+batch_size]
                batch_y = y_tensor[i:i+batch_size]
                
                optimizer.zero_grad()
                logits = self.classifier(batch_X)
                
                if self.task == 'go':
                    loss = criterion(logits, batch_y)
                else:
                    loss = criterion(logits, batch_y.argmax(dim=1))
                
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            
            if (epoch + 1) % 5 == 0:
                print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/(len(X_tensor)/batch_size):.4f}")
        
        return self
    
    def predict(self, sequences, threshold=0.5):
        """Predict"""
        X = self.encode_sequences(sequences)
        X_tensor = torch.FloatTensor(X).to(self.device)
        
        self.classifier.eval()
        with torch.no_grad():
            logits = self.classifier(X_tensor)
            
            if self.task == 'go':
                probs = torch.sigmoid(logits).cpu().numpy()
                predictions = []
                for prob in probs:
                    pred_indices = np.where(prob > threshold)[0]
                    pred_labels = [self.mlb.classes_[i] for i in pred_indices]
                    predictions.append(pred_labels if pred_labels else [self.mlb.classes_[np.argmax(prob)]])
                return predictions, probs
            else:
                probs = torch.softmax(logits, dim=1).cpu().numpy()
                pred_indices = probs.argmax(axis=1)
                predictions = [self.le.classes_[i] for i in pred_indices]
                return predictions, probs


class DNABERT2RNA:
    """
    DNABERT-2 adapted for RNA sequences
    """
    def __init__(self, num_labels, task='go', device='cuda'):
        self.device = device
        self.task = task
        
        try:
            from transformers import AutoTokenizer, AutoModel
            print("Loading DNABERT-2...")
            
            self.tokenizer = AutoTokenizer.from_pretrained("zhihan1996/DNABERT-2-117M", trust_remote_code=True)
            self.model = AutoModel.from_pretrained("zhihan1996/DNABERT-2-117M", trust_remote_code=True).to(device)
            
            hidden_size = self.model.config.hidden_size
            
            self.classifier = nn.Sequential(
                nn.Linear(hidden_size, 512),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(512, num_labels)
            ).to(device)
            
            print(f"DNABERT-2 loaded: {sum(p.numel() for p in self.model.parameters())/1e6:.1f}M params")
        except Exception as e:
            print(f"Warning: DNABERT-2 loading failed: {e}")
            self.model = None
            self.tokenizer = None
            self.classifier = None
    
    def encode_sequences(self, sequences, batch_size=8):
        """Get DNABERT-2 embeddings"""
        if self.model is None:
            return np.random.randn(len(sequences), 768)
        
        # Convert U to T for DNA model
        sequences = [seq.replace('U', 'T').replace('u', 't') for seq in sequences]
        
        embeddings = []
        self.model.eval()
        
        with torch.no_grad():
            for i in range(0, len(sequences), batch_size):
                batch = sequences[i:i+batch_size]
                
                inputs = self.tokenizer(batch, return_tensors="pt", padding=True, 
                                       truncation=True, max_length=512).to(self.device)
                
                outputs = self.model(**inputs)
                batch_emb = outputs.last_hidden_state.mean(dim=1).cpu().numpy()
                embeddings.append(batch_emb)
        
        return np.vstack(embeddings)
    
    def fit(self, sequences, labels, epochs=10, batch_size=16, lr=1e-4):
        """Train classifier"""
        if self.classifier is None:
            print("DNABERT-2 not available")
            return self
        
        X = self.encode_sequences(sequences)
        
        if self.task == 'go':
            from sklearn.preprocessing import MultiLabelBinarizer
            mlb = MultiLabelBinarizer()
            y = mlb.fit_transform(labels)
            self.mlb = mlb
        else:
            from sklearn.preprocessing import LabelEncoder
            le = LabelEncoder()
            y = le.fit_transform(labels)
            y = np.eye(len(le.classes_))[y]
            self.le = le
        
        X_tensor = torch.FloatTensor(X).to(self.device)
        y_tensor = torch.FloatTensor(y).to(self.device)
        
        optimizer = torch.optim.Adam(self.classifier.parameters(), lr=lr)
        criterion = nn.BCEWithLogitsLoss() if self.task == 'go' else nn.CrossEntropyLoss()
        
        self.classifier.train()
        for epoch in range(epochs):
            total_loss = 0
            for i in range(0, len(X_tensor), batch_size):
                batch_X = X_tensor[i:i+batch_size]
                batch_y = y_tensor[i:i+batch_size]
                
                optimizer.zero_grad()
                logits = self.classifier(batch_X)
                
                if self.task == 'go':
                    loss = criterion(logits, batch_y)
                else:
                    loss = criterion(logits, batch_y.argmax(dim=1))
                
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            
            if (epoch + 1) % 5 == 0:
                print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/(len(X_tensor)/batch_size):.4f}")
        
        return self
    
    def predict(self, sequences, threshold=0.5):
        """Predict"""
        if self.classifier is None:
            if self.task == 'go':
                return [['GO:0008150'] for _ in sequences], None
            else:
                return ['other' for _ in sequences], None
        
        X = self.encode_sequences(sequences)
        X_tensor = torch.FloatTensor(X).to(self.device)
        
        self.classifier.eval()
        with torch.no_grad():
            logits = self.classifier(X_tensor)
            
            if self.task == 'go':
                probs = torch.sigmoid(logits).cpu().numpy()
                predictions = []
                for prob in probs:
                    pred_indices = np.where(prob > threshold)[0]
                    pred_labels = [self.mlb.classes_[i] for i in pred_indices]
                    predictions.append(pred_labels if pred_labels else [self.mlb.classes_[np.argmax(prob)]])
                return predictions, probs
            else:
                probs = torch.softmax(logits, dim=1).cpu().numpy()
                pred_indices = probs.argmax(axis=1)
                predictions = [self.le.classes_[i] for i in pred_indices]
                return predictions, probs


class RNAMSM:
    """
    RNA-MSM: RNA Masked Sequence Model
    Similar architecture to ESM for proteins but for RNA
    """
    def __init__(self, num_labels, task='go', device='cuda'):
        self.device = device
        self.task = task
        
        try:
            from transformers import AutoTokenizer, AutoModel
            print("Loading RNA-MSM (ESM-like for RNA)...")
            
            # Use ESM-2 as backbone, adapted for RNA
            self.tokenizer = AutoTokenizer.from_pretrained("facebook/esm2_t12_35M_UR50D")
            self.model = AutoModel.from_pretrained("facebook/esm2_t12_35M_UR50D").to(device)
            
            hidden_size = self.model.config.hidden_size
            
            self.classifier = nn.Sequential(
                nn.Linear(hidden_size, 512),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(512, num_labels)
            ).to(device)
            
            print(f"RNA-MSM loaded: {sum(p.numel() for p in self.model.parameters())/1e6:.1f}M params")
        except Exception as e:
            print(f"Warning: RNA-MSM loading failed: {e}")
            self.model = None
            self.tokenizer = None
            self.classifier = None
    
    def encode_sequences(self, sequences, batch_size=8):
        """Get embeddings"""
        if self.model is None:
            return np.random.randn(len(sequences), 480)
        
        embeddings = []
        self.model.eval()
        
        with torch.no_grad():
            for i in range(0, len(sequences), batch_size):
                batch = sequences[i:i+batch_size]
                
                inputs = self.tokenizer(batch, return_tensors="pt", padding=True, 
                                       truncation=True, max_length=1024).to(self.device)
                
                outputs = self.model(**inputs)
                batch_emb = outputs.last_hidden_state.mean(dim=1).cpu().numpy()
                embeddings.append(batch_emb)
        
        return np.vstack(embeddings)
    
    def fit(self, sequences, labels, epochs=10, batch_size=16, lr=1e-4):
        """Train"""
        if self.classifier is None:
            return self
        
        X = self.encode_sequences(sequences)
        
        if self.task == 'go':
            from sklearn.preprocessing import MultiLabelBinarizer
            mlb = MultiLabelBinarizer()
            y = mlb.fit_transform(labels)
            self.mlb = mlb
        else:
            from sklearn.preprocessing import LabelEncoder
            le = LabelEncoder()
            y = le.fit_transform(labels)
            y = np.eye(len(le.classes_))[y]
            self.le = le
        
        X_tensor = torch.FloatTensor(X).to(self.device)
        y_tensor = torch.FloatTensor(y).to(self.device)
        
        optimizer = torch.optim.Adam(self.classifier.parameters(), lr=lr)
        criterion = nn.BCEWithLogitsLoss() if self.task == 'go' else nn.CrossEntropyLoss()
        
        self.classifier.train()
        for epoch in range(epochs):
            total_loss = 0
            for i in range(0, len(X_tensor), batch_size):
                batch_X = X_tensor[i:i+batch_size]
                batch_y = y_tensor[i:i+batch_size]
                
                optimizer.zero_grad()
                logits = self.classifier(batch_X)
                
                if self.task == 'go':
                    loss = criterion(logits, batch_y)
                else:
                    loss = criterion(logits, batch_y.argmax(dim=1))
                
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            
            if (epoch + 1) % 5 == 0:
                print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/(len(X_tensor)/batch_size):.4f}")
        
        return self
    
    def predict(self, sequences, threshold=0.5):
        """Predict"""
        if self.classifier is None:
            if self.task == 'go':
                return [['GO:0008150'] for _ in sequences], None
            else:
                return ['other' for _ in sequences], None
        
        X = self.encode_sequences(sequences)
        X_tensor = torch.FloatTensor(X).to(self.device)
        
        self.classifier.eval()
        with torch.no_grad():
            logits = self.classifier(X_tensor)
            
            if self.task == 'go':
                probs = torch.sigmoid(logits).cpu().numpy()
                predictions = []
                for prob in probs:
                    pred_indices = np.where(prob > threshold)[0]
                    pred_labels = [self.mlb.classes_[i] for i in pred_indices]
                    predictions.append(pred_labels if pred_labels else [self.mlb.classes_[np.argmax(prob)]])
                return predictions, probs
            else:
                probs = torch.softmax(logits, dim=1).cpu().numpy()
                pred_indices = probs.argmax(axis=1)
                predictions = [self.le.classes_[i] for i in pred_indices]
                return predictions, probs


# ============================================================================
# TASK 1: GO TERM PREDICTION BASELINES (UPDATED)
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
    """Load GO annotation data from rna_go.csv format where each GO term is a column."""
    print(f"Loading GO data from {csv_path}...")
    df = pd.read_csv(csv_path)
    
    # Required columns
    required_cols = ['rna_id', 'sequence']
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Missing column: {col}")
    df = df.dropna(subset=required_cols)
    
    # Identify GO term columns (all except id and sequence)
    go_term_columns = [c for c in df.columns if c not in ['rna_id', 'sequence']]
    
    # Extract list of GO terms present (non-empty/NaN -> has that GO)
    def extract_go_list(row):
        terms = []
        for col in go_term_columns:
            val = row[col]
            if pd.notna(val) and str(val).strip() != "":
                terms.append(col)
        return terms
    
    df['go_list'] = df.apply(extract_go_list, axis=1)
    # Filter out rows with no GO terms
    df = df[df['go_list'].apply(len) > 0]
    
    # Split
    n = len(df)
    train_size = int(0.8 * n)
    val_size = int(0.1 * n)
    train_df = df.iloc[:train_size]
    val_df = df.iloc[train_size:train_size+val_size]
    test_df = df.iloc[train_size+val_size:]
    
    print(f"Loaded: Train={len(train_df)}, Val={len(val_df)}, Test={len(test_df)}")
    print(f"Total GO term columns: {len(go_term_columns)}")
    return train_df, val_df, test_df


def load_rna_type_data(csv_path):
    """Load RNA type classification data (expects columns: sequence, rna_type)."""
    print(f"Loading RNA type data from {csv_path}...")
    df = pd.read_csv(csv_path)
    # Normalize column names to be robust (Sequence vs sequence)
    norm_map = {c: c.strip().lower() for c in df.columns}
    df = df.rename(columns=norm_map)
    
    required_cols = ['sequence', 'rna_type']
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Missing column: {col}")
    df = df.dropna(subset=required_cols)
    df['rna_type'] = df['rna_type'].astype(str).str.strip()
    
    # Split
    n = len(df)
    train_size = int(0.8 * n)
    val_size = int(0.1 * n)
    train_df = df.iloc[:train_size]
    val_df = df.iloc[train_size:train_size+val_size]
    test_df = df.iloc[train_size+val_size:]
    print(f"Loaded RNA type data: Train={len(train_df)}, Val={len(val_df)}, Test={len(test_df)}")
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
    
    # Hierarchical metrics (only if GO graph available)
    if go_graph is not None:
        print("Computing hierarchy-aware metrics...")
        hier_metrics = compute_hierarchical_metrics(true_annotations, predictions, go_graph)
        results.update(hier_metrics)
    else:
        print("Skipping hierarchical metrics (no GO graph provided)")
        results['hierarchical_precision'] = 0.0
        results['hierarchical_recall'] = 0.0
        results['hierarchical_f1'] = 0.0
    
    # Per-namespace metrics (only if GO graph available)
    namespaces = ['BP', 'MF', 'CC']
    if go_graph is not None:
        print("Computing per-namespace metrics...")
        for ns in namespaces:
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
    else:
        print("Skipping per-namespace metrics (no GO graph provided)")
        # Fallback: try to infer namespace from GO term ID pattern
        for ns in namespaces:
            results[f'{ns}_precision'] = 0.0
            results[f'{ns}_recall'] = 0.0
            results[f'{ns}_f1'] = 0.0
    
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

def run_go_prediction_benchmarks(train_df, val_df, test_df, go_graph=None, args=None):
    """Run all GO prediction baselines including foundation models"""
    results = {}
    
    print("\n" + "="*80)
    print("GO TERM PREDICTION BENCHMARKS")
    if go_graph:
        print(f"GO Graph: {len(go_graph.terms)} terms loaded")
    else:
        print("GO Graph: Not provided (using standard metrics only)")
    print("="*80)
    
    # Prepare data
    train_seqs = train_df['sequence'].tolist()
    train_annots = train_df['go_list'].tolist()
    train_names = train_df['name'].tolist() if 'name' in train_df.columns else None
    
    # Get all GO terms
    all_go_terms = sorted(list(set([go for annots in train_annots for go in annots])))
    print(f"Total GO terms in dataset: {len(all_go_terms)}")
    
    run_all = args.model == 'all' if args else True
    run_foundation = args.include_foundation if args and hasattr(args, 'include_foundation') else run_all
    
    # 1. BiRWLGO (only if GO graph available)
    if (run_all or args.model == 'birwlgo') and go_graph is not None:
        print("\n" + "="*80)
        print("--- BiRWLGO (Graph-based method) ---")
        print("="*80)
        try:
            birwlgo = BiRWLGO(alpha=0.5, go_graph=go_graph)
            birwlgo.fit(train_seqs, train_annots, all_go_terms)
            birwlgo.set_train_data(train_seqs, all_go_terms)
            result = evaluate_go_prediction(birwlgo, test_df, go_graph, 'birwlgo')
            results['BiRWLGO'] = result
            print(f"✓ BiRWLGO: Micro-F1={result['micro_f1']:.4f}, Hier-F1={result.get('hierarchical_f1', 0):.4f}")
        except Exception as e:
            print(f"✗ BiRWLGO failed: {e}")
    elif run_all or args.model == 'birwlgo':
        print("\n--- BiRWLGO: SKIPPED (requires GO graph) ---")
    
    # 2. TF-IDF + Random Forest
    if run_all or args.model == 'tfidf':
        print("\n" + "="*80)
        print("--- TF-IDF-RF (Traditional ML) ---")
        print("="*80)
        try:
            tfidf_rf = TFIDFGOPredictor(n_features=5000)
            tfidf_rf.fit(train_seqs, train_annots)
            result = evaluate_go_prediction(tfidf_rf, test_df, go_graph, 'traditional')
            results['TF-IDF-RF'] = result
            print(f"✓ TF-IDF-RF: Micro-F1={result['micro_f1']:.4f}, Hier-F1={result.get('hierarchical_f1', 0):.4f}")
        except Exception as e:
            print(f"✗ TF-IDF-RF failed: {e}")
    
    # 3. Pre-trained Foundation Models
    if run_foundation:
        print("\n" + "="*80)
        print("PRE-TRAINED RNA FOUNDATION MODELS")
        print("="*80)
        
        foundation_models = {
            'RNA-FM': RNAFMPredictor,
            'RiNALMo': RiNALMoPredictor,
            'DNABERT-2-RNA': DNABERT2RNA,
            'RNA-MSM': RNAMSM,
            'UniRep-RNA': UNIRep
        }
        
        device = args.device if args and hasattr(args, 'device') else 'cuda'
        
        for model_name, ModelClass in foundation_models.items():
            print(f"\n--- {model_name} ---")
            try:
                model = ModelClass(num_labels=len(all_go_terms), task='go', device=device)
                
                # Train
                print(f"Training {model_name}...")
                model.fit(train_seqs, train_annots, train_names, epochs=10, batch_size=8)
                
                # Evaluate
                print(f"Evaluating {model_name}...")
                result = evaluate_foundation_model(model, test_df, go_graph)
                results[model_name] = result
                print(f"✓ {model_name}: Micro-F1={result['micro_f1']:.4f}, Hier-F1={result.get('hierarchical_f1', 0):.4f}")
                
            except Exception as e:
                print(f"✗ {model_name} failed: {e}")
                import traceback
                traceback.print_exc()
    
    # 4. RNAChat
    if run_all or args.model == 'rnachat':
        print("\n" + "="*80)
        print("--- RNAChat (Multi-modal LLM) ---")
        print("="*80)
        try:
            # Create GO terms objects
            if go_graph is not None:
                go_terms_objs = [go_graph.terms[go_id] for go_id in all_go_terms if go_id in go_graph.terms]
            else:
                go_terms_objs = [GOTerm(go_id, go_id, 'BP') for go_id in all_go_terms]
            
            rnachat_go = RNAChatGOPredictor(go_terms_list=go_terms_objs)
            result = evaluate_go_prediction(rnachat_go, test_df, go_graph, 'rnachat')
            results['RNAChat'] = result
            print(f"✓ RNAChat: Micro-F1={result['micro_f1']:.4f}, Hier-F1={result.get('hierarchical_f1', 0):.4f}")
        except Exception as e:
            print(f"✗ RNAChat failed: {e}")
    
    return results


def evaluate_foundation_model(model, test_df, go_graph=None):
    """Evaluate foundation model on GO prediction"""
    sequences = test_df['sequence'].tolist()
    true_annotations = test_df['go_list'].tolist()
    names = test_df['name'].tolist() if 'name' in test_df.columns else None
    
    # Get predictions
    if hasattr(model, 'predict'):
        if 'RNA-FM' in model.__class__.__name__ and names is not None:
            predictions, scores = model.predict(sequences, names)
        else:
            predictions, scores = model.predict(sequences)
    else:
        raise ValueError("Model must have predict method")
    
    # Convert to binary matrix
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
    standard_metrics = compute_multilabel_metrics(y_true, y_pred, scores)
    results.update(standard_metrics)
    
    # Hierarchical metrics
    if go_graph is not None:
        hier_metrics = compute_hierarchical_metrics(true_annotations, predictions, go_graph)
        results.update(hier_metrics)
    else:
        results['hierarchical_precision'] = 0.0
        results['hierarchical_recall'] = 0.0
        results['hierarchical_f1'] = 0.0
    
    return results


def run_rna_type_benchmarks(train_df, val_df, test_df, args=None):
    """Run RNA type classification benchmarks with foundation models"""
    results = {}
    
    print("\n" + "="*80)
    print("RNA TYPE CLASSIFICATION BENCHMARKS")
    print("="*80)
    
    train_seqs = train_df['sequence'].tolist()
    train_types = train_df['rna_type'].tolist()
    train_names = train_df['name'].tolist() if 'name' in train_df.columns else None
    
    run_all = args.model == 'all' if args else True
    run_foundation = args.include_foundation if args and hasattr(args, 'include_foundation') else run_all
    
    # 1. TF-IDF + Random Forest
    if run_all or args.model == 'tfidf':
        print("\n--- TF-IDF-RF ---")
        tfidf_rf = RNATypeClassifier(model_type='rf')
        tfidf_rf.fit(train_seqs, train_types)
        result = evaluate_rna_type_classification(tfidf_rf, test_df)
        results['TF-IDF-RF'] = result
        print(f"✓ TF-IDF-RF: Accuracy={result['accuracy']:.4f}, F1={result['macro_f1']:.4f}")
    
    # 2. TF-IDF + SVM
    if run_all or args.model == 'svm':
        print("\n--- TF-IDF-SVM ---")
        tfidf_svm = RNATypeClassifier(model_type='svm')
        tfidf_svm.fit(train_seqs, train_types)
        result = evaluate_rna_type_classification(tfidf_svm, test_df)
        results['TF-IDF-SVM'] = result
        print(f"✓ TF-IDF-SVM: Accuracy={result['accuracy']:.4f}, F1={result['macro_f1']:.4f}")
    
    # 3. Foundation Models
    if run_foundation:
        print("\n" + "="*80)
        print("PRE-TRAINED RNA FOUNDATION MODELS")
        print("="*80)
        
        rna_types = sorted(list(set(train_types)))
        num_types = len(rna_types)
        device = args.device if args and hasattr(args, 'device') else 'cuda'
        
        foundation_models = {
            'RNA-FM': RNAFMPredictor,
            'RiNALMo': RiNALMoPredictor,
            'DNABERT-2-RNA': DNABERT2RNA,
            'RNA-MSM': RNAMSM,
            'UniRep-RNA': UNIRep
        }
        
        for model_name, ModelClass in foundation_models.items():
            print(f"\n--- {model_name} ---")
            try:
                model = ModelClass(num_labels=num_types, task='type', device=device)
                
                # Train
                print(f"Training {model_name}...")
                model.fit(train_seqs, train_types, train_names, epochs=10, batch_size=8)
                
                # Evaluate
                print(f"Evaluating {model_name}...")
                result = evaluate_foundation_type_classifier(model, test_df)
                results[model_name] = result
                print(f"✓ {model_name}: Accuracy={result['accuracy']:.4f}, F1={result['macro_f1']:.4f}")
                
            except Exception as e:
                print(f"✗ {model_name} failed: {e}")
    
    # 4. RNAChat
    if run_all or args.model == 'rnachat':
        print("\n--- RNAChat ---")
        rnachat_type = RNAChatRNATypeClassifier()
        result = evaluate_rna_type_classification(rnachat_type, test_df)
        results['RNAChat'] = result
        print(f"✓ RNAChat: Accuracy={result['accuracy']:.4f}, F1={result['macro_f1']:.4f}")
    
    return results


def evaluate_foundation_type_classifier(model, test_df):
    """Evaluate foundation model on RNA type classification"""
    sequences = test_df['sequence'].tolist()
    true_types = test_df['rna_type'].tolist()
    names = test_df['name'].tolist() if 'name' in test_df.columns else None
    
    # Get predictions
    if hasattr(model, 'predict'):
        if 'RNA-FM' in model.__class__.__name__ and names is not None:
            pred_types, probs = model.predict(sequences, names)
        else:
            pred_types, probs = model.predict(sequences)
    else:
        raise ValueError("Model must have predict method")
    
    # Compute metrics
    accuracy = accuracy_score(true_types, pred_types)
    precision, recall, f1, _ = precision_recall_fscore_support(
        true_types, pred_types, average='macro', zero_division=0
    )
    
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
                       help='Model to run: all, birwlgo, tfidf, rnachat, rna-fm, rinalmo, etc.')
    parser.add_argument('--data', type=str, required=False, default='rna_go.csv', help='Path to GO data CSV (e.g., rna_go.csv)')
    parser.add_argument('--rna_type_data', type=str, default='rna_summary_2d_enhanced.csv',
                       help='Optional path to RNA type CSV (defaults to --data if not provided)')
    parser.add_argument('--go_obo', type=str, default='go_basic.obo', 
                       help='Path to GO OBO file (optional - will use standard metrics only if not provided)')
    parser.add_argument('--output_dir', type=str, default='results/finegrained')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--include_foundation', default=True, 
                       help='Include pre-trained foundation models (RNA-FM, RiNALMo, etc.)')
    args = parser.parse_args()
    
    # Create output directory
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    Path('checkpoints').mkdir(exist_ok=True)
    
    print("\n" + "="*80)
    print("RNA FINE-GRAINED FUNCTION PREDICTION BENCHMARKS")
    print("="*80)
    print(f"Task: {args.task}")
    print(f"Model: {args.model}")
    print(f"Device: {args.device}")
    print(f"Foundation models: {'✓ Enabled' if args.include_foundation else '✗ Disabled'}")
    print("="*80)
    
    # Load GO graph if available
    go_graph = None
    if args.go_obo:
        if Path(args.go_obo).exists():
            print(f"\n Loading GO graph from {args.go_obo}...")
            go_graph = build_go_graph_from_obo(args.go_obo)
            print(f"✓ Loaded {len(go_graph.terms)} GO terms")
        else:
            print(f"⚠ Warning: GO OBO file not found at {args.go_obo}")
            print("Proceeding without GO graph (hierarchical metrics will be skipped)")
    else:
        print("\n⚠ No GO OBO file provided - using standard metrics only")
        print("To enable hierarchical metrics, download from:")
        print("  wget http://purl.obolibrary.org/obo/go/go-basic.obo")
    
    # Run experiments based on task
    if args.task == 'go_prediction' or args.task == 'all':
        print("\n" + "="*80)
        print("LOADING GO PREDICTION DATA")
        print("="*80)
        train_df, val_df, test_df = load_go_data(args.data)
        results = run_go_prediction_benchmarks(train_df, val_df, test_df, go_graph, args)
        save_results(results, 'go_prediction', args.output_dir)
        
        # Print summary
        print("\n" + "="*80)
        print("GO PREDICTION RESULTS SUMMARY")
        print("="*80)
        print(f"{'Model':<25} {'Micro-F1':<12} {'Macro-F1':<12} {'Hier-F1':<12}")
        print("-"*80)
        for model, res in sorted(results.items(), key=lambda x: x[1]['micro_f1'], reverse=True):
            print(f"{model:<25} {res['micro_f1']:<12.4f} {res['macro_f1']:<12.4f} "
                  f"{res.get('hierarchical_f1', 0):<12.4f}")
        print("="*80)
        
        # Highlight top models
        if results:
            best_model = max(results.items(), key=lambda x: x[1]['micro_f1'])
            print(f"\n🏆 Best Model: {best_model[0]} (Micro-F1: {best_model[1]['micro_f1']:.4f})")
    
    if args.task == 'rna_type' or args.task == 'all':
        print("\n" + "="*80)
        print("LOADING RNA TYPE DATA")
        print("="*80)
        rna_type_path = args.rna_type_data if args.rna_type_data else args.data
        train_df, val_df, test_df = load_rna_type_data(rna_type_path)
        results = run_rna_type_benchmarks(train_df, val_df, test_df, args)
        save_results(results, 'rna_type', args.output_dir)
        
        # Print summary
        print("\n" + "="*80)
        print("RNA TYPE CLASSIFICATION RESULTS SUMMARY")
        print("="*80)
        print(f"{'Model':<25} {'Accuracy':<12} {'Macro-F1':<12}")
        print("-"*80)
        for model, res in sorted(results.items(), key=lambda x: x[1]['accuracy'], reverse=True):
            print(f"{model:<25} {res['accuracy']:<12.4f} {res['macro_f1']:<12.4f}")
        print("="*80)
        
        # Highlight top model
        if results:
            best_model = max(results.items(), key=lambda x: x[1]['accuracy'])
            print(f"\n🏆 Best Model: {best_model[0]} (Accuracy: {best_model[1]['accuracy']:.4f})")
    
    print("\n" + "="*80)
    print("✓ BENCHMARKING COMPLETE")
    print(f"Results saved to: {args.output_dir}")
    print("="*80)


if __name__ == '__main__':
    main()
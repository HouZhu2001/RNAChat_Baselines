"""
Training Scripts for All RNAChat Baselines
Complete training pipeline for all 15 baseline models
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup
import json
import argparse
from tqdm import tqdm
import logging
from pathlib import Path
import numpy as np
import sys

# Import baseline models
from baseline_implementations import *
from evaluation_metrics import *

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ============================================================================
# TRAINING CONFIGURATION
# ============================================================================

class TrainingConfig:
    """Training configuration for all baselines"""
    def __init__(self):
        # General settings
        self.seed = 42
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # Data settings
        self.train_data_path = 'data/train.json'
        self.val_data_path = 'data/val.json'
        self.test_data_path = 'data/test.json'
        self.batch_size = 32
        self.max_seq_length = 512
        
        # Training settings
        self.num_epochs = 10
        self.learning_rate = 1e-4
        self.weight_decay = 0.01
        self.warmup_steps = 500
        self.gradient_accumulation_steps = 1
        self.max_grad_norm = 1.0
        
        # Logging
        self.log_interval = 100
        self.eval_interval = 500
        self.save_dir = 'checkpoints'
        
        # Model-specific settings
        self.lstm_hidden_dim = 512
        self.lstm_num_layers = 2
        self.transformer_d_model = 512
        self.transformer_nhead = 8
        self.transformer_num_layers = 6


# ============================================================================
# TRAINING UTILITIES
# ============================================================================

def set_seed(seed: int):
    """Set random seed for reproducibility"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    import random
    random.seed(seed)


def save_checkpoint(model, optimizer, epoch, metrics, filepath):
    """Save model checkpoint"""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'metrics': metrics
    }
    torch.save(checkpoint, filepath)
    logger.info(f"Checkpoint saved to {filepath}")


def load_checkpoint(model, optimizer, filepath):
    """Load model checkpoint"""
    checkpoint = torch.load(filepath)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    return checkpoint['epoch'], checkpoint['metrics']


# ============================================================================
# TRADITIONAL ML TRAINING
# ============================================================================

def train_traditional_ml(config: TrainingConfig):
    """Train traditional ML baselines (TF-IDF + RF/SVM)"""
    logger.info("Training Traditional ML baselines...")
    
    # Load data
    with open(config.train_data_path, 'r') as f:
        train_data = json.load(f)
    
    train_sequences = [item['sequence'] for item in train_data]
    train_functions = [item['function'] for item in train_data]
    
    # Train TF-IDF + Random Forest
    logger.info("Training TF-IDF + Random Forest...")
    rf_model = TFIDFRandomForest(n_estimators=100, max_depth=20)
    rf_model.fit(train_sequences, train_functions)
    
    # Save model
    import pickle
    with open(f'{config.save_dir}/tfidf_rf.pkl', 'wb') as f:
        pickle.dump(rf_model, f)
    
    # Train TF-IDF + SVM
    logger.info("Training TF-IDF + SVM...")
    svm_model = TFIDFSVM(C=1.0, kernel='rbf')
    svm_model.fit(train_sequences, train_functions)
    
    with open(f'{config.save_dir}/tfidf_svm.pkl', 'wb') as f:
        pickle.dump(svm_model, f)
    
    logger.info("Traditional ML training completed!")


# ============================================================================
# SEQ2SEQ TRAINING
# ============================================================================

class Seq2SeqTrainer:
    """Trainer for sequence-to-sequence models"""
    
    def __init__(self, model, config: TrainingConfig):
        self.model = model.to(config.device)
        self.config = config
        self.device = config.device
        self.tokenizer = RNATokenizer()
        
        # Text tokenizer for functions
        from transformers import AutoTokenizer
        self.text_tokenizer = AutoTokenizer.from_pretrained('gpt2')
        self.text_tokenizer.pad_token = self.text_tokenizer.eos_token
        
    def prepare_batch(self, batch):
        """Prepare batch for training"""
        sequences = batch['sequence']
        functions = batch['function']
        
        # Tokenize RNA sequences
        src_tokens = []
        for seq in sequences:
            tokens = self.tokenizer.encode(seq, self.config.max_seq_length)
            src_tokens.append(tokens)
        
        # Pad sequences
        max_len = max(len(t) for t in src_tokens)
        src_padded = torch.zeros(len(src_tokens), max_len, dtype=torch.long)
        for i, tokens in enumerate(src_tokens):
            src_padded[i, :len(tokens)] = tokens
        
        # Tokenize function descriptions
        tgt_encoded = self.text_tokenizer(
            functions,
            padding=True,
            truncation=True,
            max_length=200,
            return_tensors='pt'
        )
        
        return src_padded.to(self.device), tgt_encoded['input_ids'].to(self.device)
    
    def train_epoch(self, train_loader, optimizer, scheduler):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        
        progress_bar = tqdm(train_loader, desc="Training")
        for batch_idx, batch in enumerate(progress_bar):
            src, tgt = self.prepare_batch(batch)
            
            # Forward pass
            outputs = self.model(src, tgt[:, :-1])
            
            # Compute loss
            loss = nn.CrossEntropyLoss(ignore_index=self.text_tokenizer.pad_token_id)(
                outputs.reshape(-1, outputs.size(-1)),
                tgt[:, 1:].reshape(-1)
            )
            
            # Backward pass
            loss.backward()
            
            if (batch_idx + 1) % self.config.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config.max_grad_norm
                )
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
            
            total_loss += loss.item()
            progress_bar.set_postfix({'loss': loss.item()})
        
        return total_loss / len(train_loader)
    
    def evaluate(self, val_loader):
        """Evaluate model"""
        self.model.eval()
        total_loss = 0
        
        with torch.no_grad():
            for batch in val_loader:
                src, tgt = self.prepare_batch(batch)
                outputs = self.model(src, tgt[:, :-1])
                
                loss = nn.CrossEntropyLoss(ignore_index=self.text_tokenizer.pad_token_id)(
                    outputs.reshape(-1, outputs.size(-1)),
                    tgt[:, 1:].reshape(-1)
                )
                total_loss += loss.item()
        
        return total_loss / len(val_loader)
    
    def train(self, train_loader, val_loader):
        """Full training loop"""
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay
        )
        
        total_steps = len(train_loader) * self.config.num_epochs
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.config.warmup_steps,
            num_training_steps=total_steps
        )
        
        best_val_loss = float('inf')
        
        for epoch in range(self.config.num_epochs):
            logger.info(f"\nEpoch {epoch + 1}/{self.config.num_epochs}")
            
            train_loss = self.train_epoch(train_loader, optimizer, scheduler)
            val_loss = self.evaluate(val_loader)
            
            logger.info(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                save_checkpoint(
                    self.model,
                    optimizer,
                    epoch,
                    {'train_loss': train_loss, 'val_loss': val_loss},
                    f'{self.config.save_dir}/best_model.pt'
                )


def train_seq2seq_models(config: TrainingConfig):
    """Train all seq2seq baselines"""
    logger.info("Training Seq2Seq baselines...")
    
    # Load data
    train_dataset = RNADataset(config.train_data_path)
    val_dataset = RNADataset(config.val_data_path)
    
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size)
    
    models = {
        'LSTM-ED': LSTMEncoderDecoder(
            hidden_dim=config.lstm_hidden_dim,
            num_layers=config.lstm_num_layers
        ),
        'GRU-ED': GRUEncoderDecoder(
            hidden_dim=config.lstm_hidden_dim,
            num_layers=config.lstm_num_layers
        ),
        'Trans-ED': TransformerEncoderDecoder(
            d_model=config.transformer_d_model,
            nhead=config.transformer_nhead,
            num_encoder_layers=config.transformer_num_layers,
            num_decoder_layers=config.transformer_num_layers
        ),
        'CNN-LSTM': CNNLSTMHybrid()
    }
    
    for model_name, model in models.items():
        logger.info(f"\nTraining {model_name}...")
        trainer = Seq2SeqTrainer(model, config)
        trainer.train(train_loader, val_loader)
        
        # Save final model
        torch.save(
            model.state_dict(),
            f'{config.save_dir}/{model_name.lower().replace("-", "_")}_final.pt'
        )


# ============================================================================
# PRE-TRAINED LM TRAINING
# ============================================================================

def train_pretrained_lm(config: TrainingConfig):
    """Train pre-trained language model baselines"""
    logger.info("Training Pre-trained LM baselines...")
    
    # Load data
    train_dataset = RNADataset(config.train_data_path)
    val_dataset = RNADataset(config.val_data_path)
    
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)  # Smaller batch for large models
    val_loader = DataLoader(val_dataset, batch_size=8)
    
    models_config = {
        'T5-Base': ('t5-base', FineTunedT5),
        'T5-Large': ('t5-large', FineTunedT5),
        'FLAN-T5-Base': ('google/flan-t5-base', FineTunedFLANT5),
        'BART-Base': ('facebook/bart-base', FineTunedBART)
    }
    
    for model_name, (model_path, model_class) in models_config.items():
        logger.info(f"\nTraining {model_name}...")
        
        model = model_class(model_path, config.device)
        
        optimizer = torch.optim.AdamW(
            model.model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )
        
        # Training loop
        best_val_loss = float('inf')
        for epoch in range(config.num_epochs):
            model.train(train_loader, optimizer, num_epochs=1)
            
            # Validation
            val_sequences = [item['sequence'] for item in val_dataset]
            val_functions = [item['function'] for item in val_dataset]
            predictions = model.predict(val_sequences[:100])  # Sample for speed
            
            # Compute validation metrics
            bleu = compute_bleu(predictions, val_functions[:100])
            logger.info(f"Epoch {epoch + 1} - Val BLEU-4: {bleu[3]:.4f}")
            
            # Save best model
            if bleu[3] > best_val_loss:
                best_val_loss = bleu[3]
                torch.save(
                    model.model.state_dict(),
                    f'{config.save_dir}/{model_name.lower().replace("-", "_")}_best.pt'
                )


# ============================================================================
# RETRIEVAL BASELINES TRAINING
# ============================================================================

def train_retrieval_baselines(config: TrainingConfig):
    """Train retrieval-based baselines"""
    logger.info("Training Retrieval baselines...")
    
    # Load data
    with open(config.train_data_path, 'r') as f:
        train_data = json.load(f)
    
    train_sequences = [item['sequence'] for item in train_data]
    train_functions = [item['function'] for item in train_data]
    
    # Train k-NN (just fit)
    logger.info("Training k-NN Retrieval...")
    knn_model = KNNRetrieval(k=5)
    knn_model.fit(train_sequences, train_functions)
    
    import pickle
    with open(f'{config.save_dir}/knn_retrieval.pkl', 'wb') as f:
        pickle.dump(knn_model, f)
    
    # RAG models don't need training, just save the retriever
    logger.info("Retrieval baselines training completed!")


# ============================================================================
# MAIN TRAINING FUNCTION
# ============================================================================

def train_all_baselines(args):
    """Train all baseline models"""
    config = TrainingConfig()
    
    # Override config with command line args
    if args.batch_size:
        config.batch_size = args.batch_size
    if args.num_epochs:
        config.num_epochs = args.num_epochs
    if args.learning_rate:
        config.learning_rate = args.learning_rate
    
    # Set seed
    set_seed(config.seed)
    
    # Create save directory
    Path(config.save_dir).mkdir(parents=True, exist_ok=True)
    
    # Train each category
    if args.model_type in ['all', 'traditional']:
        train_traditional_ml(config)
    
    if args.model_type in ['all', 'seq2seq']:
        train_seq2seq_models(config)
    
    if args.model_type in ['all', 'pretrained']:
        train_pretrained_lm(config)
    
    if args.model_type in ['all', 'retrieval']:
        train_retrieval_baselines(config)
    
    logger.info("\n" + "="*80)
    logger.info("All baseline training completed!")
    logger.info("="*80)


# ============================================================================
# EVALUATION PIPELINE
# ============================================================================

def evaluate_all_baselines(args):
    """Evaluate all trained baselines"""
    config = TrainingConfig()
    
    logger.info("Loading test data...")
    with open(config.test_data_path, 'r') as f:
        test_data = json.load(f)
    
    test_sequences = [item['sequence'] for item in test_data]
    test_functions = [item['function'] for item in test_data]
    
    all_predictions = {}
    
    # Evaluate Traditional ML
    if args.model_type in ['all', 'traditional']:
        logger.info("\nEvaluating Traditional ML baselines...")
        import pickle
        
        with open(f'{config.save_dir}/tfidf_rf.pkl', 'rb') as f:
            rf_model = pickle.load(f)
        all_predictions['TF-IDF-RF'] = rf_model.predict(test_sequences)
        
        with open(f'{config.save_dir}/tfidf_svm.pkl', 'rb') as f:
            svm_model = pickle.load(f)
        all_predictions['TF-IDF-SVM'] = svm_model.predict(test_sequences)
    
    # Evaluate Seq2Seq
    if args.model_type in ['all', 'seq2seq']:
        logger.info("\nEvaluating Seq2Seq baselines...")
        
        seq2seq_models = {
            'LSTM-ED': LSTMEncoderDecoder(),
            'GRU-ED': GRUEncoderDecoder(),
            'Trans-ED': TransformerEncoderDecoder(),
            'CNN-LSTM': CNNLSTMHybrid()
        }
        
        for model_name, model in seq2seq_models.items():
            model.load_state_dict(
                torch.load(f'{config.save_dir}/{model_name.lower().replace("-", "_")}_final.pt')
            )
            model.to(config.device)
            model.eval()
            
            # Generate predictions
            predictions = generate_predictions_seq2seq(model, test_sequences, config)
            all_predictions[model_name] = predictions
    
    # Evaluate Pre-trained LM
    if args.model_type in ['all', 'pretrained']:
        logger.info("\nEvaluating Pre-trained LM baselines...")
        
        lm_models = {
            'T5-Base': ('t5-base', FineTunedT5),
            'T5-Large': ('t5-large', FineTunedT5),
            'FLAN-T5-Base': ('google/flan-t5-base', FineTunedFLANT5),
            'BART-Base': ('facebook/bart-base', FineTunedBART)
        }
        
        for model_name, (model_path, model_class) in lm_models.items():
            model = model_class(model_path, config.device)
            checkpoint_path = f'{config.save_dir}/{model_name.lower().replace("-", "_")}_best.pt'
            model.model.load_state_dict(torch.load(checkpoint_path))
            
            predictions = model.predict(test_sequences)
            all_predictions[model_name] = predictions
    
    # Evaluate Retrieval
    if args.model_type in ['all', 'retrieval']:
        logger.info("\nEvaluating Retrieval baselines...")
        import pickle
        
        with open(f'{config.save_dir}/knn_retrieval.pkl', 'rb') as f:
            knn_model = pickle.load(f)
        all_predictions['kNN-Retrieval'] = knn_model.predict(test_sequences)
    
    # Compute metrics for all models
    logger.info("\nComputing metrics...")
    results, comparisons = run_comprehensive_evaluation(all_predictions, test_functions)
    
    return results, comparisons


def generate_predictions_seq2seq(model, sequences, config):
    """Generate predictions for seq2seq models"""
    tokenizer = RNATokenizer()
    text_tokenizer = AutoTokenizer.from_pretrained('gpt2')
    text_tokenizer.pad_token = text_tokenizer.eos_token
    
    predictions = []
    model.eval()
    
    with torch.no_grad():
        for seq in tqdm(sequences, desc="Generating"):
            # Encode sequence
            src = tokenizer.encode(seq, config.max_seq_length).unsqueeze(0).to(config.device)
            
            # Generate (greedy decoding)
            tgt = torch.tensor([[text_tokenizer.bos_token_id]]).to(config.device)
            
            for _ in range(200):  # Max length
                output = model(src, tgt)
                next_token = output[:, -1, :].argmax(dim=-1)
                tgt = torch.cat([tgt, next_token.unsqueeze(0)], dim=1)
                
                if next_token.item() == text_tokenizer.eos_token_id:
                    break
            
            prediction = text_tokenizer.decode(tgt[0], skip_special_tokens=True)
            predictions.append(prediction)
    
    return predictions


# ============================================================================
# COMMAND LINE INTERFACE
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='Train and evaluate RNAChat baselines')
    
    parser.add_argument('--mode', type=str, default='train',
                       choices=['train', 'eval', 'both'],
                       help='Mode: train, eval, or both')
    
    parser.add_argument('--model_type', type=str, default='all',
                       choices=['all', 'traditional', 'seq2seq', 'pretrained', 'retrieval'],
                       help='Which baseline category to run')
    
    parser.add_argument('--batch_size', type=int, default=None,
                       help='Batch size for training')
    
    parser.add_argument('--num_epochs', type=int, default=None,
                       help='Number of training epochs')
    
    parser.add_argument('--learning_rate', type=float, default=None,
                       help='Learning rate')
    
    parser.add_argument('--save_dir', type=str, default='checkpoints',
                       help='Directory to save checkpoints')
    
    parser.add_argument('--data_dir', type=str, default='data',
                       help='Directory containing data files')
    
    args = parser.parse_args()
    
    logger.info("="*80)
    logger.info("RNAChat Baseline Training Pipeline")
    logger.info("="*80)
    logger.info(f"Mode: {args.mode}")
    logger.info(f"Model Type: {args.model_type}")
    logger.info(f"Device: {'cuda' if torch.cuda.is_available() else 'cpu'}")
    logger.info("="*80)
    
    if args.mode in ['train', 'both']:
        train_all_baselines(args)
    
    if args.mode in ['eval', 'both']:
        results, comparisons = evaluate_all_baselines(args)
        
        logger.info("\n" + "="*80)
        logger.info("FINAL RESULTS")
        logger.info("="*80)
        logger.info("\nSee comprehensive_results.json for detailed metrics")
        logger.info("See statistical_comparisons.json for significance tests")


if __name__ == '__main__':
    main()
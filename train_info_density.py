import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer, 
    AutoModel,
    get_linear_schedule_with_warmup
)
import pandas as pd
import numpy as np
from tqdm import tqdm
import json
import re
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

class WikiSQEDataset(Dataset):
    """Dataset class for WikiSQE with information density features"""
    
    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        
        # Tokenize
        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        # Calculate simple information density features
        word_count = len(text.split())
        unique_words = len(set(text.lower().split()))
        lexical_diversity = unique_words / max(word_count, 1)
        
        # Count specific entities (simple heuristic: capitalized words, numbers)
        specific_terms = len(re.findall(r'\b[A-Z][a-z]+\b|\b\d+\b', text))
        specificity = specific_terms / max(word_count, 1)
        
        # Sentence length
        sentence_length = word_count
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'lexical_diversity': torch.tensor(lexical_diversity, dtype=torch.float),
            'specificity': torch.tensor(specificity, dtype=torch.float),
            'sentence_length': torch.tensor(sentence_length, dtype=torch.float),
            'label': torch.tensor(label, dtype=torch.long)
        }


class InfoDensityClassifier(nn.Module):
    """
    Simple model that combines:
    1. BERT embeddings for semantic understanding
    2. Hand-crafted information density features
    """
    
    def __init__(self, model_name='bert-base-uncased', num_classes=2, dropout=0.3):
        super(InfoDensityClassifier, self).__init__()
        
        # Load pre-trained BERT
        self.bert = AutoModel.from_pretrained(model_name)
        
        # Freeze early layers (optional - comment out for full fine-tuning)
        for param in list(self.bert.parameters())[:-24]:  # Freeze all but last 2 layers
            param.requires_grad = False
        
        bert_hidden_size = self.bert.config.hidden_size  # 768 for base
        
        # Additional feature processing
        self.feature_layer = nn.Sequential(
            nn.Linear(3, 32),  # 3 handcrafted features
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Combined classifier
        self.classifier = nn.Sequential(
            nn.Linear(bert_hidden_size + 32, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes)
        )
        
    def forward(self, input_ids, attention_mask, lexical_diversity, 
                specificity, sentence_length):
        
        # BERT encoding
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output  # [CLS] token representation
        
        # Process handcrafted features
        features = torch.stack([
            lexical_diversity, 
            specificity, 
            sentence_length / 100.0  # Normalize
        ], dim=1)
        
        feature_output = self.feature_layer(features)
        
        # Combine both representations
        combined = torch.cat([pooled_output, feature_output], dim=1)
        
        # Classification
        logits = self.classifier(combined)
        
        return logits


def load_wikisqe_data(sample_size=50000):
    """
    Load WikiSQE dataset from Hugging Face - inspect and adapt
    """
    print(f"Loading WikiSQE dataset from Hugging Face...")
    
    from datasets import load_dataset
    
    # Define workslop and substantive configs
    WORKSLOP_CONFIGS = [
        'vague', 'dubious', 'peacock term', 'weasel words', 'buzzword', 
        'jargon', 'colloquialism', 'editorializing', 'promotion', 
        'marketing material', 'opinion', 'a fact or an opinion', 
        'unbalanced opinion', 'tone', 'ambiguous', 'incomprehensible',
        'neologism', 'non sequitur', 'speculation', 'over-explained', 
        'repetition'
    ]
    
    SUBSTANTIVE_CONFIGS = [
        'citation needed', 'who', 'whom', 'by whom', 'from whom', 
        'to whom', 'with whom', 'like whom', 'when', 'as of', 
        'until when', 'timeframe', 'where', 'which', 'whose', 
        'why', 'how', 'how often', 'by how much', 'according to whom',
        'says who', 'who said this', 'quantify', 'specify',
        'example needed', 'clarification needed', 'context needed',
        'better source needed', 'additional citation needed',
        'verification needed', 'verify', 'failed verification',
        'chronology citation needed', 'medical citation needed',
        'scientific citation needed'
    ]
    
    all_data = []
    
    # Load WORKSLOP examples
    print("\n" + "="*50)
    print("Loading WORKSLOP examples...")
    print("="*50)
    
    for config in WORKSLOP_CONFIGS:
        try:
            ds = load_dataset("ando55/WikiSQE", config, split="train")
            print(f"\n'{config}': {len(ds)} samples")
            
            count = 0
            for item in ds:
                text = None
                # Try different possible field names
                for field in ['sentence', 'text', 'content', 'input', 'source']:
                    if field in item:
                        text = item[field]
                        break
                
                if text and isinstance(text, str) and len(text.split()) >= 3:
                    all_data.append({'text': text, 'label': 1})  # 1 = workslop
                    count += 1
            
            print(f"  Extracted: {count} valid sentences")
        
        except Exception as e:
            print(f"  ERROR loading '{config}': {e}")
    
    workslop_count = len(all_data)
    print(f"\nTotal workslop extracted: {workslop_count}")
    
    # Load SUBSTANTIVE examples
    print("\n" + "="*50)
    print("Loading SUBSTANTIVE examples...")
    print("="*50)
    
    for config in SUBSTANTIVE_CONFIGS:
        try:
            ds = load_dataset("ando55/WikiSQE", config, split="train")
            print(f"\n'{config}': {len(ds)} samples")
            
            count = 0
            for item in ds:
                text = None
                for field in ['sentence', 'text', 'content', 'input', 'source']:
                    if field in item:
                        text = item[field]
                        break
                
                if text and isinstance(text, str) and len(text.split()) >= 3:
                    all_data.append({'text': text, 'label': 0})  # 0 = substantive
                    count += 1
            
            print(f"  Extracted: {count} valid sentences")
        
        except Exception as e:
            print(f"  ERROR loading '{config}': {e}")
    
    substantive_count = len(all_data) - workslop_count
    print(f"\nTotal substantive extracted: {substantive_count}")
    
    if len(all_data) == 0:
        print("\n" + "="*50)
        print("DEBUGGING: No data extracted!")
        print("="*50)
        print("Let's look at the actual structure...")
        ds = load_dataset("ando55/WikiSQE", 'vague', split="train")
        print(f"\nDataset type: {type(ds)}")
        print(f"Dataset[0] type: {type(ds[0])}")
        print(f"Dataset[0] content: {ds[0]}")
        print(f"Dataset[0] keys: {ds[0].keys() if hasattr(ds[0], 'keys') else 'No keys method'}")
        raise ValueError("Could not extract any data - see debug info above")
    
    # Balance the dataset to handle class imbalance
    print("\n" + "="*50)
    print("Balancing dataset...")
    print("="*50)
    
    workslop_data = [d for d in all_data if d['label'] == 1]
    substantive_data = [d for d in all_data if d['label'] == 0]
    
    print(f"Before balancing:")
    print(f"  Workslop: {len(workslop_data)}")
    print(f"  Substantive: {len(substantive_data)}")
    
    # Downsample majority class to match minority class
    min_count = min(len(workslop_data), len(substantive_data))
    
    import random
    random.seed(42)
    workslop_data = random.sample(workslop_data, min_count)
    substantive_data = random.sample(substantive_data, min_count)
    
    all_data = workslop_data + substantive_data
    
    # Shuffle
    random.shuffle(all_data)
    
    print(f"After balancing:")
    print(f"  Workslop: {len(workslop_data)}")
    print(f"  Substantive: {len(substantive_data)}")
    print(f"  Total: {len(all_data)}")
    
    # Sample if needed
    if len(all_data) > sample_size:
        all_data = random.sample(all_data, sample_size)
        print(f"\nSampled down to: {sample_size}")
    
    texts = [d['text'] for d in all_data]
    labels = [d['label'] for d in all_data]
    
    print("\n" + "="*50)
    print("FINAL DATASET")
    print("="*50)
    print(f"Total sentences: {len(texts)}")
    print(f"Workslop: {sum(labels)} ({sum(labels)/len(labels)*100:.1f}%)")
    print(f"Substantive: {len(labels) - sum(labels)} ({(len(labels)-sum(labels))/len(labels)*100:.1f}%)")
    
    # Show examples
    print("\n" + "="*50)
    print("Sample Examples:")
    print("="*50)
    
    print("\nWorkslop examples:")
    workslop_examples = [texts[i] for i in range(len(texts)) if labels[i] == 1][:3]
    for i, ex in enumerate(workslop_examples, 1):
        print(f"{i}. {ex[:150]}...")
    
    print("\nSubstantive examples:")
    substantive_examples = [texts[i] for i in range(len(texts)) if labels[i] == 0][:3]
    for i, ex in enumerate(substantive_examples, 1):
        print(f"{i}. {ex[:150]}...")
    
    return texts, labels

def train_epoch(model, dataloader, optimizer, scheduler, device):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    predictions = []
    true_labels = []
    
    progress_bar = tqdm(dataloader, desc="Training")
    
    for batch in progress_bar:
        # Move to device
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        lexical_diversity = batch['lexical_diversity'].to(device)
        specificity = batch['specificity'].to(device)
        sentence_length = batch['sentence_length'].to(device)
        labels = batch['label'].to(device)
        
        # Forward pass
        optimizer.zero_grad()
        logits = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            lexical_diversity=lexical_diversity,
            specificity=specificity,
            sentence_length=sentence_length
        )
        
        # Compute loss
        loss = nn.CrossEntropyLoss()(logits, labels)
        
        # Backward pass
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        
        # Track metrics
        total_loss += loss.item()
        preds = torch.argmax(logits, dim=1).cpu().numpy()
        predictions.extend(preds)
        true_labels.extend(labels.cpu().numpy())
        
        # Update progress bar
        progress_bar.set_postfix({'loss': loss.item()})
    
    avg_loss = total_loss / len(dataloader)
    accuracy = accuracy_score(true_labels, predictions)
    f1 = f1_score(true_labels, predictions, average='binary')
    
    return avg_loss, accuracy, f1


def evaluate(model, dataloader, device):
    """Evaluate the model"""
    model.eval()
    total_loss = 0
    predictions = []
    true_labels = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            # Move to device
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            lexical_diversity = batch['lexical_diversity'].to(device)
            specificity = batch['specificity'].to(device)
            sentence_length = batch['sentence_length'].to(device)
            labels = batch['label'].to(device)
            
            # Forward pass
            logits = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                lexical_diversity=lexical_diversity,
                specificity=specificity,
                sentence_length=sentence_length
            )
            
            # Compute loss
            loss = nn.CrossEntropyLoss()(logits, labels)
            total_loss += loss.item()
            
            # Track predictions
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            predictions.extend(preds)
            true_labels.extend(labels.cpu().numpy())
    
    avg_loss = total_loss / len(dataloader)
    accuracy = accuracy_score(true_labels, predictions)
    f1 = f1_score(true_labels, predictions, average='binary')
    
    return avg_loss, accuracy, f1, predictions, true_labels


def plot_training_curves(train_losses, val_losses, train_accs, val_accs):
    """Plot training curves"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Loss curves
    ax1.plot(train_losses, label='Train Loss', marker='o')
    ax1.plot(val_losses, label='Val Loss', marker='s')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.legend()
    ax1.grid(True)
    
    # Accuracy curves
    ax2.plot(train_accs, label='Train Accuracy', marker='o')
    ax2.plot(val_accs, label='Val Accuracy', marker='s')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.set_title('Training and Validation Accuracy')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig('results/training_curves.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Saved training curves to results/training_curves.png")


def main():
    # Hyperparameters
    CONFIG = {
        'model_name': 'bert-base-uncased',
        'max_length': 128,
        'batch_size': 32,
        'learning_rate': 2e-5,
        'num_epochs': 3,
        'sample_size': 50000,  # Use subset for faster training
        'test_size': 0.2,
        'val_size': 0.1
    }
    
    print("="*50)
    print("Workslop Detection via Information Density")
    print("="*50)
    print(f"\nConfiguration:")
    for key, value in CONFIG.items():
        print(f"  {key}: {value}")
    print()
    
    # Load data
    texts, labels = load_wikisqe_data(CONFIG['sample_size'])
    
    # Split data
    X_temp, X_test, y_temp, y_test = train_test_split(
        texts, labels, test_size=CONFIG['test_size'], random_state=42, stratify=labels
    )
    
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=CONFIG['val_size']/(1-CONFIG['test_size']), 
        random_state=42, stratify=y_temp
    )
    
    print(f"\nDataset splits:")
    print(f"  Train: {len(X_train)} samples")
    print(f"  Val: {len(X_val)} samples")
    print(f"  Test: {len(X_test)} samples")
    
    # Initialize tokenizer
    print(f"\nLoading tokenizer: {CONFIG['model_name']}")
    tokenizer = AutoTokenizer.from_pretrained(CONFIG['model_name'])
    
    # Create datasets
    train_dataset = WikiSQEDataset(X_train, y_train, tokenizer, CONFIG['max_length'])
    val_dataset = WikiSQEDataset(X_val, y_val, tokenizer, CONFIG['max_length'])
    test_dataset = WikiSQEDataset(X_test, y_test, tokenizer, CONFIG['max_length'])
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=CONFIG['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=CONFIG['batch_size'])
    test_loader = DataLoader(test_dataset, batch_size=CONFIG['batch_size'])
    
    # Initialize model
    print(f"\nInitializing model: {CONFIG['model_name']}")
    model = InfoDensityClassifier(model_name=CONFIG['model_name'], num_classes=2)
    model = model.to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Setup optimizer and scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=CONFIG['learning_rate'])
    
    total_steps = len(train_loader) * CONFIG['num_epochs']
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=total_steps // 10,
        num_training_steps=total_steps
    )
    
    # Training loop
    print("\n" + "="*50)
    print("Starting Training")
    print("="*50)
    
    train_losses, val_losses = [], []
    train_accs, val_accs = [], []
    best_val_f1 = 0
    
    for epoch in range(CONFIG['num_epochs']):
        print(f"\nEpoch {epoch + 1}/{CONFIG['num_epochs']}")
        print("-" * 50)
        
        # Train
        train_loss, train_acc, train_f1 = train_epoch(
            model, train_loader, optimizer, scheduler, device
        )
        
        # Validate
        val_loss, val_acc, val_f1, _, _ = evaluate(model, val_loader, device)
        
        # Store metrics
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)
        
        print(f"\nResults:")
        print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | Train F1: {train_f1:.4f}")
        print(f"  Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f} | Val F1: {val_f1:.4f}")
        
        # Save best model
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            torch.save(model.state_dict(), 'models/best_model.pt')
            print(f"  ✓ New best model saved! (F1: {best_val_f1:.4f})")
    
    # Plot training curves
    plot_training_curves(train_losses, val_losses, train_accs, val_accs)
    
    # Final evaluation on test set
    print("\n" + "="*50)
    print("Final Evaluation on Test Set")
    print("="*50)
    
    # Load best model
    model.load_state_dict(torch.load('models/best_model.pt'))
    
    test_loss, test_acc, test_f1, predictions, true_labels = evaluate(
        model, test_loader, device
    )
    
    print(f"\nTest Results:")
    print(f"  Loss: {test_loss:.4f}")
    print(f"  Accuracy: {test_acc:.4f}")
    print(f"  F1 Score: {test_f1:.4f}")
    
    print("\nClassification Report:")
    print(classification_report(
        true_labels, predictions,
        target_names=['Substantive', 'Workslop']
    ))
    
    # Save results
    results = {
        'config': CONFIG,
        'test_accuracy': test_acc,
        'test_f1': test_f1,
        'train_losses': train_losses,
        'val_losses': val_losses,
        'train_accs': train_accs,
        'val_accs': val_accs
    }
    
    with open('results/results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print("\n" + "="*50)
    print("Training Complete!")
    print("="*50)
    print(f"Best model saved to: models/best_model.pt")
    print(f"Results saved to: results/results.json")
    print(f"Training curves saved to: results/training_curves.png")


if __name__ == "__main__":
    main()

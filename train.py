import os
import ast
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import RobertaTokenizer, RobertaForSequenceClassification
from torch.optim import AdamW
from sklearn.metrics import f1_score
from sklearn.utils.class_weight import compute_class_weight
from tqdm import tqdm

# ==========================================
# 1. Configuration & Hyperparameters
# ==========================================
MAX_LEN = 128
BATCH_SIZE = 16
EPOCHS = 4
LEARNING_RATE = 2e-5
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_NAME = 'roberta-base'

# ==========================================
# 2. Data Loading & Preprocessing
# ==========================================
def prepare_data():
    print("Loading and merging data...")

    train_labels = pd.read_csv('data/train_semeval_parids-labels.csv')
    dev_labels = pd.read_csv('data/dev_semeval_parids-labels.csv')
    
    # Parse strings like "[1, 0, 0...]" into actual Python lists
    train_labels['label_list'] = train_labels['label'].apply(ast.literal_eval)
    dev_labels['label_list'] = dev_labels['label'].apply(ast.literal_eval)


    train_labels['task1_label'] = train_labels['label_list'].apply(lambda x: 1 if sum(x) > 0 else 0)
    dev_labels['task1_label'] = dev_labels['label_list'].apply(lambda x: 1 if sum(x) > 0 else 0)

    texts_df = pd.read_csv(os.path.join('data', 'dontpatronizeme_pcl.tsv'), sep='\t', skiprows=4, 
                           names=['par_id', 'art_id', 'keyword', 'country', 'text', 'orig_label'])
    
    texts_df['par_id'] = texts_df['par_id'].astype(str)
    train_labels['par_id'] = train_labels['par_id'].astype(str)
    dev_labels['par_id'] = dev_labels['par_id'].astype(str)

    train_df = pd.merge(train_labels, texts_df[['par_id', 'text']], on='par_id', how='left').dropna(subset=['text'])
    dev_df = pd.merge(dev_labels, texts_df[['par_id', 'text']], on='par_id', how='left').dropna(subset=['text'])

    return train_df.reset_index(drop=True), dev_df.reset_index(drop=True)

# ==========================================
# 3. PyTorch Dataset Class
# ==========================================
class PCLDataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_len):
        self.tokenizer = tokenizer
        self.text = dataframe['text']
        self.targets = dataframe['task1_label']
        self.max_len = max_len

    def __len__(self):
        return len(self.text)

    def __getitem__(self, index):
        text = str(self.text[index])
        text = " ".join(text.split()) 

        inputs = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )

        return {
            'input_ids': inputs['input_ids'].flatten(),
            'attention_mask': inputs['attention_mask'].flatten(),
            'targets': torch.tensor(self.targets[index], dtype=torch.long)
        }

# ==========================================
# 4. Training and Evaluation Loops
# ==========================================
def train():

    train_df, dev_df = prepare_data()
    
    tokenizer = RobertaTokenizer.from_pretrained(MODEL_NAME)
    
    train_dataset = PCLDataset(train_df, tokenizer, MAX_LEN)
    dev_dataset = PCLDataset(dev_df, tokenizer, MAX_LEN)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    dev_loader = DataLoader(dev_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    model = RobertaForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)
    model = model.to(DEVICE)
    
    class_weights = compute_class_weight(
        class_weight='balanced',
        classes=np.unique(train_df['task1_label']),
        y=train_df['task1_label'].to_numpy()
    )
    weights = torch.tensor(class_weights, dtype=torch.float).to(DEVICE)
    
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss(weight=weights)

    best_f1 = 0
    for epoch in range(EPOCHS):
        print(f"\n======== Epoch {epoch+1} / {EPOCHS} ========")
        model.train()
        total_train_loss = 0
        
        for batch in tqdm(train_loader, desc="Training"):
            input_ids = batch['input_ids'].to(DEVICE)
            attention_mask = batch['attention_mask'].to(DEVICE)
            targets = batch['targets'].to(DEVICE)
            
            optimizer.zero_grad()
            
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            
            loss = criterion(logits, targets)
            total_train_loss += loss.item()
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
        avg_train_loss = total_train_loss / len(train_loader)
        print(f"Average Training Loss: {avg_train_loss:.4f}")
        
        model.eval()
        predictions, true_labels = [], []
        
        with torch.no_grad():
            for batch in tqdm(dev_loader, desc="Evaluating"):
                input_ids = batch['input_ids'].to(DEVICE)
                attention_mask = batch['attention_mask'].to(DEVICE)
                targets = batch['targets'].to(DEVICE)
                
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                logits = outputs.logits
                
                preds = torch.argmax(logits, dim=1).cpu().numpy()
                labels = targets.cpu().numpy()
                
                predictions.extend(preds)
                true_labels.extend(labels)
        
        val_f1 = f1_score(true_labels, predictions, pos_label=1)
        print(f"Validation PCL F1-Score: {val_f1:.4f}")
        
        # Save Best Model
        if val_f1 > best_f1:
            best_f1 = val_f1
            print("New best model found! Saving to 'BestModel/'...")
            if not os.path.exists("BestModel"):
                os.makedirs("BestModel")
            model.save_pretrained("BestModel/")
            tokenizer.save_pretrained("BestModel/")

if __name__ == "__main__":
    train()
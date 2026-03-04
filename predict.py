import os
import pandas as pd
import torch
from transformers import RobertaTokenizer, RobertaForSequenceClassification
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

# ==========================================
# 1. Configuration
# ==========================================
MAX_LEN = 128
BATCH_SIZE = 32  
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = "BestModel/"

# ==========================================
# 2. PyTorch Inference Dataset
# ==========================================
class PCLInferenceDataset(Dataset):
    def __init__(self, texts, tokenizer, max_len):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, index):
        text = str(self.texts[index])
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
            'attention_mask': inputs['attention_mask'].flatten()
        }

# ==========================================
# 3. Prediction Function
# ==========================================
def generate_predictions(texts, model, tokenizer):
    dataset = PCLInferenceDataset(texts, tokenizer, MAX_LEN)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    predictions = []
    
    with torch.no_grad():
        for batch in tqdm(loader, desc="Predicting"):
            input_ids = batch['input_ids'].to(DEVICE)
            attention_mask = batch['attention_mask'].to(DEVICE)
            
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            predictions.extend(preds)
            
    return predictions

# ==========================================
# 4. Main Execution
# ==========================================
def main():
    print("Loading fine-tuned model and tokenizer from 'BestModel/'...")
    tokenizer = RobertaTokenizer.from_pretrained(MODEL_PATH)
    model = RobertaForSequenceClassification.from_pretrained(MODEL_PATH)
    model.to(DEVICE)
    model.eval()

    print("Loading Official Dev Set...")
    dev_labels = pd.read_csv('data/dev_semeval_parids-labels.csv')
    dev_labels['par_id'] = dev_labels['par_id'].astype(str)
    
    texts_df = pd.read_csv(os.path.join('data', 'dontpatronizeme_pcl.tsv'), sep='\t', skiprows=4, 
                           names=['par_id', 'art_id', 'keyword', 'country', 'text', 'orig_label'])
    texts_df['par_id'] = texts_df['par_id'].astype(str)
    
    dev_df = pd.merge(dev_labels, texts_df[['par_id', 'text']], on='par_id', how='left')
    dev_texts = dev_df['text'].fillna("").tolist() 

    print("Loading Official Test Set...")
    test_df = pd.read_csv(os.path.join('data', 'task4_test.tsv'), sep='\t', names=['par_id', 'art_id', 'keyword', 'country', 'text'])
    test_texts = test_df['text'].fillna("").tolist()
    
    if len(test_texts) != 3832:
        print(f"Warning: Expected 3832 test samples, but found {len(test_texts)}.")

    dev_preds = generate_predictions(dev_texts, model, tokenizer)
    test_preds = generate_predictions(test_texts, model, tokenizer)

    print("Writing predictions to dev.txt...")
    with open('dev.txt', 'w') as f:
        for p in dev_preds:
            f.write(f"{p}\n")
            
    print("Writing predictions to test.txt...")
    with open('test.txt', 'w') as f:
        for p in test_preds:
            f.write(f"{p}\n")

if __name__ == "__main__":
    main()
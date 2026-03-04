import pandas as pd
import numpy as np
import ast
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, PrecisionRecallDisplay

def perform_error_analysis():
    print("Loading datasets...")
    dev_labels = pd.read_csv('data/dev_semeval_parids-labels.csv')
    dev_labels['label_list'] = dev_labels['label'].apply(ast.literal_eval)
    dev_labels['y_true'] = dev_labels['label_list'].apply(lambda x: 1 if sum(x) > 0 else 0)
    
    texts_df = pd.read_csv('data/dontpatronizeme_pcl.tsv', sep='\t', skiprows=4, 
                           names=['par_id', 'art_id', 'keyword', 'country', 'text', 'orig_label'])
    dev_labels['par_id'] = dev_labels['par_id'].astype(str)
    texts_df['par_id'] = texts_df['par_id'].astype(str)
    
    df = pd.merge(dev_labels, texts_df[['par_id', 'text', 'keyword']], on='par_id', how='left')
    
    print("Loading model predictions...")
    try:
        with open('dev.txt', 'r') as f:
            preds = [int(line.strip()) for line in f.readlines()]
        df['y_pred'] = preds
    except FileNotFoundError:
        print("Error: 'dev.txt' not found. Please run predict.py first.")
        return
        
    print("Generating Confusion Matrix...")
    cm = confusion_matrix(df['y_true'], df['y_pred'])
    plt.figure(figsize=(6,5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Non-PCL (0)', 'PCL (1)'], 
                yticklabels=['Non-PCL (0)', 'PCL (1)'])
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix: Best Model vs. Dev Set')
    plt.tight_layout()
    plt.savefig('eval_confusion_matrix.png', dpi=300)
    plt.close()
    
    print("\n--- Classification Report ---")
    print(classification_report(df['y_true'], df['y_pred']))
    
    print("Generating Precision-Recall Curve...")
    PrecisionRecallDisplay.from_predictions(df['y_true'], df['y_pred'], name="Class-Weighted RoBERTa")
    plt.title('Precision-Recall Curve')
    plt.savefig('eval_pr_curve.png', dpi=300)
    plt.close()

    print("Extracting failure cases for manual inspection...")
    false_positives = df[(df['y_true'] == 0) & (df['y_pred'] == 1)]
    false_negatives = df[(df['y_true'] == 1) & (df['y_pred'] == 0)]
    
    print(f"\nTotal False Positives (Predicted PCL, actually Non-PCL): {len(false_positives)}")
    print(f"Total False Negatives (Predicted Non-PCL, actually PCL): {len(false_negatives)}")
    
    false_positives.to_csv('error_false_positives.csv', index=False)
    false_negatives.to_csv('error_false_negatives.csv', index=False)
    
    print("\nTop 5 Keywords triggering False Positives:")
    print(false_positives['keyword'].value_counts().head(5))
    
    print("\nTop 5 Keywords triggering False Negatives:")
    print(false_negatives['keyword'].value_counts().head(5))

if __name__ == "__main__":
    perform_error_analysis()
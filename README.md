# Don't Patronize Me! - PCL Detection (SemEval 2022 Task 4)

This repository contains the source code and final model for Subtask 1 of the SemEval 2022 Task 4: "Don't Patronize Me!". The goal of this project is to identify Patronizing and Condescending Language (PCL) directed at vulnerable communities within news media text.

## 🚀 Proposed Approach
The dataset exhibits a severe 90/10 class imbalance. To surpass the baseline `RoBERTa-base` performance, this project implements a **Cost-Sensitive Learning strategy**. By applying a dynamically calculated Class-Weighted Cross-Entropy loss, the model heavily penalizes false negatives, forcing the network to prioritize the hard-to-classify PCL minority class and maximizing the positive class F1-score.

## 📁 Repository Structure

* `BestModel/`: Contains the fine-tuned RoBERTa-base model weights and tokenizer (required for evaluation).
* `train.py`: The main training pipeline. Handles data splitting, tokenization (max_length=128), and cost-sensitive fine-tuning.
* `evaluate.py`: Generates local evaluation metrics (Confusion Matrix, Precision-Recall curves) and extracts specific False Positives/False Negatives for manual error analysis.
* `predict.py`: Runs inference on the official Dev and Test sets, formatting the outputs into `dev.txt` and `test.txt` for submission.
* `dont_patronize_me.py`: Official data loading helper script.
* `data/`: Directory containing the label CSVs (`train_semeval_parids-labels.csv`, `dev_semeval_parids-labels.csv`).

## ⚙️ Environment Setup
This pipeline was developed and tested on an RTX 3080 (8GB VRAM) using CUDA 12.5. 

To recreate the environment, install the following dependencies:
```bash
conda create -n pcl_env python=3.10 -y
conda activate pcl_env
pip install pandas matplotlib seaborn scikit-learn tqdm
pip install torch torchvision torchaudio --index-url [https://download.pytorch.org/whl/cu124](https://download.pytorch.org/whl/cu124)
pip install transformers datasets accelerate

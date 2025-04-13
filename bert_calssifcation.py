import pandas as pd
from sklearn.metrics import classification_report, precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from transformers import DataCollatorWithPadding
from datasets import Dataset, ClassLabel
import torch

# # === Load train and test data ===
# train_df = pd.read_csv("C:/Users/kaust/OneDrive/Documents/GCN Project/Text-Level-GNN-main/Text-Level-GNN-main/train_trec_coarse_data.csv")
# test_df = pd.read_csv("C:/Users/kaust/OneDrive/Documents/GCN Project/Text-Level-GNN-main/Text-Level-GNN-main/test_trec_coarse_data.csv")

# # Rename columns
# train_df.rename(columns={"||__text": "text", "||__coarse_label": "label"}, inplace=True)
# test_df.rename(columns={"||__text": "text", "||__coarse_label": "label"}, inplace=True)

# # Sanity check (optional)
# print("Train labels:", train_df["label"].unique())
# print("Test labels:", test_df["label"].unique())

# # Ensure labels are integers
# train_df["label"] = train_df["label"].astype(int)
# test_df["label"] = test_df["label"].astype(int)


# === Load the combined dataset ===
df = pd.read_csv("C:/Users/kaust/Downloads/bloom_split.csv")

# Rename columns
df.rename(columns={"||__text": "text", "||__label": "label"}, inplace=True)

# Ensure labels are integers
df["label"] = df["label"].astype(int)

# === Split into train and test ===
train_df, test_df = train_test_split(df, test_size=0.2, stratify=df["label"], random_state=42)

# Sanity check (optional)
print("Train labels:", train_df["label"].unique())
print("Test labels:", test_df["label"].unique())


# Convert to Hugging Face Dataset
train_dataset = Dataset.from_pandas(train_df)
test_dataset = Dataset.from_pandas(test_df)

# Cast label to ClassLabel type
class_label = ClassLabel(num_classes=6, names=["0", "1", "2","3", "4", "5"])
train_dataset = train_dataset.cast_column("label", class_label)
test_dataset = test_dataset.cast_column("label", class_label)

# Tokenization
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

def tokenize_function(example):
    return tokenizer(example["text"], truncation=True)

train_dataset = train_dataset.map(tokenize_function, batched=True)
test_dataset = test_dataset.map(tokenize_function, batched=True)

# Define model with correct number of labels
model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=6)

# Training arguments
training_args = TrainingArguments(
    output_dir="./results",
    eval_strategy="epoch",  # fixed typo
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    weight_decay=0.01,
    load_best_model_at_end=True,
    logging_dir="./logs",
    metric_for_best_model="f1",
    logging_steps=10
)

# Evaluation Metrics
def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='macro', zero_division=0)
    return {
        'precision': precision,
        'recall': recall,
        'f1': f1
    }

# Trainer Setup
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics
)

# Train
trainer.train()

# Final Evaluation
results = trainer.evaluate()
print("Final Evaluation Metrics:", results)

# Classification Report
predictions = trainer.predict(test_dataset)
print(classification_report(predictions.label_ids, predictions.predictions.argmax(-1)))

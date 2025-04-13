import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, precision_recall_fscore_support, ConfusionMatrixDisplay
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding
)
from datasets import Dataset, ClassLabel
import matplotlib.pyplot as plt

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



# Use DeBERTa tokenizer and model
model_name = "microsoft/deberta-v3-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=6)

# Tokenization function
def tokenize_function(example):
    return tokenizer(example["text"], truncation=True)

# Tokenize datasets
train_dataset = train_dataset.map(tokenize_function, batched=True)
test_dataset = test_dataset.map(tokenize_function, batched=True)

# Training arguments
training_args = TrainingArguments(
    output_dir="./results",
    eval_strategy="epoch",
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

# Evaluation metric function
def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='macro', zero_division=0)
    return {
        'precision': precision,
        'recall': recall,
        'f1': f1
    }

# Data collator
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics
)

# === Train the model ===
trainer.train()

# === Final evaluation ===
results = trainer.evaluate()
print("Final Evaluation Metrics:", results)

# === Predictions + report ===
predictions = trainer.predict(test_dataset)
y_true = predictions.label_ids
y_pred = predictions.predictions.argmax(-1)

print("\nClassification Report:\n")
print(classification_report(y_true, y_pred))

# === Save predictions to CSV ===
output_df = test_df.copy()
output_df["predicted_label"] = y_pred
output_df.to_csv("deberta_predictions_trec_coarse.csv", index=False)
print("\nPredictions saved to 'deberta_predictions.csv'.")

# === Optional: Confusion matrix ===
ConfusionMatrixDisplay.from_predictions(y_true, y_pred, display_labels=["0", "1", "2","3", "4", "5"])
plt.title("DeBERTa Confusion Matrix")
plt.tight_layout()
plt.show()

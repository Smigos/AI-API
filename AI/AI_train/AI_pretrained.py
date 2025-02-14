from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
import pandas as pd
import torch
import os

# Define the storage path
BASE_PATH = "D:\AI stuff\Important"
MODEL_PATH = os.path.join(BASE_PATH, "fine_tuned_gpt2")
DATA_PATH = os.path.join(BASE_PATH, "training_data.csv")

# Check if training data exists
if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(f"Training data file not found at: {DATA_PATH}. Please ensure it exists.")

# Load the tokenizer and model
model_name = "gpt2"
device = "cuda" if torch.cuda.is_available() else "cpu"

tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(model_name)
model.resize_token_embeddings(len(tokenizer))
model.to(device)

# Load the preprocessed training data
combined_data = pd.read_csv(DATA_PATH)

# Shuffle and Split Dataset
train_size = int(0.8 * len(combined_data))
train_data, eval_data = torch.utils.data.random_split(
    combined_data.to_dict(orient="records"),
    [train_size, len(combined_data) - train_size],
    generator=torch.Generator().manual_seed(42),
)

# Dataset Class for PyTorch
class ConversationDataset(torch.utils.data.Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        tokens = tokenizer(
            f"User: {item['prompt']} AI: {item['response']}",
            truncation=True,
            padding="max_length",
            max_length=128,
            return_tensors="pt",
        )
        tokens["labels"] = tokens["input_ids"].clone()
        return {key: val.squeeze(0) for key, val in tokens.items()}

# Create PyTorch Datasets
train_dataset = ConversationDataset(train_data)
eval_dataset = ConversationDataset(eval_data)

# Training Arguments
training_args = TrainingArguments(
    output_dir=MODEL_PATH,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    learning_rate=5e-5,
    num_train_epochs=3,
    logging_dir=os.path.join(MODEL_PATH, "logs"),
    save_total_limit=2,
)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    tokenizer=tokenizer,
)

# Train the Model
print("Training started...")
trainer.train()
print("Training complete!")

# Save the Fine-Tuned Model
model.save_pretrained(MODEL_PATH)
tokenizer.save_pretrained(MODEL_PATH)
print(f"Fine-tuned model saved to: {MODEL_PATH}")

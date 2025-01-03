import os
import torch
import sys
from torch.utils.data import DataLoader
from transformers import BertTokenizer
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from src.dataset import CustomDataset
from src.model import create_model
import pandas as pd

def split_data(data_path, train_val_ratio=0.7, val_ratio=0.2):
    # Load the dataset
    df = pd.read_csv(data_path)

    # Split 70% for train+validation, 30% for prediction testing
    train_val_df, test_df = train_test_split(df, test_size=1 - train_val_ratio, random_state=42)

    # Further split train_val into training and validation (e.g., 80% train, 20% validation)
    train_df, val_df = train_test_split(train_val_df, test_size=val_ratio, random_state=42)

    # Save the splits into CSV files
    train_df.to_csv("assets/training_data/train.csv", index=False)
    val_df.to_csv("assets/training_data/val.csv", index=False)
    test_df.to_csv("assets/training_data/test.csv", index=False)

    print("Data split complete:")
    print(f"Training size: {len(train_df)} rows")
    print(f"Validation size: {len(val_df)} rows")
    print(f"Test size: {len(test_df)} rows\n")

def train_model(data_path, output_dir="model", epochs=10, batch_size=32, lr=5e-5, max_length=128):
    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Load the data
    df = pd.read_csv(data_path)
    df = df.dropna(subset=['text', 'label'])
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        df['text'].tolist(), df['label'].tolist(), test_size=0.2, random_state=42
    )

    # Load tokenizer
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    tokenizer.save_pretrained(os.path.join(output_dir, "tokenizer"))

    # Prepare datasets and dataloaders
    train_dataset = CustomDataset(train_texts, train_labels, tokenizer, max_length)
    val_dataset = CustomDataset(val_texts, val_labels, tokenizer, max_length)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # Initialize model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = create_model()
    model.to(device)

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    # Training loop
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        

        progress_bar = tqdm(train_loader, desc=f"Training Epoch {epoch + 1}", ncols=100)

        for batch in progress_bar:
            optimizer.zero_grad()
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            progress_bar.set_postfix(loss=loss.item())
            

        print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss / len(train_loader):.4f}\n")

    # Save the model weights and tokenizer
    model_save_path = os.path.join(output_dir, "bert_sentiment_model.pt")
    torch.save(model.state_dict(), model_save_path)
    print(f"Model saved to {model_save_path}")

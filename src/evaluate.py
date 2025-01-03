import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from src.dataset import CustomDataset
from src.model import create_model
from transformers import BertTokenizer
import pandas as pd
from sklearn.metrics import accuracy_score

def evaluate_model(data_path, model_path, batch_size=32, max_length=128):
    # Load Data
    df = pd.read_csv(data_path)
    texts = df['text'].tolist()
    labels = df['label'].tolist()

    # Tokenizer and Dataset
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    dataset = CustomDataset(texts, labels, tokenizer, max_length)
    loader = DataLoader(dataset, batch_size=batch_size)

    # Load Model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = create_model()
    model.load_state_dict(torch.load(model_path, weights_only=True))
    model.to(device)
    model.eval()

    # Evaluate
    predictions, true_labels = [], []
    with torch.no_grad():
        progress_bar = tqdm(loader, desc=f"Evaluating Labels", ncols=100)
        for batch in progress_bar:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            preds = torch.argmax(outputs.logits, dim=1)
            predictions.extend(preds.cpu().numpy())
            true_labels.extend(labels.cpu().numpy())

    accuracy = accuracy_score(true_labels, predictions)
    print(f"Validation Accuracy: {accuracy:.4f}")

import os
import torch
from transformers import BertTokenizer
from src.model import create_model



def predict_sentiment(text, model_dir="model", max_length=128):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load Tokenizer
    tokenizer_path = os.path.join(model_dir, "tokenizer")
    tokenizer = BertTokenizer.from_pretrained(tokenizer_path)

    # Load Model
    model_path = os.path.join(model_dir, "bert_sentiment_model.pt")
    model = create_model()
    model.load_state_dict(torch.load(model_path, weights_only=True, map_location=device))
    model.to(device)
    model.eval()

    # Tokenize Input
    inputs = tokenizer.encode_plus(
        text,
        max_length=max_length,
        truncation=True,
        padding="max_length",
        return_tensors="pt"
    )
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)

    # Make Prediction
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        prediction = torch.argmax(outputs.logits, dim=1).item()

    return "Positive" if prediction == 1 else "Negative"
import os
import torch
from transformers import BertTokenizer
from src.model import create_model

class SentimentPredictor:
    def __init__(self, model_dir="model", max_length=128):
        """
        Initializes the Sentiment Predictor by loading the model and tokenizer.
        
        Args:
            model_dir (str): Directory containing the model and tokenizer files.
            max_length (int): Maximum sequence length for tokenization.
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.max_length = max_length

        # Load Tokenizer
        tokenizer_path = os.path.join(model_dir, "tokenizer")
        self.tokenizer = BertTokenizer.from_pretrained(tokenizer_path)

        # Load Model
        model_path = os.path.join(model_dir, "bert_sentiment_model.pt")
        self.model = create_model()
        self.model.load_state_dict(torch.load(model_path, map_location=self.device, weights_only=True))
        self.model.to(self.device)
        self.model.eval()

    def predict(self, text: str) -> str:
        """
        Predicts the sentiment of the given text.

        Args:
            text (str): Input text for sentiment prediction.

        Returns:
            str: Predicted sentiment ("Positive" or "Negative").
        """
        # Tokenize the input text
        inputs = self.tokenizer.encode_plus(
            text,
            max_length=self.max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt"
        )
        input_ids = inputs["input_ids"].to(self.device)
        attention_mask = inputs["attention_mask"].to(self.device)

        # Make Prediction
        with torch.no_grad():
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            prediction = torch.argmax(outputs.logits, dim=1).item()
            

        return "Positive" if prediction == 1 else "Negative"
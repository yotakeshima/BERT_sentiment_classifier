import os
from src.train import train_model
from src.evaluate import evaluate_model
from src.predictor import SentimentPredictor
from src.train import split_data
from src.utils import check_model
from transformers import logging

logging.set_verbosity_error()

def main():
    data_path = "assets/reviews_for_train.csv"
    
    print("\nWould you like to train the model? Otherwise, a model will be loaded for predictions\n(if no model is found, training begins automatically)")
    initial_input = input("Y or N: ").strip().lower()

    # Check if training is required
    if initial_input == "y" or not check_model():
        print("\nStarting the training process...")
        if not os.path.exists("assets/training_data/train.csv"):
            print("\nSplitting the dataset...")
            split_data(data_path)
        train_model("assets/training_data/train.csv")

        print("\nEvaluating the model...")
        evaluate_model("assets/training_data/val.csv", "model/bert_sentiment_model.pt")
        evaluate_model("assets/training_data/test.csv", "model/bert_sentiment_model.pt")
        print("\nTraining and evaluation completed.")

    # Load predictor for sentiment analysis
    predictor = SentimentPredictor(model_dir="model")
    while True:
        print("\nEnter a sentence to predict sentiment (q = exit):")
        input_text = input("Sentence: ").strip()
        if input_text.lower() == 'q':
            print("Exiting the program. Goodbye!")
            break
        sentiment = predictor.predict(input_text)
        print(f"Prediction: The overall sentiment of this sentence is {sentiment}")

if __name__ == "__main__":
    main()

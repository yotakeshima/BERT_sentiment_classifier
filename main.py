import os
from src.train import train_model
from src.evaluate import evaluate_model
from src.predict import predict_sentiment
from src.train import split_data
from src.utils import check_model
from src.predictor import SentimentPredictor
from transformers import logging


logging.set_verbosity_error()



if __name__ == "__main__":
    data_path = "assets/reviews_for_train.csv"
    if not check_model():
         # Train the model
        print("\nStarting the training process...")
        if not os.path.exists("assets/training_data/train.csv"):
            print("\nSplitting the dataset...")
            split_data(data_path)
        train_model("assets/training_data/train.csv")

        # Evaluate the model
        print("\nEvaluating the model...")
        evaluate_model("assets/training_data/val.csv", "model/bert_sentiment_model.pt")
        # Test on the test set
        evaluate_model("assets/training_data/test.csv", "model/bert_sentiment_model.pt")
        print("\n Training and evualuation completed")
    else:
        predictor = SentimentPredictor(model_dir="model")
        while True:
            # Predict a User Input sentence sentiment
            print("\nEnter a sentence to predict sentiment (q = exit):")
            input_text = input("Sentence: ").strip()
            if os.path.exists("model/bert_sentiment_model.pt"):
                if input_text.lower() == 'q':
                    break
                sentiment = predictor.predict(input_text)
                print(f"Prediction: The overall sentiment of this sentence is {sentiment}")
   
   
   
    # # Step 1: Split the dataset
    # split_data(data_path)

    # # Step 2: Train the model
    # train_model("assets/training_data/train.csv")

    # # Step 3: Evaluate the model
    # evaluate_model("assets/training_data/val.csv", "model/bert_sentiment_model.pt")

    # # Step 4: Test the model on unseen prediction test set
    # evaluate_model("assets/training_data/test.csv", "model/bert_sentiment_model.pt")

    # # Example prediction
    # example_text = "This product is fantastic and works perfectly!"
    # sentiment = predict_sentiment(example_text)
    # print(f"Prediction: {sentiment}")

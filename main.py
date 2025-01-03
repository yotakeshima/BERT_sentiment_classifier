import os
from src.train import train_model
from src.evaluate import evaluate_model
from src.predict import predict_sentiment
from src.train import split_data
from src.utils import check_model
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
        load_model:
        while True:
            # Predict a User Input sentence sentiment
            print("\nEnter a sentence to predict sentiment:")
            input_text = input("Sentence: ").strip()
            if os.path.exists("model/bert_sentiment_model.pt"):
                sentiment = predict_sentiment(input_text, "model")
                print(f"Prediction: {sentiment}")
            elif:
                print("Error: Trained model not found. Please train the model first.")

            elif: choice == "q":
                print("Exiting program")
                break
            else:
                print("Invalid choice. Please enter 1, 2, or 'q'.")
   
   
   
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

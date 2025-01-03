import os
def check_model() -> bool: 
    if os.path.exists("model/bert_sentiment_model.pt"):
       print("Loading model and tokenizer...")
       return True
    else:
        print("Trained model not found in the model directory. Please train the model before continuing")
        return False

        
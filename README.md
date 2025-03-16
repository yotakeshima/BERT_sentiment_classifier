# Sentiment Analysis with BERT

## 📌 Overview
This project implements a sentiment analysis model using **BERT (Bidirectional Encoder Representations from Transformers)**. It classifies text as **positive** or **negative** based on Amazon reviews.

## 🚀 Features
- **Data Processing**: Extracts and cleans review data from a JSONL file.
- **Model Training**: Fine-tunes BERT for sentiment classification.
- **Evaluation**: Computes accuracy on test and validation sets.
- **Prediction**: Allows real-time sentiment analysis on user-inputted text.

## 📂 Project Structure
```
📦 sentiment-analysis-bert
├── assets/                 # Processed datasets
├── data_processing/        # Scripts for data loading & label creation
├── model/                  # Trained model & tokenizer
├── src/                    # Main scripts for training & prediction
├── main.py                 # Entry point to train or predict sentiment
├── requirements.txt        # Dependencies list
└── ReadMe.MD               # Project documentation
```

## 📊 Dataset
The dataset consists of **Amazon product reviews**, extracted and processed through:
1. `load_data.py`: Filters reviews based on ASIN.
2. `create_labels.py`: Assigns sentiment labels based on review ratings.

| Rating | Sentiment |
|--------|----------|
| 1-2    | Negative |
| 3+     | Positive |

## 🛠 Installation
### Prerequisites
- Python 3.8+
- Pip package manager

### Install Dependencies
Run the following command to install all required libraries:
```bash
pip install -r requirements.txt
```

## 🔥 Usage
### Train the Model
To train the model from scratch:
```bash
python main.py --train
```

### Predict Sentiment
To use a pre-trained model for sentiment analysis:
```bash
python main.py --predict
```
You will be prompted to enter text for prediction.

### Example Output
```
Enter a sentence to predict sentiment (q = exit):
Sentence: I love this product, it works great!
Prediction: The overall sentiment of this sentence is Positive
```

## 📌 Notes
- If a pre-trained model exists, the script loads it automatically.
- If no model is found, training begins by default.

## 🔍 Future Enhancements
- Support for **neutral sentiment classification**.
- Integration with **a web-based UI** for real-time analysis.

## 📜 License
This project is licensed under the **MIT License**.

---
💡 **Contributions & Issues**: Feel free to submit pull requests or report bugs!

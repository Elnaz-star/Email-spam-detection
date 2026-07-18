# Email Spam Detection

A modular machine learning project for classifying emails as spam or ham using Natural Language Processing and a custom Logistic Regression implementation.

## Project Pipeline

```
Email Text
    |
    v
Text Cleaning
    |
    v
TF-IDF Feature Extraction
    |
    v
Custom Logistic Regression
    |
    v
Spam / Ham Prediction
```

## Model

The project uses Logistic Regression implemented from scratch.

Features:

- Custom TF-IDF vectorizer
- Custom gradient descent training
- Binary classification
- No PyTorch dependency

## Model Evaluation

Dataset split:

- Training: 80%
- Testing: 20%

Test Accuracy:

```
86.99%
```

## Project Structure

```
Email-spam-detection/

├── data/
│   └── spam.csv
│
├── models/
│   ├── model.pkl
│   └── vectorizer.pkl
│
├── src/
│   ├── preprocessing.py
│   ├── tfidf.py
│   ├── logistic_regression.py
│   ├── train.py
│   └── predict.py
│
├── README.md
├── requirements.txt
└── LICENSE
```

## Technologies

- Python
- Pandas
- NumPy
- Natural Language Processing
- Machine Learning

## How To Run

Install dependencies:

```
pip install -r requirements.txt
```

Train the model:

```
python src/train.py
```

Predict a new email:

```
python src/predict.py
```

## Author

Elnaz

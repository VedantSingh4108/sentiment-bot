# Social Sentiment Analysis Bot

An NLP pipeline that analyzes sentiment (positive/negative) from unstructured text like tweets and reviews.

## Tech Stack
- Python 3.11
- Scikit-learn (Logistic Regression, TF-IDF)
- NLTK (Text preprocessing)
- Pandas & NumPy

## Features
- Text cleaning and preprocessing
- TF-IDF vectorization
- Logistic Regression classification
- Performance metrics and visualization

## Setup

1. Clone the repository:
```bash
git clone https://github.com/YOUR-USERNAME/sentiment-bot.git
cd sentiment-bot
```

2. Create virtual environment:
```bash
python -m venv venv
venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Run the analysis:
```bash
python sentiment_analysis.py
```

## How It Works
1. **Text Cleaning**: Removes stopwords and punctuation
2. **TF-IDF Vectorization**: Converts text to numerical features
3. **Classification**: Logistic Regression model predicts sentiment
4. **Evaluation**: Accuracy, precision, recall metrics

## Project Structure
```
sentiment-bot/
├── data/                  # Dataset files
├── models/                # Saved models
├── sentiment_analysis.py  # Main analysis script
├── requirements.txt       # Dependencies
└── README.md             # Documentation
```

## Author
Your Name
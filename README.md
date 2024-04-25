
# Evaluation of Transformers on AI-Generated Tweets

## Description
This project explores the challenge of distinguishing AI-generated tweets from those written by humans. Utilizing Transformer-based models, particularly BERT, we compare their effectiveness against traditional RNNs and LSTMs. Our study leverages a dataset comprising 10,011 AI-generated tweets and 16,000 human-written tweets to test these models. The results show BERT's superior ability in detecting AI-generated content, offering potential enhancements in digital communication integrity.

## Installation

```bash
git clone https://github.com/VictoriaXie/413.git
cd 413
```

## Usage

To run the baseline model:

```bash
python3 baseline.py
```

To run the bert model:

```bash
python3 bert/classifier.py
```

## Technologies Used

- Python
- PyTorch
- BERT

## Dataset

The project uses two main datasets:
- **GenAITweets10k**: 10,011 AI-generated tweets.
- **Sentiment140**: 16,000 human-written tweets. 

Both datasets are preprocessed for consistency in format and tokenization.

## License

This project is licensed under the MIT License - see the LICENSE.md file for details.


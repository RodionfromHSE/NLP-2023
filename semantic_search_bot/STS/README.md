# STS
Semantic Text Similiarity

# Text Similarity Model

This repository contains the source code and notebooks for a text similarity model. The model is trained and fine-tuned on a dataset of stack traces, which has been converted into an STS dataset.

## Contents

- [STS](#sts)
- [Text Similarity Model](#text-similarity-model)
  - [Contents](#contents)
  - [Notes](#notes)
  - [Models Comparison](#models-comparison)
- [Interpretation](#interpretation)

## Notes

1. `pip3 install -r requirements.txt` to install dependencies
2. Dataset used for STS fine-tune: `stsb_multi_mt` (en)
3. Stack traces [source](https://www.kaggle.com/datasets/simiotic/python-tracebacks)

## Models Comparison


In this section, I compare different models and their performance on the text similarity task. I have tried various pre-trained models, such as `paraphrase-MiniLM-L6-v2` and `paraphrase-distilroberta-base-v1`, and compared their performance using metrics like cosine similarity to measure semantic similarity between embeddings. The motivation behind using an embedding-based approach is that it can capture the semantic meaning of the text, making it suitable for measuring text similarity.

The `models_comparison` notebook contains the code and results of these comparisons. Based on the results, I chose the model that performed best on the dataset.

# Interpretation

Let's analyze the model's performance for each pair of sentences: 

1. **Example 1**
    - Sentence1: "US, China fail to paper over cracks in ties"
    - Sentence2: "China: Relief in focus as hope for missing fades"
    - Ground truth similarity: 0.08
    - Model prediction: 0.266
    - Difference: 0.186

   The model overestimates the similarity between these two sentences, which are about different topics related to China.

2. **Example 2**
    - Sentence1: "A man with a bicycle at a coffee house."
    - Sentence2: "Man walking bicycle to patio of a coffee shop."
    - Ground truth similarity: 0.64
    - Model prediction: 0.828
    - Difference: 0.188

   The model overestimates the similarity between these two sentences, which describe similar scenes involving a man and a bicycle at a coffee shop.

3. **Example 3**
    - Sentence1: "Two men standing in grass staring at a car."
    - Sentence2: "A woman in a pink top posing with beer."
    - Ground truth similarity: 0.04
    - Model prediction: -0.053
    - Difference: 0.093

   The model slightly underestimates the similarity between these two sentences, which describe unrelated scenes.

4. **Example 4**
    - Sentence1: "Some men are sawing."
    - Sentence2: "Men are sawing logs."
    - Ground truth similarity: 0.68
    - Model prediction: 0.727
    - Difference: 0.047

   The model provides a close estimate of the similarity between these two sentences, which describe men sawing (logs).

5. **Example 5**
    - Sentence1: "A baby panda goes down a slide."
    - Sentence2: "A panda slides down a slide."
    - Ground truth similarity: 0.88
    - Model prediction: 0.913
    - Difference: 0.033

   The model provides a close estimate of the similarity between these two sentences, which describe a panda going down a slide.

Overall, the model seems to perform well in some cases, but it tends to overestimate the similarity between sentences in other cases. To improve the model's performance, you could try different approaches, such as:
- Using a different pre-trained model (e.g., BERT, RoBERTa, GPT-2 (?)) and fine-tuning it on the STS dataset
- Try feature extraction methods (e.g., TF-IDF, word2vec, GloVe, etc.)
- Collect more data
- Use a different metric to evaluate the model's performance and loss functions
# DSE-697

**Large Language Modeling and Generative AI Projects**

This document summarizes two projects developed for the DSE 697 course, focused on applying and fine-tuning Large Language Models (LLMs) for Persian language tasks.

---

## 📘 Project 1: Fine-Tuning BERT for Persian Language Modeling

This project fine-tunes a pre-trained BERT model (`google-bert/bert-base-uncased`) on Persian text to enhance performance in **masked language modeling** tasks.

### Key Steps

1. **Tokenizer Training**
  
  - A new WordPiece tokenizer is trained using the `RohanAiLab/persian_daily_news` dataset.
  - The tokenizer is specifically optimized for Persian and uploaded to the Hugging Face Hub.
2. **Data Preprocessing**
  
  - Uses the `RohanAiLab/persian_blog` dataset for fine-tuning.
  - Preprocessing includes:
    - Tokenizing with the new Persian tokenizer.
    - Chunking text into manageable segments.
    - Applying **whole-word masking**: masking entire words instead of subwords for improved learning.
3. **Model Fine-tuning**
  
  - The pre-trained BERT model is loaded and adjusted to match the new tokenizer’s vocabulary.
  - Fine-tuning is performed with Hugging Face’s `Trainer` API.
  - Important hyperparameters such as batch size, learning rate, and training epochs are configured and tracked.
4. **Performance Evaluation**
  
  - Evaluation is conducted using a fill-mask task:  
    Example: `"سلام، من [MASK] هستم"` (Hello, I am [MASK])
    - Performance before and after fine-tuning is compared to assess model improvements.

---

## 🍽️ Project 2: Persian Food Sentiment Analysis using Hugging Face Transformers

This project fine-tunes a domain-specific Persian BERT model (`shekar-ai/BERT-base-Persian`) to classify sentiment in **Persian food reviews**.

### Key Steps

1. **Data Loading and Tokenization**
  
  - Loads the dataset from `asparius/Persian-Food-Sentiment` on Hugging Face.
  - Tokenizes input using the associated tokenizer.
2. **Model Setup**
  
  - Initializes BERT for binary **sequence classification** (positive vs. negative sentiment).
3. **Training**
  
  - Fine-tunes the model using the `Trainer` API.
  - Defines training parameters: learning rate, batch size, number of epochs, evaluation strategy.
  - Saves the best-performing model and pushes it to the Hugging Face Hub.
4. **Model Comparison**
  
  - Evaluates sentiment predictions using both the **base** and **fine-tuned** models on sample sentences.
  - Highlights the performance improvement from fine-tuning.

### 📈 Results

- The fine-tuned model shows **notable improvement** in accuracy and sentiment prediction quality compared to the pre-trained version.
- Demonstrates the value of domain-specific fine-tuning for natural language understanding in Persian.
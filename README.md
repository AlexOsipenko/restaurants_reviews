# Restaurants Reviews Analysis

Welcome to the Restaurants Reviews Analysis project! This repository contains tools and scripts for analyzing restaurant reviews, including classification of reviews and visualization of frequent words in positive and negative reviews.

## Table of Contents

- [Introduction](#introduction)
- [Features](#features)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
- [Usage](#usage)
  - [Converting TSKV to DataFrame](#converting-tskv-to-dataframe)
  - [Classification](#classification)
  - [Visualization](#visualization)
- [Project Structure](#project-structure)
- [API](#api)
- [License](#license)

## Introduction

The Restaurants Reviews Analysis project aims to explore and demonstrate various techniques in natural language processing (NLP) and data visualization using restaurant reviews. This project includes functionalities for classifying reviews, visualizing frequent words in positive and negative reviews, and interacting with a conversational AI assistant to generate and obtain reviews.
## Features

- **Review Classification:** Classify reviews into positive or negative categories.
- **Word Frequency Visualization:** Visualize the most frequent words in positive and negative reviews.
- **Conversational AI:** Interact with an AI assistant to generate and retrieve reviews.
- **Data Preprocessing:** Clean and preprocess review text data.

## Getting Started

### Prerequisites

Ensure you have the following software installed:
- Python 3.10 or higher
- `pandas`
- `nltk`
- `sqlalchemy`
- `tensorflow`
- `transformers`
- `bs4`
- `openai`
- `sklearn`
- `wordcloud`
- `matplotlib`
- `numpy`
### Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/AlexOsipenko/restaurants_reviews.git
    cd restaurants_reviews
    ```

2. Install the required Python packages:
    ```bash
    pip install pandas nltk sqlalchemy tensorflow transformers beautifulsoup4 openai scikit-learn wordcloud matplotlib numpy
    ```

3. Set up OpenAI API:
    - Create a file named `.env` in the project root directory.
    - Add your OpenAI API key to the `.env` file:
        ```plaintext
        OPENAI_API_KEY=your_openai_api_key
        ```
## Usage
### Data Preprocessing
Data preprocessing involves cleaning and preparing the text data for analysis. This includes removing HTML tags, punctuation, and stop words, as well as converting text to lowercase. This ensures the data is in a consistent format for further analysis.

File: gpt_restaurant.ipynb
Tools: `BeautifulSoup`, `re`, `nltk`
```python
def clean_text(text):
    text = BeautifulSoup(text, "html.parser").get_text()
    text = re.sub(r'[^\w\s]', '', text)
    text = text.lower()
    return text

df_restaurants['text'] = df_restaurants['text'].apply(clean_text)
```
### Converting TSKV to DataFrame

TSKV (Tab-Separated Key-Value) is a structured data format. This project includes functionality to convert TSKV formatted data directly into a pandas DataFrame, which is easier to manipulate and analyze in Python.

File: gpt_restaurant.ipynb
Tools: pandas
```python
def tsv2json(input_file, output_file):
    arr = []
    with open(input_file, 'r', encoding='utf-8') as file:
        for line in file:
            items = line.split('\t')
            d = {}
            for item in items:
                key, value = item.split('=', 1)
                d[key.strip()] = value.strip()
            arr.append(d)
    with open(output_file, 'w', encoding='utf-8') as file:
        json.dump(arr, file, ensure_ascii=False, indent=4)

tsv2json(filename_tskv, filename_json)
df = pd.read_json(filename_json, encoding='utf-8')
df_restaurants = df[df['rubrics'].str.contains('Ресторан', case=False)]
```
### Classification
The classification functionality uses the OpenAI API to classify restaurant reviews as positive or negative. This demonstrates how NLP and machine learning can be applied to sentiment analysis in text data.

File: gpt_restaurant.ipynb
Tools: openai, transformers, tensorflow
```python
from transformers import BertTokenizer
import tensorflow as tf

texts = df_restaurants['text'].tolist()
labels = df_restaurants['label'].tolist()
tokenizer = BertTokenizer.from_pretrained('DeepPavlov/rubert-base-cased')
tokens = tokenizer(texts, padding=True, truncation=True, max_length=100, return_tensors="tf")
input_ids = tokens['input_ids']
attention_masks = tokens['attention_mask']
```
### Visualization
Visualization helps identify trends and common themes in data. This project includes functionality to generate word clouds for positive and negative reviews, allowing users to see the most frequent words and phrases in the reviews. This can provide insights into what customers like or dislike about a restaurant.

File: gpt_restaurant.ipynb
Tools: wordcloud, matplotlib
```python
from wordcloud import WordCloud
import matplotlib.pyplot as plt

positive_reviews = df_restaurants[df_restaurants['label'] == 1]['text']
negative_reviews = df_restaurants[df_restaurants['label'] == 0]['text']
positive_reviews_string = ' '.join(positive_reviews)
negative_reviews_string = ' '.join(negative_reviews)

positive_cloud = WordCloud(width=800, height=400, background_color='white').generate(positive_reviews_string)
negative_cloud = WordCloud(width=800, height=400, background_color='white').generate(negative_reviews_string)

plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.imshow(positive_cloud, interpolation='bilinear')
plt.axis('off')
plt.title('Positive Reviews')

plt.subplot(1, 2, 2)
plt.imshow(negative_cloud, interpolation='bilinear')
plt.axis('off')
plt.title('Negative Reviews')

plt.show()
```
Example Visualization
Here's an example of a word cloud for positive reviews:
<p align="center">
   <img src=https://github.com/user-attachments/assets/45544946-c26e-41f2-9c2d-0dd368fac9b7 alt="Positive Reviews Word Cloud">
</p>
<p align="center">
And an example for negative reviews:
  <img src=(https://github.com/user-attachments/assets/506656e9-3420-42ec-ac03-8ed32f32ad4d alt="Positive Reviews Word Cloud">
</p>

## Project Structure

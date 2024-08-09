<p align="center">
   [![GitHub](<svg role="img" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg"><title>GitHub</title><path d="M12 .297c-6.63 0-12 5.373-12 12 0 5.303 3.438 9.8 8.205 11.385.6.113.82-.258.82-.577 0-.285-.01-1.04-.015-2.04-3.338.724-4.042-1.61-4.042-1.61C4.422 18.07 3.633 17.7 3.633 17.7c-1.087-.744.084-.729.084-.729 1.205.084 1.838 1.236 1.838 1.236 1.07 1.835 2.809 1.305 3.495.998.108-.776.417-1.305.76-1.605-2.665-.3-5.466-1.332-5.466-5.93 0-1.31.465-2.38 1.235-3.22-.135-.303-.54-1.523.105-3.176 0 0 1.005-.322 3.3 1.23.96-.267 1.98-.399 3-.405 1.02.006 2.04.138 3 .405 2.28-1.552 3.285-1.23 3.285-1.23.645 1.653.24 2.873.12 3.176.765.84 1.23 1.91 1.23 3.22 0 4.61-2.805 5.625-5.475 5.92.42.36.81 1.096.81 2.22 0 1.606-.015 2.896-.015 3.286 0 .315.21.69.825.57C20.565 22.092 24 17.592 24 12.297c0-6.627-5.373-12-12-12"/></svg>)]([https://github.com/yourusername](https://github.com/AlexOsipenko))
   [![Telegram](https://img.shields.io/badge/Telegram-2CA5E0?style=for-the-badge&logo=telegram&logoColor=white)](https://t.me/yourusername)
   [![HeadHunter](https://img.shields.io/badge/HeadHunter-FF0000?style=for-the-badge&logo=hh&logoColor=white)](https://hh.ru/resume/yourprofile)
</p>
<p align="center">
   <img src=https://github.com/user-attachments/assets/4078b96c-760f-4868-80dc-832757dd7368 alt="logo">
</p>

# Restaurants Reviews Analysis

Welcome to the Restaurants Reviews Analysis project! This repository contains tools and scripts for analyzing restaurant reviews, including classification of reviews and visualization of frequent words in positive and negative reviews.

## Table of Contents

- [Introduction](#introduction)
- [Features](#features)
- [Dataset](#dataset)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
- [Preparation](#preparation)
  - [Converting TSKV to DataFrame](#converting-tskv-to-dataframe)
  - [Classification](#classification)
  - [Visualization](#visualization)
- [Project Structure](#project-structure)
- [API](#api)
- [License](#license)

## Introduction

The Restaurants Reviews Analysis project aims to explore and demonstrate various techniques in natural language processing (NLP) and data visualization using restaurant reviews. This project includes functionalities for classifying reviews, visualizing frequent words in positive and negative reviews.

## Features

- **Review Classification:** Classify reviews into positive or negative categories.
- **Word Frequency Visualization:** Visualize the most frequent words in positive and negative reviews.
- **Conversational AI:** Interact with an AI assistant to generate and retrieve reviews.
- **Data Preprocessing:** Clean and preprocess review text data.

## Dataset

The dataset used in this project is provided by Yandex and is available as an open-source dataset. It has been released by Yandex as part of their commitment to supporting research and development in the fields of natural language processing and sentiment analysis. The dataset can be freely accessed and used for various analytical purposes, including the development of machine learning models like the one demonstrated in this project.

By utilizing a dataset from a reputable source like Yandex, I ensure the quality and relevance of the data used in this analysis. The dataset's open-source nature also promotes transparency and allows others to replicate and build upon the work done in this project.

## Getting Started

### Prerequisites

Ensure you have the following software installed:
- Python 3.10 or higher
- `pandas` `nltk` `nltk` `sqlalchemy` `tensorflow` `transformers` `bs4` `sklearn` `wordcloud` `matplotlib` `numpy`

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


## Preparation
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
And an example for negative reviews:
<p align="center">
  <img src=https://github.com/user-attachments/assets/23e03b29-0bf5-4555-8ba2-dbd80f94edb1 alt="Negative Reviews Word Cloud">
</p>

## Model Description
File: gpt_restaurant.ipynb
Tools: `tensorflow`, `TFBertModel`, `nltk`
The model used for classifying restaurant reviews is based on the BERT architecture, specifically leveraging the 'DeepPavlov/rubert-base-cased' pre-trained model. This section provides a detailed description of the model architecture and the reasoning behind the choice of layers.

Load Pre-trained BERT Model:

```python
bert_model = TFBertModel.from_pretrained('DeepPavlov/rubert-base-cased', from_pt=True)
```

BERT (Bidirectional Encoder Representations from Transformers) is one of the most powerful models for natural language understanding tasks. We chose the 'DeepPavlov/rubert-base-cased' model because it is fine-tuned on the Russian language, making it highly suitable for processing and understanding Russian restaurant reviews.

Define Input Layers:

```python
input_ids = Input(shape=(100,), dtype=tf.int32, name='input_ids')
attention_mask = Input(shape=(100,), dtype=tf.int32, name='attention_mask')
```

The input_ids and attention_mask are essential components for feeding data into the BERT model. The input_ids represent the tokenized review text, while the attention_mask indicates which tokens should be attended to by the model (useful for handling padded sequences).

BERT Layer:

```python
def bert_layer(inputs):
    input_ids, attention_mask = inputs
    return bert_model(input_ids=input_ids, attention_mask=attention_mask)[0]

bert_output = Lambda(bert_layer, output_shape=(100, 768))([input_ids, attention_mask])
```
I wrap the BERT model in a Lambda layer to integrate it into our custom architecture. The BERT model processes the input tokens and produces contextual embeddings for each token. These embeddings capture rich semantic information about the text.

Pooling and Dropout Layers:

```python
x = GlobalAveragePooling1D()(bert_output)
x = Dropout(0.3)(x)
```

GlobalAveragePooling1D: This layer aggregates the token embeddings by averaging them across the sequence length. This results in a fixed-size output regardless of the input sequence length, which simplifies further processing and helps capture the overall meaning of the review.
Dropout: We apply a dropout rate of 30% to prevent overfitting. Dropout randomly deactivates neurons during training, which encourages the model to learn robust features rather than relying on specific neurons.
Output Layer:

```python
output = Dense(1, activation='sigmoid')(x)
Reasoning:
The final dense layer has a single neuron with a sigmoid activation function. This setup is ideal for binary classification tasks, as it outputs a probability between 0 and 1, indicating the likelihood of a review being positive.
```

Compile and Train the Model:

```python
model = Model(inputs=[input_ids, attention_mask], outputs=output)
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
model_checkpoint = ModelCheckpoint(
    filepath='model_reviews.keras',
    save_best_only=True,
    monitor='val_loss',
    mode='min',
    save_weights_only=False,
    verbose=1
)

history = model.fit(
    [train_inputs, train_masks],
    train_labels,
    validation_data=([test_inputs, test_masks], test_labels),
    epochs=50,
    batch_size=16,
    callbacks=[early_stopping, model_checkpoint],
)
```
Adam Optimizer: Chosen for its adaptive learning rate capabilities, making it well-suited for complex models like BERT.
Binary Crossentropy Loss: Suitable for binary classification tasks, penalizing incorrect predictions more severely.
EarlyStopping and ModelCheckpoint: These callbacks help to avoid overfitting and ensure the best model is saved during training.
Evaluate the Model:

```python
model = tf.keras.models.load_model('model_reviews.keras', custom_objects={'bert_layer': bert_layer})
loss, accuracy = model.evaluate([test_inputs, test_masks], test_labels)
print(f'Loss: {loss}')
print(f'Accuracy: {accuracy}')
```
After training, I load the best model saved during training and evaluate its performance on the test set to ensure it generalizes well to unseen data. The evaluation metrics include loss and accuracy, which provide insights into the model's performance.

By combining BERT's powerful contextual embeddings with additional layers for pooling and dropout, I create a robust model capable of accurately classifying restaurant reviews. This architecture leverages state-of-the-art natural language processing techniques to provide reliable results for sentiment analysis.

## Project Structure

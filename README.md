# Restaurants Reviews Analysis

Welcome to the Restaurants Reviews Analysis project! This repository contains tools and scripts for analyzing restaurant reviews, including classification of reviews and visualization of frequent words in positive and negative reviews.

## Table of Contents

- [Introduction](#introduction)
- [Features](#features)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
- [Usage](#usage)
  - [Converting TSKV to DataFrame](#converting)
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
##Converting TSKV to DataFrame

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

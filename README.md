# Restaurants Reviews Analysis

Welcome to the Restaurants Reviews Analysis project! This repository contains tools and scripts for analyzing restaurant reviews, including classification of reviews and visualization of frequent words in positive and negative reviews.

## Table of Contents

- [Introduction](#introduction)
- [Features](#features)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
- [Usage](#usage)
  - [Data Preprocessing](#data-preprocessing)
  - [Classification](#classification)
  - [Visualization](#visualization)
- [Project Structure](#project-structure)
- [API](#api)
- [License](#license)

## Introduction

This project provides a comprehensive solution for analyzing restaurant reviews. It includes functionalities for classifying reviews as positive or negative, visualizing the most frequent words in reviews, and interacting with a conversational AI assistant to obtain and generate reviews.

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



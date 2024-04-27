# Fake Job Postings Detection

This repo contains code for detecting fake job postings using machine learning techniques (`app_1_ml.ipynb`) and various Neural Network techniques(`app_2_nn.ipynb`). The code is written in Python and utilizes various libraries such as Pandas, NumPy, Seaborn, Matplotlib, NLTK, Scikit-learn, BeautifulSoup, Spacy, and WordCloud, LightGBM, XGBoost, Keras, CatBoost, and TensorFlow.

## Overview for approach-1

The code performs the following tasks:

1. Import necessary modules.
2. Import the dataset (`fake.csv`).
3. Data exploration and preprocessing:
   - Handling missing values.
   - Checking for outliers and removing them.
   - Exploring the distribution of categorical and numerical features.
4. Visualization of target variables.
5. Feature engineering and preprocessing:
   - Combining text features.
   - Text preprocessing (removing HTML tags, URLs, special characters, stopwords, lemmatization, etc.).
   - Vectorizing text data using CountVectorizer.
   - Splitting the dataset into training and testing sets.
6. Model building and evaluation:
   - Logistic Regression, Multinomial Naive Bayes, Support Vector Machine, and Decision Tree Classifier are trained and evaluated.
   - Evaluation metrics include accuracy, precision, recall, F1 score, and confusion matrix.
   
## Overview for approach-2


The code performs the following tasks:

1. Import necessary modules and libraries.
2. Import the dataset (`fake.csv`) containing job postings.
3. Data cleaning and preprocessing:
   - Standardize text fields by removing special characters, URLs, and non-alphanumeric characters.
   - Tokenize and lemmatize text data using NLTK.
   - Split the dataset into features (`X`) and target variable (`y`).
4. Data balancing:
   - Use NearMiss technique for balancing the dataset.
5. Model building and evaluation:
   - Train various classification models including Logistic Regression, SGD Classifier, Decision Tree Classifier, Random Forest Classifier, AdaBoost Classifier, Gradient Boosting Classifier, HistGradientBoosting Classifier, LightGBM Classifier, XGBoost Classifier, CatBoost Classifier, and LSTM.
   - Evaluate models using classification metrics such as accuracy, precision, recall, F1 score, ROC-AUC score, and confusion matrix.
   - Compare models' performance with balanced and unbalanced data.
6. Use of TF-IDF Vectorizer and Word Embeddings for text representation.
7. Plotting training and validation loss/accuracy curves for deep learning models.

## Usage

To run the code:

1. Clone this repository:

```bash
git clone https://github.com/RAHULFROST7/Fake-Job-post-Detection.git
```

2. Navigate to the cloned directory:

```bash
cd fake job detection
```

3. Install the required dependencies.

4. Execute the `app_1_ml.ipynb` an machine learing approach or `app_2_nn.ipynb` an neural network approach.

## Requirements

Ensure you have Python installed on your system. Additionally, the following Python packages are required:

- numpy
- pandas
- seaborn
- matplotlib
- nltk
- scikit-learn
- wordcloud
- beautifulsoup4
- spacy
- lightgbm
- xgboost
- keras
- catboost
- tensorflow
- imbalanced-learn
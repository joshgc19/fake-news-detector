Fake News Detector
===================

The proposed project focuses on developing a Fake News Detection System using data science methodologies. The system aims to classify news articles as either fake or real by leveraging the Term Frequency-Inverse Document Frequency (TF-IDF) technique and a tree decision classifier. The project involves several key components, including data preprocessing to clean and standardize the dataset, feature extraction using own TF-IDF implementation to represent the articles as numerical vectors, and using a decision classifier to classify the articles. The resulting model will be tested on a new dataset to evaluate its accuracy and effectiveness in detecting fake news. Overall, the project seeks to contribute to the ongoing efforts in combating misinformation by providing a reliable and efficient fake news detection solution.

[//]: <> (Badges should go here)

![Python](https://img.shields.io/badge/-Python-3776ab?style=flat&logo=Python&logoColor=F7DF1E)
![ScikitLearn](https://img.shields.io/badge/-ScikitLearn-F7931E?style=flat&logo=scikitlearn&logoColor=white)
![Pandas](https://img.shields.io/badge/-Pandas-150458?style=flat&logo=pandas&logoColor=white)
![SciPy](https://img.shields.io/badge/-SciPy-8CAAE6?style=flat&logo=scipy&logoColor=white)
[![License: MIT](https://img.shields.io/badge/License-MIT-brightgreen.svg)](https://opensource.org/licenses/MIT)
![code size](https://img.shields.io/github/languages/code-size/joshgc19/fake-news-recognizer)
![GitHub last commit](https://img.shields.io/github/last-commit/joshgc19/fake-news-recognizer)


# Table of contents
* [Fake News Detector](#fake-news-detector)
  * [Table of contents](#table-of-contents)
  * [Installation](#installation)
  * [Code Structure](#code-structure)
  * [Implementation Overview](#implementation-overview)
    * [Preprocessing](#preprocessing)
    * [Feature Extraction](#feature-extraction)
    * [Classification](#classification)
  * [Data](#data)
  * [Results](#results)
    * [Training](#training)
    * [Testing](#testing)
   * [Future Work](#future-work)
  * [Acknowledgements](#acknowledgments)
  * [Project Author](#project-author)
  * [License](#license)

# Installation
To install needed dependencies for this project you can use the following pip command:
```bash
pip install requirements.txt
```
The project was implemented using python v3.12.2 and the latest libraries versions.

# Code Structure
```bash
├── data
│   ├── features # Here the features matrices are stores with the words mapping and words list
│   ├── processed # Here is were the pre-processing module will store transformed and cleaned data
│   └── raw # Here is the dataset an is treat as immutable
├── models
│   └── decision_tree_model.joblib # Here is the model as a binary file 
├── src
│   ├── common 
│   │   ├── __init__.py
│   │   └── files_utils.py # Utils needed to retrieve and save files
│   ├── data 
│   │   ├── __init__.py
│   │   └── make_dataset.py # Contains functions that cleans and formats datasets
│   ├── features
│   │   ├── __init__.py
│   │   └── build_features.py # Contains functions that retrieve features from a dataset and creates the recognition model
│   └── models
│   │   ├── __init__.py
│   │   └── train_model.py # Here is the train function for the model
│   ├── __init__.py
│   ├── features_extraction.py # This is the script that extracts the features of a dataset using TF-IDF
│   ├── preprocessing.py # This is the script that processes the datasets and saves them as npz files
│   └── classifier.py # This is the script that trains the model and checks it accuracy
├── requirements.txt
├── LICENSE
├── README.md
└── .gitignore
```

# Implementation Overview
## Preprocessing
During the preprocessing stage, only the "text" column, containing the article text, was considered for analysis. The text data underwent several cleaning steps to prepare it for further processing. These steps included:

1. **Removing Punctuation:** Punctuation marks were removed from the text to focus on the actual words.
2. **Lowercasing:** All text was converted to lowercase to ensure consistency in word representations.
3. **Tokenization:** The text was tokenized into individual words to facilitate further analysis.
4. **Stopwords Removal:** Common stopwords, such as "and," "the," and "is," were filtered out to reduce noise in the data and focus on meaningful words.

By applying these preprocessing techniques, the text data was cleaned and transformed into a format suitable for analysis. Other techniques that could have improved the model's accuracy and efficiency are depicted on the section [*Future Work*](#future-work).

## Feature Extraction
For the features extraction stage, a Term Frequency-Inverse Document Frequency (TF-IDF) vectorization approach was implemented from scratch. TF-IDF is a widely used technique in natural language processing to convert textual data into numerical features.

TF-IDF is a good choice for feature extraction in this context because it transforms qualitative categorization (words in the text) into quantitative categorization (numerical features). This transformation allows the current project to work with the textual data and make predictions based on the numerical features extracted from the text.

## Classification
In the classification stage, the *DecisionTreeClassifier* from the scikit-learn library was utilized. In this project, the *DecisionTreeClassifier* was a good choice for the fake news detection task because of its interpretability, versatility, and ability to handle both numerical and categorical data. Its ability to automatically select relevant features and model non-linear relationships also made it suitable for the task at hand.

# Data
The dataset used for this project is the **ISOT Fake News Dataset**, sourced from Kaggle. This dataset contains two types of articles: fake and real news. The truthful articles were obtained by crawling articles from Reuters.com, a reputable news website. On the other hand, the fake news articles were collected from unreliable websites flagged by Politifact, a fact-checking organization in the USA, and Wikipedia.

The dataset covers a wide range of topics, with a focus on political and world news. It consists of two CSV files: "True.csv," which contains 21,417 articles from Reuters.com, and "Fake.csv," which contains 23,481 articles from various fake news outlets. Each article includes information such as the title, text, type, and publication date.

This dataset provides a robust foundation for training and testing the fake news detector, allowing for thorough analysis and evaluation of the model's performance.

# Results
## Training
During the training stage, a corpus of 33,687 articles was utilized, representing 75% of the entire dataset. This corpus contained a total of 174,005 unique words across all articles. As stated before, the articles were processed using TF-IDF (Term Frequency-Inverse Document Frequency) vectorization. The TF-IDF vectorization resulted in a features matrix with dimensions of 33,687 rows (one for each article) and 174,005 columns (one for each unique word in the corpus). This matrix represents the frequency of each word in each article, weighted by the inverse document frequency to highlight the importance of rare words in distinguishing between fake and real news articles. This features matrix served as the input for training the fake news detection model.
## Testing
The objetive of testing the current project is to determine the accuracy of the generated model. The testing stage utilized a dataset containing 11,224 articles, representing 25% of the entire dataset. The trained fake news detection model achieved an impressive accuracy score of 99.5545% on this testing dataset.

Using the automated testing process within the classification module the following confusion matrix was found: 

|                  | **Predicted True** | **Predicted False** | **Total** |
|:----------------:|:------------------:|:-------------------:|:---------:|
| **Actual True**  |        5257        |         26          |   5283    |
| **Actual False** |         24         |        5917         |   5941    |
|    **Total**     |        5281        |        5943         |   11224   |


# Future Work
In the future, there are several avenues to explore for enhancing the fake news detector's performance. One key area is the investigation of different classifiers to test the accuracy levels of the model. While the Decision Tree Classifier has shown promising results, other classifiers such as Random Forest, Support Vector Machines, or even neural networks could be evaluated to determine if they offer improved accuracy or efficiency.

Additionally, implementing a stemming algorithm, such as the Porter stemming algorithm, could be beneficial in reducing the feature matrix resulting from the training process. This would help in simplifying the model and potentially improving its performance by reducing overfitting and increasing generalization to new data. Stemming algorithms can help in reducing the complexity of the text data by converting words to their base or root form, which can lead to a more efficient and effective feature representation for the classifier.

# Acknowledgments
  * Dataset provided by [Kaggle](https://www.kaggle.com/datasets/emineyetm/fake-news-detection-datasets/data). 
  * Base project tutorial [Learnbay_Official](https://medium.com/@Learnbay_official/step-by-step-guide-to-fake-news-detection-system-from-scratch-f4f04f852b1f).
  * TF-IDF implementation base by [Ashwin M](https://medium.com/@ashwinnaidu1991/creating-a-tf-idf-model-from-scratch-in-python-71047f16494e).

# Project author

## **Joshua Gamboa Calvo**<br>
Computing Engineering Undergraduate & Assistant Researcher<br>
Costa Rica Technological Institute (Instituto Tecnológico de Costa Rica)<br>
<br>
[![LinkedIn](https://img.shields.io/badge/-LinkedIn-0A66C2?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/joshgc19)
[![GitHub](https://img.shields.io/badge/-GitHUB-181717?style=for-the-badge&logo=github&logoColor=white)](https://github.com/joshgc19)
[![Medium](https://img.shields.io/badge/-Medium-white?style=for-the-badge&logo=medium&logoColor=black)](https://medium.com/@joshgc.19)


# License
>You can checkout the full license [here (opens in the same tab)](https://github.com/joshgc19/fake-news-recognizer/blob/master/LICENSE). 

This project is licensed under the terms of the **MIT** license. 
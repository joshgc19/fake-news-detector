from setuptools import find_packages, setup

setup(
    name='src',
    packages=find_packages(),
    version='0.1.0',
    description='This project implements a fake news detection system using a Decision Tree classifier and TF-IDF vectorization. The system is trained on labeled datasets to classify news articles as real or fake. It aims to help identify and mitigate the spread of misinformation.',
    author='Joshua Gamboa Calvo',
    license='MIT',
)

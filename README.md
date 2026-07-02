# Sentiment Analysis Model Benchmark

A machine learning benchmark comparing multiple classical classification algorithms on the IMDb Movie Review Dataset.

The project evaluates how different supervised learning models perform when they use the same dataset, train/test split, TF-IDF text representation, and evaluation process.

## Project Goal

The goal is to classify movie reviews as:

* Positive
* Negative

This is a supervised binary classification problem.

```text
Movie Review
      ↓
TF-IDF Vectorization
      ↓
Machine Learning Model
      ↓
Positive or Negative
```

## Dataset

This project uses the IMDb Dataset of 50K Movie Reviews.

* 50,000 total movie reviews
* 25,000 positive reviews
* 25,000 negative reviews
* Balanced binary classification dataset

The dataset is not included in this repository because of its size.

After downloading it, place it at:

```text
data/IMDB Dataset.csv
```

## Train/Test Split

The dataset is divided into:

* 40,000 training reviews
* 10,000 testing reviews

The training set is used to teach the models. The test set contains unseen reviews and is used to measure how well each model generalizes.

## Text Representation

Machine learning models cannot directly process raw text.

TF-IDF converts each movie review into a numerical feature vector.

```text
"This movie was amazing"
            ↓
          TF-IDF
            ↓
[0.00, 0.73, 0.12, 0.00, ...]
```

The vectorizer uses a maximum of 5,000 text features.

## Models Compared

1. Logistic Regression
2. Multinomial Naive Bayes
3. Linear Support Vector Machine
4. Decision Tree
5. Random Forest

## Results

| Model                   | Test Accuracy |
| ----------------------- | ------------: |
| Logistic Regression     |    **89.50%** |
| Linear SVM              |    **88.80%** |
| Random Forest           |    **85.36%** |
| Multinomial Naive Bayes |    **85.17%** |
| Decision Tree           |    **70.82%** |

Logistic Regression achieved the highest test accuracy.

## Logistic Regression Evaluation

The Logistic Regression model achieved:

* Accuracy: 89.50%
* Negative precision: 0.90
* Negative recall: 0.88
* Negative F1-score: 0.89
* Positive precision: 0.89
* Positive recall: 0.91
* Positive F1-score: 0.90

```text
              precision    recall  f1-score   support

    negative       0.90      0.88      0.89      4961
    positive       0.89      0.91      0.90      5039

    accuracy                           0.90     10000
   macro avg       0.90      0.89      0.89     10000
weighted avg       0.90      0.90      0.89     10000
```

## Model Observations

### Logistic Regression

Logistic Regression performed best. It works well with high-dimensional, sparse TF-IDF features because it learns how strongly individual words push a prediction toward positive or negative sentiment.

### Linear SVM

Linear SVM performed close to Logistic Regression. It attempts to find a separating boundary with the largest possible margin between positive and negative review vectors.

### Multinomial Naive Bayes

Naive Bayes uses word probabilities to classify reviews. It is fast and useful as a text-classification baseline, although its assumption that features are independent can limit performance.

### Decision Tree

The Decision Tree performed considerably worse than the linear models. An unrestricted tree can learn overly specific rules and overfit the training data.

### Random Forest

Random Forest improved substantially over the single Decision Tree by combining the predictions of many trees. However, it still performed worse than the linear models because tree-based algorithms are generally less suited to high-dimensional sparse TF-IDF features.

## Machine Learning Pipeline

```text
Load IMDb Dataset
        ↓
Explore Reviews and Labels
        ↓
Separate Inputs and Labels
        ↓
Create Train/Test Split
        ↓
Fit TF-IDF on Training Reviews
        ↓
Transform Training and Testing Reviews
        ↓
Train Classification Models
        ↓
Predict Test Review Sentiments
        ↓
Evaluate and Compare Results
```

## Technologies Used

* Python 3.9
* pandas
* NumPy
* scikit-learn
* Matplotlib
* Jupyter Notebook
* Git
* GitHub

## Repository Structure

```text
sentiment-analysis-benchmark/
├── data/
│   └── IMDB Dataset.csv
├── notebooks/
│   └── 01_logistic_regression_tfidf.ipynb
├── results/
├── src/
├── .gitignore
├── README.md
└── requirements.txt
```

## Installation

Clone the repository:

```bash
git clone https://github.com/Tejveersingh27/sentiment-analysis-benchmark.git
cd sentiment-analysis-benchmark
```

Create and activate a virtual environment:

```bash
python3 -m venv venv
source venv/bin/activate
```

Install the dependencies:

```bash
pip install -r requirements.txt
```

Place the dataset at:

```text
data/IMDB Dataset.csv
```

Start Jupyter Notebook:

```bash
jupyter notebook
```

Then open:

```text
notebooks/01_logistic_regression_tfidf.ipynb
```

## Concepts Demonstrated

* Supervised learning
* Binary classification
* Features and labels
* Train/test splitting
* Generalization
* Overfitting and underfitting
* Balanced datasets
* TF-IDF feature extraction
* Logistic Regression
* Naive Bayes
* Support Vector Machines
* Decision Trees
* Random Forests
* Ensemble learning
* Accuracy
* Precision
* Recall
* F1-score
* Model benchmarking

## Key Conclusion

The experiment demonstrates that a more complex model does not automatically produce better results.

For this sentiment-classification task, Logistic Regression and Linear SVM performed better than the tree-based models because linear algorithms are well suited to sparse, high-dimensional TF-IDF features.

Model selection should be based on data characteristics, experimental results, evaluation metrics, and generalization performance.

## Next Steps

* Add confusion matrices for every model
* Compare precision, recall, and F1-scores
* Create model-performance visualizations
* Add cross-validation
* Perform hyperparameter tuning
* Move reusable code into the `src/` directory
* Compare classical models with an LSTM
* Compare results with DistilBERT
* Write a research-style comparison report

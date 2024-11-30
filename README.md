
###  Supervised and Unsupervised Learning

Machine learning is a field of artificial intelligence that enables systems to learn patterns and make decisions from data. 
The two primary types of machine learning paradigms are supervised and unsupervised learning.

---

#### **Supervised Learning**

Supervised learning involves training a model on a labeled dataset, where the inputs are paired with corresponding outputs. 
The model learns to map inputs to outputs and is later used for predictions on unseen data.

**Types of Supervised Learning:**
1. **Classification:** The goal is to predict a categorical outcome.
2. **Regression:** The goal is to predict a continuous numerical value.

---

##### **1. Classification Algorithms**

1. **K-Nearest Neighbors (KNN):**
   - A simple algorithm that predicts the class of a data point based on the majority class of its `k` nearest neighbors.
   - **Libraries Used:** `caret`, `class`
   - Example:
     ```r
     library(class)
     data(iris)
     set.seed(123)
     sample_index <- sample(1:nrow(iris), 0.8 * nrow(iris))
     iris_train <- iris[sample_index, ]
     iris_test <- iris[-sample_index, ]
     train_features <- iris_train[, -5]
     train_labels <- iris_train[, 5]
     test_features <- iris_test[, -5]
     predicted_species <- knn(train = train_features, test = test_features, cl = train_labels, k = 3)
     ```
   - **Applications:** Handwriting recognition, recommendation systems.

2. **Naive Bayes:**
   - Based on Bayes' Theorem, assumes independence between features.
   - **Libraries Used:** `e1071`
   - Example:
     ```r
     library(e1071)
     data(iris)
     trainIndex <- createDataPartition(iris$Species, p = 0.7, list = FALSE)
     trainData <- iris[trainIndex, ]
     testData <- iris[-trainIndex, ]
     nb_model <- naiveBayes(Species ~ ., data = trainData)
     predictions <- predict(nb_model, testData)
     ```
   - **Applications:** Spam filtering, sentiment analysis.

3. **Decision Tree:**
   - A tree-like model of decisions and their possible consequences.
   - **Libraries Used:** `rpart`, `rpart.plot`
   - Example:
     ```r
     library(rpart)
     data(iris)
     trainIndex <- createDataPartition(iris$Species, p = 0.7, list = FALSE)
     trainData <- iris[trainIndex, ]
     testData <- iris[-trainIndex, ]
     tree_model <- rpart(Species ~ ., data = trainData, method = "class")
     predictions <- predict(tree_model, testData, type = "class")
     ```
   - **Applications:** Fraud detection, customer segmentation.

---

##### **2. Regression Algorithms**

1. **Linear Regression:**
   - Models the relationship between a dependent variable and one or more independent variables.
   - **Libraries Used:** `base`
   - Example:
     ```r
     bedrooms <- c(2, 3, 4, 3, 5)
     price <- c(200000, 250000, 275000, 240000, 310000)
     model <- lm(price ~ bedrooms)
     predict(model, data.frame(bedrooms = 6))
     ```
   - **Applications:** House price prediction, stock market forecasting.

2. **Multiple Regression:**
   - Extends linear regression to multiple independent variables.
   - **Libraries Used:** `datarium`, `caTools`
   - Example:
     ```r
     library(datarium)
     data("marketing")
     model <- lm(sales ~ youtube + facebook + newspaper, data = marketing)
     predict(model, marketing)
     ```
   - **Applications:** Marketing effectiveness analysis.

---

##### **3. Dual Use Algorithms**

1. **Neural Networks:**
   - Mimics the functioning of the human brain using layers of interconnected nodes.
   - **Libraries Used:** `neuralnet`
   - Example:
     ```r
     library(neuralnet)
     data(concrete)
     concrete_model <- neuralnet(strength ~ cement + slag + ash + water +
                                 superplasticizer + coarseagg + fineagg + age,
                                 data = concrete_train, hidden = c(1, 5))
     ```
   - **Applications:** Image recognition, natural language processing.

2. **Support Vector Machines (SVM):**
   - Separates data points using a hyperplane and works well for both classification and regression.
   - **Libraries Used:** `e1071`
   - Example:
     ```r
     library(e1071)
     classifier <- svm(Species ~ ., data = iris, kernel = "linear")
     predict(classifier, iris)
     ```
   - **Applications:** Face detection, text categorization.

---

#### **Unsupervised Learning**

Unsupervised learning deals with datasets without labeled outputs. The model identifies patterns or structures in the data.

**Types of Unsupervised Learning:**
1. **Clustering:** Grouping similar data points.
2. **Association Rules:** Finding relationships between variables.

---

##### **1. Clustering**

1. **K-Means Clustering:**
   - Partitions data into `k` clusters by minimizing variance within each cluster.
   - **Libraries Used:** `base`
   - Example:
     ```r
     data(iris)
     kmeans_model <- kmeans(iris[, -5], centers = 3)
     ```
   - **Applications:** Market segmentation, document clustering.

---

##### **2. Association Rules**

1. **Apriori Algorithm:**
   - Used for mining frequent item sets and relevant association rules.
   - Example:
     ```r
     library(arules)
     data(Groceries)
     rules <- apriori(Groceries, parameter = list(supp = 0.001, conf = 0.8))
     inspect(rules)
     ```
   - **Applications:** Market basket analysis, recommender systems.

---

#### **Model Performance Metrics**

Evaluating model performance ensures the model's reliability and generalization ability. 

**Metrics for Classification:**
- **Accuracy:** Proportion of correctly predicted instances.
- **Precision:** Proportion of true positive predictions.
- **Recall:** Proportion of actual positives correctly predicted.
- **F1-Score:** Harmonic mean of precision and recall.

**Metrics for Regression:**
- **RMSE (Root Mean Squared Error):** Measures the average error magnitude.
- **R-squared:** Indicates the proportion of variance explained by the model.

---

#### **Advanced Topics**

1. **Ensemble Learning:**
   - Combines multiple models to improve performance.
   - Example: Random Forest, Gradient Boosting.

2. **Dimensionality Reduction:**
   - Reduces the number of features while preserving important information.
   - Example: Principal Component Analysis (PCA).

---

#### **Conclusion**

This document provides an overview of supervised and unsupervised learning techniques, focusing on essential algorithms, 
their applications, and performance evaluation metrics. By understanding these concepts, you can select appropriate methods for 
various machine learning tasks.

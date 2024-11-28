
#### **Supervised Learning**

**1. Classification**
   - **KNN**  
     **Libraries Used:** `caret`, `class`  
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
     
   - **Naive Bayes**  
     **Libraries Used:** `e1071`  
     ```r
     library(e1071)
     data(iris)
     trainIndex <- createDataPartition(iris$Species, p = 0.7, list = FALSE)
     trainData <- iris[trainIndex, ]
     testData <- iris[-trainIndex, ]
     nb_model <- naiveBayes(Species ~ ., data = trainData)
     predictions <- predict(nb_model, testData)
     ```
     
   - **Decision Tree**  
     **Libraries Used:** `rpart`, `rpart.plot`  
     ```r
     library(rpart)
     data(iris)
     trainIndex <- createDataPartition(iris$Species, p = 0.7, list = FALSE)
     trainData <- iris[trainIndex, ]
     testData <- iris[-trainIndex, ]
     tree_model <- rpart(Species ~ ., data = trainData, method = "class")
     predictions <- predict(tree_model, testData, type = "class")
     ```

**2. Regression**
   - **Linear Regression**  
     **Libraries Used:** `base`  
     ```r
     bedrooms <- c(2, 3, 4, 3, 5)
     price <- c(200000, 250000, 275000, 240000, 310000)
     model <- lm(price ~ bedrooms)
     predict(model, data.frame(bedrooms = 6))
     ```
     
   - **Multiple Regression**  
     **Libraries Used:** `datarium`, `caTools`  
     ```r
     library(datarium)
     data("marketing")
     model <- lm(sales ~ youtube + facebook + newspaper, data = marketing)
     predict(model, marketing)
     ```

**3. Dual Use**
   - **Neural Networks**  
     **Libraries Used:** `neuralnet`  
     ```r
     library(neuralnet)
     data(concrete)
     concrete_model <- neuralnet(strength ~ cement + slag + ash + water +
                                 superplasticizer + coarseagg + fineagg + age,
                                 data = concrete_train, hidden = c(1, 5))
     ```
     
   - **SVM**  
     **Libraries Used:** `e1071`  
     ```r
     library(e1071)
     classifier <- svm(Species ~ ., data = iris, kernel = "linear")
     predict(classifier, iris)
     ```

#### **Unsupervised Learning**

**1. Clustering**
   - **K-Means Clustering**  
     **Libraries Used:** `base`  
     ```r
     data(iris)
     kmeans_model <- kmeans(iris[, -5], centers = 3)
     ```
     
**2. Association Rules**  
No working code present in the file.

#### **Model Performance**
   - **Random Forests**  
     **Libraries Used:** `randomForest`  
     ```r
     library(randomForest)
     data(PimaIndiansDiabetes)
     rf_model <- randomForest(diabetes ~ ., data = train, importance = TRUE)
     predict(rf_model, test)
     ```

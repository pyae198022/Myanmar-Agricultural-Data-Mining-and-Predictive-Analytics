# Agricultural Data Analytics and Decision Support System

An ML-powered web application for agricultural data analysis, prediction, clustering, and decision support.

The system combines machine learning and descriptive data mining techniques to analyze agricultural conditions and provide insights for crop type, yield level, crop yield, and agricultural clustering.

---

## Features

### 🌾 Crop Type Prediction

Predict the crop type based on agricultural, environmental, and economic conditions using the deployed machine learning model.

### 📊 Yield Level Prediction

Classify agricultural conditions into yield-level categories using the existing trained classification model.

### 📈 Crop Yield Prediction

Predict crop yield using a Random Forest Regression model.

The system also provides:

- Actual vs Predicted analysis
- Cross-validation reference results
- Feature importance
- Model comparison

### 🔍 Descriptive Mining

The application includes several descriptive data mining techniques:

- Correlation Analysis
- Association Rule Mining
- Frequent Pattern Mining
- Sequential Pattern Mining
- Clustering Analysis

### 🧩 Clustering Analysis

Agricultural records are grouped using K-Means clustering.

The final clustering configuration uses:

- K = 8 clusters
- Log transformation for highly skewed numerical features
- Min-Max Scaling
- Hopkins Statistic for cluster tendency
- Elbow Method
- Silhouette Score

### 📉 Model Evaluation

The application provides model evaluation and comparison features including:

- Accuracy
- Precision
- Recall
- F1-score
- R²
- RMSE
- MAE
- ROC/AUC
- Actual vs Predicted analysis
- Feature Importance
- Clustering evaluation

### 📚 Historical Agricultural Trends

Explore historical agricultural trends and statistics through interactive visualizations.

### 📋 Data Statistics

View dataset-level statistics and agricultural data summaries.

---

## Machine Learning Models

The application uses the existing trained and verified machine learning artifacts.

| Task | Model |
|---|---|
| Crop Type Prediction | Random Forest Classifier |
| Yield Level Prediction | Existing deployed classification model |
| Crop Yield Prediction | Random Forest Regressor |
| Clustering | K-Means (K = 8) |
| Crop Type ROC/AUC | Existing classification models |
| Feature Importance | Random Forest native feature importance |

The application uses pre-trained model artifacts and does not retrain models during prediction.

---

## Clustering

The final clustering analysis uses eight numerical features:

- Sown Acre
- Harvested Acre
- Production Ton
- Fertilizer Import Value (USD)
- Average Temperature
- Total Rainfall
- Myanmar GDP (USD)
- Average Humidity

Highly right-skewed variables are transformed using `log1p`, followed by Min-Max scaling.

Categorical attributes such as region, crop type, soil type, and water source are used for cluster interpretation rather than as direct clustering inputs.

---

# XGBoost Regression - House Price Prediction

## Project Overview
Predicting house prices is a common and valuable task in the real estate market, enabling stakeholders to estimate property values based on a set of features such as square footage, number of bedrooms, location, and other factors. This project applies XGBoost regression to forecast house prices with high accuracy, using historical data and relevant home features from the `kc_house_data.csv` dataset.

## Why XGBoost Regression?
XGBoost (Extreme Gradient Boosting) is a powerful and efficient machine learning algorithm, particularly useful for handling structured and tabular data. XGBoost’s gradient boosting approach allows it to build accurate predictive models by sequentially minimizing errors from previous iterations. Its regularization parameters help to avoid overfitting, making it ideal for regression tasks such as predicting house prices, where high accuracy and model robustness are crucial.

## Steps and Code Overview
The following file demonstrates a step-by-step guide to preparing data, exploring correlations, applying XGBoost regression, and enhancing accuracy through hyperparameter tuning.

### Key Steps
1. **Data Import and Preparation**  
   - Libraries such as `pandas`, `numpy`, and `scikit-learn` are used to handle data and preprocessing.
   - The data is loaded from `kc_house_data.csv`, with key features like `sqft_living`, `grade`, and `bathrooms` identified for prediction. Data is cleaned to remove null values, and irrelevant columns (e.g., ID, date) are dropped.

2. **Exploratory Data Analysis (EDA)**  
   - Correlation analysis identifies features most strongly correlated with price.
   - Visualizations, including scatter plots and box plots, highlight key data patterns.

3. **Splitting Data and Model Training**  
   - The dataset is split into training and testing sets.
   - Initial model training is conducted using XGBoost’s default parameters, achieving a baseline accuracy.

4. **Hyperparameter Tuning with RandomizedSearchCV**  
   - To improve model performance, `RandomizedSearchCV` is used to find optimal parameters, such as `n_estimators`, `learning_rate`, and `max_depth`.
   - After tuning, the model achieves improved accuracy, as shown by higher R² scores and lower RMSE.

5. **Evaluation**  
   - The final model’s R² score and RMSE indicate its predictive accuracy, showing a significant improvement post-tuning.

### Results
- **Initial R² Score**: 0.84
- **Post-Tuning R² Score**: 0.88
- **Root Mean Squared Error (RMSE) Improvement**: From 127,993 to 112,686

This approach highlights the impact of tuning and demonstrates how XGBoost can accurately predict house prices based on historical data and property features.

## How to Run
1. Clone the repository and ensure `kc_house_data.csv` is in the working directory.
2. Install dependencies:
   ```shell
   pip install -r requirements.txt

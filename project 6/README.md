# Taxi Order Prediction
## Project Description
The company has collected historical data on taxi orders at airports. To attract more drivers during peak periods, it is necessary to predict the number of taxi orders for the next hour. Build a model for this prediction.
The RMSE metric on the test set should not exceed 48.

## Data Description
The number of orders is in the column num_orders

## Libraries Used
- Pandas: For data manipulation and analysis.
- NumPy: For numerical operations.
- Matplotlib: For creating visualizations.
- Seaborn: For enhanced data visualization, including heatmaps.
- Scikit-learn: For machine learning tasks, including:
train_test_split: To split the dataset.
RandomizedSearchCV: Hyperparameter tuning
cross_val_score: For cross-validation.
StandardScaler and MinMaxScaler and OrdinalEncoder: For feature scaling.
SimpleImputer: For handling missing data.
Pipeline and ColumnTransformer: For streamlined preprocessing.
DummyRegressor: For baseline performance comparison.
mean_squared_error, mean_absolute_error: Model evaluation metrics.
LinearRegression: Linear regression modeling.
DummyRegressor: Baseline performance comparison.
RandomForestRegressor: Decision tree ensemble modeling.
- LightGBM (LGBMRegressor): For efficient gradient boosting.
- CatBoost (CatBoostRegressor): Gradient boosting optimized for categorical features.
- XGBoost (XGBRegressor): Gradient boosting framework for robust predictions.
- Statsmodels: For statistical analysis, including:
- seasonal_decompose: Seasonal decomposition of time-series data.
- Phik: For calculating correlations using phik_matrix.
- SHAP: For feature importance analysis.

## Findings and Recommendations
During the analysis of taxi order data, preliminary processing of time series was performed, including resampling by the hour to obtain aggregated values. The data was examined for duplicates and missing values, after which new features were created, such as lag values and moving averages, as well as temporal characteristics like month, day, day of the week, and hour. These features were used to build a model for predicting the number of orders using various algorithms, including RandomForest, LightGBM, CatBoost, and XGBoost.
As a result of applying the random search method (RandomizedSearchCV), the optimal regression model was identified: CatBoostRegressor with the following parameters:
{'model__learning_rate': 0.08, 'model__l2_leaf_reg': 5, 'model__iterations': 500, 'model__depth': 5, 'model__border_count': 64, 'model__bagging_temperature': 1}
The model achieved a minimum RMSE of 4.93 on the training set, while on the test set the RMSE was 38.99. The model demonstrated a significant improvement compared to the baseline (constant) model, confirming the effectiveness of the feature engineering approach and cross-validation.
The visualization of residuals and predicted values allowed for an analysis of model quality and identification of areas where the model made errors. The results confirm that the developed model can significantly improve prediction accuracy compared to simpler methods.
# Car Price Prediction
## Project Description
A used car sales service is developing an application to attract new customers. The app allows users to quickly estimate the market value of their vehicles. We have historical data at our disposal, including technical specifications, configurations, and car prices. The task is to build a model for car price prediction.

Key requirements for the client:
- Prediction accuracy
- Prediction speed
- Training time

## Data Description
The data is provided in file.

Features:
- DateCrawled — date when the listing was crawled from the database
- VehicleType — type of car body
- RegistrationYear — vehicle registration year
- Gearbox — type of transmission
- Power — engine power (hp)
- Model — car model
- Kilometer — mileage (km)
- RegistrationMonth — vehicle registration month
- FuelType — type of fuel
- Brand — car brand
- Repaired — whether the car was repaired or not
- DateCreated — date when the listing was created
- NumberOfPictures — number of car images
- PostalCode — postal code of the listing owner
- LastSeen — date of the user's last activity
- Price — price (EUR)

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
mean_squared_error: Model evaluation metric
LightGBM (LGBMRegressor): For efficient gradient boosting
Statsmodels: For statistical analysis:
variance_inflation_factor: Assessing multicollinearity
add_constant: Adding a constant term to features
- Phik: For calculating correlations using phik_matrix
- SHAP: For feature importance analysis

## Findings and Recommendations
As part of the project for the used car sales service, the task was to develop a model for determining the market value of cars. The client's key criteria were prediction accuracy, training time, and prediction speed. The dataset provided included technical specifications, configurations, and car prices, which were used to build the model.
A hyperparameter tuning strategy using RandomizedSearchCV was applied to two models: DecisionTreeRegressor and LGBMRegressor. Data preprocessing was performed using a pipeline that included OrdinalEncoder for categorical features and MinMaxScaler for numerical ones. The best parameters were selected for each model, with LGBMRegressor showing the best results using the parameters learning_rate=0.18, max_depth=8, and n_estimators=150.
The training and evaluation of the models yielded the following results:
Training Time: The Decision Tree model trained faster, completing in 0.75 seconds, while LGBM required 4.66 seconds.
Prediction Speed: Decision Tree also made predictions faster, taking 0.12 seconds, compared to 0.48 seconds for LGBM.
Prediction Accuracy: The Root Mean Squared Error (RMSE) was used to evaluate prediction accuracy — the lower the RMSE, the better the model. LGBM achieved a lower RMSE (1623.73) compared to Decision Tree (2173.26), indicating superior prediction quality.
In conclusion, if prediction accuracy is the primary criterion, LGBM is the better model due to its significantly lower RMSE. However, if training and prediction speed are more critical, the Decision Tree model outperforms LGBM by providing faster training and prediction processes.
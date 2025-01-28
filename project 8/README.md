# Customer Age Definition (CV)
## Project Description
A network supermarket is implementing a computer vision system to process customer photos. Photo documentation at the checkout area will help determine the age of customers in order to:
Analyze purchases and suggest products that may interest customers in this age group.
Monitor the integrity of cashiers when selling alcohol.
A model needs to be built that will estimate a person's approximate age based on their photo. We have a dataset of people's photos with age labels.


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
- Phik: For calculating correlations using phik_matrix.
- tensorflow.keras.preprocessing.image: For image preprocessing using ImageDataGenerator and other functions.
- os: For interacting with the operating system and file paths.
- tensorflow.keras.applications.ResNet50: For the ResNet50 pre-trained model for image recognition.
- tensorflow.keras.models.Sequential: For building neural network models.
- tensorflow.keras.layers: For adding layers to the neural network (e.g., GlobalAveragePooling2D, Dense, Dropout).
- tensorflow.keras.optimizers.Adam: For setting the Adam optimizer.
- tensorflow.keras.callbacks: For callbacks like EarlyStopping and ReduceLROnPlateau to manage training.
- tensorflow.keras.regularizers.l2: For adding regularization to the model.

## Findings and Recommendations
The project aimed to develop a computer vision system for the retail supermarket "Khleb-Sol," designed to automatically determine the age of customers based on photos taken in the checkout area. The system is intended for more accurate audience segmentation and control over the sale of age-restricted products, such as alcohol. This would allow for personalized recommendations to customers and help ensure cashiers adhere to age-restriction regulations.
For the project, a dataset was used, containing 7,591 records with age information and image file names. The age of individuals in this dataset ranged from 1 to 100 years, with a distribution skewed toward younger age groups: the average age was 31.2 years, and the median age was 29 years. The images featured faces of both genders, captured from various angles, with different lighting conditions and facial expressions, making the task more challenging.
To enhance the model's robustness, data augmentation techniques were applied, such as rotation, brightness adjustment, and scaling. The model was built using the ResNet50 architecture as a pre-trained base, with an additional final regression layer for age prediction. The data was split into training and validation (test) sets, with image normalization and augmentation applied. During training, the Adam optimizer, regularization, and learning rate reduction mechanisms were used to stabilize validation metrics.
Upon completion, the model achieved a mean absolute error (MAE) of 6.68 on the validation set, which is a good result considering the diversity and distribution of the data. The model showed high resilience to image variability and is ready for deployment on new data, contributing to audience segmentation and control over age-restricted product sales.
For further improvement, the training dataset could be expanded with new images, especially from underrepresented age groups (which would help the model better distinguish features characteristic of different ages). Current augmentation methods (rotation, brightness adjustment, scaling) could be supplemented with new techniques, such as adding noise or using generative learning methods (e.g., GANs) to create synthetic data. Additionally, more refined hyperparameter tuning, including learning rate, regularization coefficients, and layer parameters, could potentially lead to better model training. Exploring more complex architectures beyond ResNet50 may also be beneficial. Furthermore, more meticulous image preprocessing, such as brightness and color normalization or face detection, could improve the model's performance.
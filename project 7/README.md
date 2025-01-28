# Project for "Wikishop" with BERT
## Project Description
The online store is launching a new service. Now users can edit and supplement product descriptions, similar to wiki communities. Customers can propose their edits and comment on others' changes. The store needs a tool that will identify toxic comments and send them for moderation.
We will train a model to classify comments as positive or negative. We have a dataset with labels indicating the toxicity of the edits.
The goal is to build a model with an F1 score of at least 0.75.

## Data Description
The column "text" contains the comment text, and "toxic" is the target variable.

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
mean_squared_error, mean_absolute_error: Model evaluation metrics.
LogisticRegression, KNeighborsClassifier, DecisionTreeClassifier: Classification models.
- RandomForestClassifier: Decision tree ensemble modeling.
- LGBMClassifier: Efficient gradient boosting.
- CatBoostClassifier: Gradient boosting optimized for categorical features.
- XGBClassifier: Gradient boosting framework for robust predictions.
- Statsmodels: For statistical analysis, including:
seasonal_decompose: Seasonal decomposition of time-series data.
- Phik: For calculating correlations using phik_matrix.
- SHAP: For feature importance analysis.
- Torch and Transformers: For using BERT-based models for text classification.
- NLTK: For sentiment analysis with SentimentIntensityAnalyzer.
- WordCloud: For generating word clouds for text data visualization.
- Pickle: For saving and loading model objects.

## Findings and Recommendations
As part of the project, a tool was developed for the online store to automatically identify toxic comments among user edits and suggestions. This tool will allow potentially offensive and toxic comments to be sent for moderation, thereby maintaining a friendly and constructive atmosphere on the platform. To achieve this goal, the following steps were taken: The dataset containing comment texts and toxicity labels was loaded, explored, and preprocessed. 
Each comment was tokenized using BertTokenizer, after which tensors were generated. 
A pre-trained BERT model (unitary/toxic-bert), specialized in detecting toxic statements, was used in the project. With this model, embeddings were obtained for each comment in batches. Several algorithms were tested for classification, including: RandomForestClassifier, LGBMClassifier, XGBClassifier, and LogisticRegression. 
For each of them, RandomizedSearchCV with cross-validation was used to select hyperparameters, optimizing the f1 metric. The best model turned out to be RandomForestClassifier with the following parameters: max_depth=15, min_samples_leaf=4, n_estimators=900, random_state=42. The best f1 score on the training set was 0.9490, and on the test data, it was 0.9448. 
The close values on both the training and test sets (0.9490 and 0.9448) confirm that the model is not overfitted. 
The model showed significant superiority compared to the constant DummyClassifier model, confirming the effectiveness of the feature engineering approach and cross-validation.
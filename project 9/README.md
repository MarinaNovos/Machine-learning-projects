# Customer churn prediction for a telecom operator
## Project Description
The telecom operator is facing the problem of customer churn, which is one of the most pressing challenges in the telecommunications industry. Losing customers can lead to reduced profits, a damaged reputation, and high costs for acquiring new subscribers. 
To effectively combat churn, the company needs to proactively identify users who may cancel their services and offer them attractive conditions, such as promo codes or special discounts. This not only helps reduce churn but also boosts customer loyalty, which holds long-term value for the business.
To address this issue, a model is needed that can predict whether a subscriber will terminate their contract. The operator's team has already gathered extensive customer information, including personal data, details about the tariffs and services they use. 
These data provide unique opportunities to develop a model that will analyze customer behavior and identify potential users who are likely to leave.
In this project, our goal is to train a model using the customer data provided by the operator to predict subscriber churn. The effectiveness of such a model will not only help reduce churn but also optimize marketing efforts by offering personalized retention conditions for customers.

## Data Description
The data consists of several files obtained from different sources:
Contract information
Customer personal data
Internet service details
Telephony service details

Contract Information:
- customerID — subscriber identifier
- BeginDate — contract start date
- EndDate — contract end date
- Type — payment type: annually, bi-annually, or monthly
- PaperlessBilling — paperless billing option
- PaymentMethod — payment method
- MonthlyCharges — monthly charges
- TotalCharges — total charges for the subscriber

Personal Data:
- customerID — user identifier
- gender — gender
- SeniorCitizen — whether the subscriber is a senior citizen
- Partner — whether the subscriber has a partner
- Dependents — whether the subscriber has dependents

Internet Services:
- customerID — user identifier
- InternetService — type of internet connection
- OnlineSecurity — protection from dangerous websites
- OnlineBackup — cloud storage for data backup
- DeviceProtection — antivirus protection
- TechSupport — dedicated technical support line
- StreamingTV — streaming television service
- StreamingMovies — movie streaming catalog

Telephony Services:
- customerID — user identifier
- MultipleLines — connection to multiple phone lines simultaneously

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
confusion_matrix, make_scorer, roc_auc_score, accuracy_score, roc_curve: For model evaluation.
- Statsmodels: For statistical analysis, including:
- variance_inflation_factor: For checking multicollinearity.
- Phik: For calculating correlations using the phik_matrix.
- SHAP: For feature importance analysis.
- Gradient Boosting Models:
CatBoostClassifier: Gradient boosting optimized for categorical features.
LGBMClassifier: Efficient gradient boosting.
XGBClassifier: Robust gradient boosting framework.

## Findings and Recommendations
The telecom operator "TeleDom" faced the issue of customer churn and tasked us with predicting which subscribers may terminate their contracts. To address this challenge, we needed to develop a model capable of predicting the likelihood of customer churn based on collected data. The company offered a wide range of services, including landline phone services, internet, and additional services such as antivirus software, technical support, cloud storage, and streaming services. To train the model, we went through a data integration process, merging several tables containing information about customers, their tariffs, and services. 
As a result, a unified table was created, ready for analysis and further model development.
During the data merging process, we identified missing values due to the absence of information on certain services, such as internet and telephone. Data preprocessing involved several key steps: converting the data types for date columns into the datetime format, correcting cost data initially presented as text and converting it into numeric values, and creating a new feature indicating whether the customer left or remained active. 
Missing values in columns related to additional services were filled with logical values such as 'None' for unavailable services. After preprocessing, exploratory data analysis was performed. It revealed that monthly spending by subscribers had a skewed distribution with peaks at certain values, while total customer spending also exhibited a skewed distribution with a clear peak. Contract duration data showed two groups of customers, one preferring short-term contracts and the other long-term ones.
Following this, a correlation analysis was conducted, and based on the results, features with low relevance to the classification task, such as dependents, internet_service, and gender, were excluded. These features were removed from the training data due to their low correlation with the target variable or high multicollinearity with other features. To address the customer churn prediction task, three models were trained: LGBMClassifier, XGBClassifier, and CatBoostClassifier. 
Among them, the best results were demonstrated by the CatBoostClassifier model with optimized hyperparameters: 'preprocessor': 'passthrough', 'model__scale_pos_weight': 1, 'model__learning_rate': 0.1, 'model__iterations': 1500, 'model__depth': 6. This model achieved a ROC AUC of 0.9021 and an accuracy of 91.8% on the test set. Analysis of the confusion matrix allowed us to make the following conclusions: the model correctly predicted that most customers would stay, as well as identified a portion of customers who would leave the service. 
However, the model made some errors. It incorrectly predicted that a few customers would stay when they were planning to leave, and some customers were wrongly predicted to leave when they stayed. This suggests that the model performs well in identifying retained customers but occasionally makes errors in predicting churn, missing some customers who intended to leave.
A comparison with a constant model showed that our best model significantly outperforms it. This confirms that all efforts in creating new features, tuning them, and performing cross-validation were justified, and our model significantly improves prediction accuracy compared to simple approaches. 
In conclusion, feature importance evaluation was conducted, allowing us to identify the most influential characteristics on model predictions: contract duration, monthly spending, and total spending over the entire period. This information can be useful for further model optimization and result interpretation, as well as for the business in understanding which factors influence customer churn. 
Overall, there is still room for improvement in our model to better predict churn. For example, we could gather or artificially synthesize data for the minority class (as we have already tested balancing models) or reevaluate the classification threshold. 
For instance, adjusting this threshold based on the precision-recall curve could help reduce the number of false negative results.
Based on the data analysis and the churn prediction model, some recommendations for the business can be formulated. From the feature importance analysis, we identified the factors that have the greatest impact on a customer's decision to leave the company. These factors should be the focus of customer retention strategies. For example, if the model shows that contract duration, payment type, or usage of specific services are important features, efforts should be made to improve these aspects. Using model predictions, we can identify groups of customers with a high risk of churn. Personalized offers, such as discounts, special conditions, or better service quality, should be developed to retain these customers. Services that are not actively used by most customers could be improved or offered in a different format (e.g., targeting customers with multiple lines, online backup, device protection, streaming TV, and streaming movies). These services could be offered for free or as bonuses to increase their appeal and reduce churn. Since many customers tend to prefer long-term contracts, additional benefits could be offered to customers who select short-term contracts. This could include loyalty programs, extra bonuses, or discounts for switching to long-term contracts, helping to retain them and reduce churn risk in the future. Different payment channels, such as electronic checks, postal checks, or automatic transfers, may influence churn levels. Offering convenient and advantageous conditions for each payment channel could help increase customer satisfaction and reduce the likelihood of churn.
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error
import xgboost as xgb

# Load the datasets (replace file paths with actual ones)
users_train = pd.read_csv("C:/Users/Lenovo/OneDrive/Masaüstü/users_train.csv")
user_features_train = pd.read_csv("C:/Users/Lenovo/OneDrive/Masaüstü/user_features_train.csv")
targets_train = pd.read_csv("C:/Users/Lenovo/OneDrive/Masaüstü/targets_train.csv")

users_test = pd.read_csv("C:/Users/Lenovo/OneDrive/Masaüstü/users_test.csv")
user_features_test = pd.read_csv("C:/Users/Lenovo/OneDrive/Masaüstü/user_features_test.csv")

# Merge the datasets on 'ID'
train_data = pd.merge(users_train, user_features_train, on='ID')
train_data = pd.merge(train_data, targets_train, on='ID')  # Include 'TARGET' column
test_data = pd.merge(users_test, user_features_test, on='ID')

# Convert 'first_open_timestamp' to seconds by dividing by 1e9
train_data['first_open_timestamp'] = pd.to_datetime(train_data['first_open_timestamp'] / 1e9, unit='s')
test_data['first_open_timestamp'] = pd.to_datetime(test_data['first_open_timestamp'] / 1e9, unit='s')

# Extract time-based features from 'first_open_timestamp'
train_data['first_open_day'] = train_data['first_open_timestamp'].dt.dayofweek
train_data['first_open_hour'] = train_data['first_open_timestamp'].dt.hour
test_data['first_open_day'] = test_data['first_open_timestamp'].dt.dayofweek
test_data['first_open_hour'] = test_data['first_open_timestamp'].dt.hour

# Drop 'first_open_timestamp' columns
train_data.drop(['first_open_timestamp'], axis=1, inplace=True)
test_data.drop(['first_open_timestamp'], axis=1, inplace=True)

# Drop 'first_open_date' as it's non-numeric
X = train_data.drop(['ID', 'TARGET', 'first_open_date'], axis=1)
y = train_data['TARGET']

# Align train_data and test_data after removing 'ID' and 'first_open_date'
X, test_data_aligned = X.align(test_data.drop(['ID', 'first_open_date'], axis=1), join='inner', axis=1)

# Split the data into training and validation sets
X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=42)

# Preprocessing pipelines for numerical and categorical data
numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
categorical_cols = [col for col in X.columns if col not in numerical_cols]

# Handle numerical columns using SimpleImputer for missing values
numerical_transformer = SimpleImputer(strategy='mean')

# Handle categorical columns using OneHotEncoder
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# Bundle preprocessing for numerical and categorical data
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_cols),
        ('cat', categorical_transformer, categorical_cols)
    ])

# Preprocess the training and validation data
X_train_preprocessed = preprocessor.fit_transform(X_train)
X_valid_preprocessed = preprocessor.transform(X_valid)

# Convert the preprocessed data into XGBoost DMatrix format
train_data_xgb = xgb.DMatrix(X_train_preprocessed, label=y_train)
valid_data_xgb = xgb.DMatrix(X_valid_preprocessed, label=y_valid)

# Define the XGBoost model parameters
xgb_params = {
    'objective': 'reg:squarederror',  # RMSE for regression
    'learning_rate': 0.005,
    'max_depth': 6,
    'min_child_weight': 10,
    'reg_alpha': 0.1,
    'reg_lambda': 0.1,
    'random_state': 42,
    'eval_metric': 'rmse'
}

# Train the model using XGBoost's native API with early stopping
xgb_model = xgb.train(
    xgb_params,
    train_data_xgb,
    num_boost_round=1500,
    evals=[(train_data_xgb, 'train'), (valid_data_xgb, 'valid')],
    early_stopping_rounds=50
)

# Calculate RMSE on validation set
y_pred_valid_xgb = xgb_model.predict(valid_data_xgb)
rmse_valid_xgb = np.sqrt(mean_squared_error(y_valid, y_pred_valid_xgb))
print(f'Validation RMSE: {rmse_valid_xgb}')

# Preprocess the test data
X_test_preprocessed = preprocessor.transform(test_data_aligned)

# Convert preprocessed test data to DMatrix format for XGBoost
test_data_xgb = xgb.DMatrix(X_test_preprocessed)

# Make predictions on the test data
y_pred_test = xgb_model.predict(test_data_xgb)

# Create the submission file (assuming 'ID' column exists in test data)
submission = pd.DataFrame({
    'ID': users_test['ID'],
    'TARGET': y_pred_test
})

# Save submission file
submission.to_csv("submission.csv", index=False)
print("Submission file created successfully!")

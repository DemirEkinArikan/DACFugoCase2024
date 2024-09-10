import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
import lightgbm as lgb

# Load the datasets
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
train_data['first_open_timestamp'] = pd.to_datetime(train_data['first_open_timestamp'] / 1e9, unit='s', errors='coerce')
test_data['first_open_timestamp'] = pd.to_datetime(test_data['first_open_timestamp'] / 1e9, unit='s', errors='coerce')

# Extract time-based features from 'first_open_timestamp'
train_data['first_open_day'] = train_data['first_open_timestamp'].dt.dayofweek
train_data['first_open_hour'] = train_data['first_open_timestamp'].dt.hour
test_data['first_open_day'] = test_data['first_open_timestamp'].dt.dayofweek
test_data['first_open_hour'] = test_data['first_open_timestamp'].dt.hour

# Drop 'first_open_timestamp' and non-numeric columns
X = train_data.drop(['ID', 'TARGET', 'first_open_timestamp', 'first_open_date'], axis=1)
y = train_data['TARGET']

# Align train_data and test_data after removing 'ID' and 'first_open_date'
X, test_data_aligned = X.align(test_data.drop(['ID', 'first_open_timestamp', 'first_open_date'], axis=1), join='inner', axis=1)

# Defining the numerical and categorical columns
numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
categorical_cols = [col for col in X.columns if col not in numerical_cols]

# Preprocessing pipelines for numerical and categorical data
numerical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())  # Scale numeric features
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))  # One-hot encode categorical features
])

# Combine the transformers into a preprocessor
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_cols),
        ('cat', categorical_transformer, categorical_cols)
    ])

# Apply the preprocessing pipeline to both the train and test data
X_train_processed = preprocessor.fit_transform(X)
test_data_processed = preprocessor.transform(test_data_aligned)

# Split the processed data into training and validation sets
X_train, X_valid, y_train, y_valid = train_test_split(X_train_processed, y, test_size=0.2, random_state=42)

# Convert the processed datasets into LightGBM Dataset format
train_data = lgb.Dataset(X_train, label=y_train)
valid_data = lgb.Dataset(X_valid, label=y_valid, reference=train_data)

# Define the LightGBM model parameters
params = {
    'objective': 'regression',
    'metric': 'rmse',
    'learning_rate': 0.005,
    'num_leaves': 31,
    'max_depth': 6,
    'min_child_samples': 10,
    'reg_alpha': 0.1,
    'reg_lambda': 0.1,
    'random_state': 42
}

# Train the model using the native LightGBM API with early stopping
model = lgb.train(
    params,
    train_data,
    num_boost_round=1000,  # Number of boosting rounds
    valid_sets=[train_data, valid_data],  # Train and validation datasets
    callbacks=[lgb.early_stopping(stopping_rounds=50)],  # Early stopping if no improvement in 50 rounds

)

# Make predictions on the validation set
y_pred_valid = model.predict(X_valid)
rmse_valid = np.sqrt(mean_squared_error(y_valid, y_pred_valid))
print(f'Validation RMSE: {rmse_valid}')

# Make predictions on the test data
test_predictions_lgb = model.predict(test_data_processed)

# Prepare the submission file
submission_lgb = pd.DataFrame({
    'ID': test_data['ID'],
    'TARGET': test_predictions_lgb
})

# Save the submission file
submission_lgb.to_csv('submission_lgb.csv', index=False)


'''
users_train = users_train.head(200)
user_features_train = user_features_train.head(200)
targets_train = targets_train.head(200)
users_test = users_test.head(200)
user_features_test = user_features_test.head(200)
'''

'''
from sklearn.model_selection import cross_val_score

cv_scores = cross_val_score(pipeline_lgb, X_train, y_train, cv=5, scoring='neg_mean_squared_error')
rmse_cv = np.sqrt(-cv_scores)
print(f'Cross-Validated RMSE: {rmse_cv.mean()}')
'''


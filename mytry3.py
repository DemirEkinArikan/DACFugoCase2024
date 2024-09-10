import pandas as pd

users_train = pd.read_csv("C:/Users/Lenovo/OneDrive/Masaüstü/users_train.csv")
user_features_train = pd.read_csv("C:/Users/Lenovo/OneDrive/Masaüstü/user_features_train.csv")
targets_train = pd.read_csv("C:/Users/Lenovo/OneDrive/Masaüstü/targets_train.csv")

users_test = pd.read_csv("C:/Users/Lenovo/OneDrive/Masaüstü/users_test.csv")
user_features_test = pd.read_csv("C:/Users/Lenovo/OneDrive/Masaüstü/user_features_test.csv")

users_train = users_train.head(10000)
user_features_train = user_features_train.head(10000)
targets_train = targets_train.head(10000)
users_test = users_test.head(10000)
user_features_test = user_features_test.head(10000)

users_train.to_csv("first_10000_rows_users_train.csv",index=False)
user_features_train.to_csv("first_10000_rows_user_features_train.csv",index=False)
targets_train.to_csv("first_10000_rows_targets_train.csv",index=False)
users_test.to_csv("first_10000_rows_users_test.csv",index=False)
user_features_test.to_csv("first_10000_rows_user_features_test.csv",index=False)
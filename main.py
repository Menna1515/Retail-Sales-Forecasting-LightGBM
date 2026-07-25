import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt

# 1. Load the dataset (using relative path for portability)
file_path = "retail_sales.csv"  # Update path if placed inside a subfolder
df = pd.read_csv(file_path)

print(f"Dataset successfully loaded! Total rows: {df.shape[0]:,}, Columns: {df.shape[1]}")

# 2. Comprehensive Missing Values & Noise Check
print("--- Missing Values Report ---")
print(df.isnull().sum())

print("--- Masked Nulls Check ---")
for col in df.columns:
    masked_null_count = df[col].astype(str).str.strip().str.lower().isin(['null', 'nan', 'na', '?', '']).sum()
    print(f"Column [{col}]: Contains {masked_null_count} masked null values.")

# 3. Duplicate Rows Check
duplicate_count = df.duplicated().sum()
print(f"Total duplicate rows: {duplicate_count}")

# 4. Data Preprocessing & Feature Engineering
df['date'] = pd.to_datetime(df['date'])

# Extract time features
df['year'] = df['date'].dt.year
df['quarter'] = df['date'].dt.quarter
df['is_weekend'] = df['weekday'].apply(lambda x: 1 if x in [4, 5] else 0)

# Clean IDs and sort chronologically
df['store_id'] = df['store_id'].astype(str).str.replace('store_', '').astype(int)
df['item_id'] = df['item_id'].astype(str).str.replace('item_', '').astype(int)
df = df.sort_values(by=['date', 'store_id', 'item_id']).reset_index(drop=True)

# Create Lag Features
df['sales_lag_1'] = df.groupby(['store_id', 'item_id'])['sales'].shift(1)
df['sales_lag_7'] = df.groupby(['store_id', 'item_id'])['sales'].shift(7)

# Drop missing values resulting from lags
df.dropna(inplace=True)

# Drop unnecessary columns
df.drop(['weekday', 'month'], axis=1, inplace=True, errors='ignore')

# Save cleaned dataset
cleaned_output_path = "retail_sales_cleaned.csv"
df.to_csv(cleaned_output_path, index=False)
print(f"Cleaned dataset saved successfully. Final shape: {df.shape[0]:,} rows.")

# 5. Train / Test Split
split_date = '2023-10-01'
train_df = df[df['date'] < split_date].copy()
test_df = df[df['date'] >= split_date].copy()

target_col = 'sales'
features = [col for col in df.columns if col not in ['date', target_col]]

X_train, y_train = train_df[features], train_df[target_col]
X_test, y_test = test_df[features], test_df[target_col]

# 6. LightGBM Model Training
categorical_features = ['store_id', 'item_id', 'quarter']

train_data = lgb.Dataset(X_train, label=y_train, categorical_feature=categorical_features)
test_data = lgb.Dataset(X_test, label=y_test, reference=train_data, categorical_feature=categorical_features)

params = {
    'objective': 'regression',
    'metric': 'rmse',
    'boosting_type': 'gbdt',
    'learning_rate': 0.05,
    'num_leaves': 31,
    'verbose': -1,
    'n_jobs': -1
}

print("Training the LightGBM model...")
model = lgb.train(
    params,
    train_data,
    num_boost_round=500,
    valid_sets=[train_data, test_data],
    callbacks=[lgb.early_stopping(stopping_rounds=20), lgb.log_evaluation(50)]
)
print("Training completed successfully!")

# 7. Model Evaluation
y_pred = model.predict(X_test)
final_rmse = np.sqrt(mean_squared_error(y_test, y_pred))
final_mae = mean_absolute_error(y_test, y_pred)

print(f"Final RMSE: {final_rmse:.4f}")
print(f"Final MAE: {final_mae:.4f}")

# 8. Feature Importance Plot
lgb.plot_importance(model, max_num_features=10, figsize=(10, 6), importance_type='gain')
plt.title("Feature Importance - LightGBM")
plt.show()

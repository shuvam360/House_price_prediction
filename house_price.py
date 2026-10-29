import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# =========================================================
# Step 1: Load train.csv and test.csv, then print shape and first 5 rows
# =========================================================
train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")

print("--- Step 1: Data Dimensions ---")
print("Train data shape:", train.shape)
print("Test data shape:", test.shape)

print("\n--- First 5 rows of Train data ---")
print(train.head())


# =========================================================
# Step 2: Show basic info (top 10 missing value columns and SalePrice summary)
# =========================================================
print("\n--- Step 2: Missing Values (Top 10 columns) ---")
print(train.isnull().sum().sort_values(ascending=False).head(10))

print("\n--- SalePrice Summary Statistics ---")
print(train["SalePrice"].describe())


# =========================================================
# Step 3: Make 2 simple graphs (Histogram and Heatmap)
# =========================================================
# Graph 1: Histogram of target variable SalePrice
plt.figure(figsize=(8, 5))
sns.histplot(train["SalePrice"], kde=True, color="skyblue")
plt.title("Distribution of SalePrice")
plt.xlabel("Sale Price")
plt.ylabel("Count")
plt.tight_layout()
plt.show()

# Graph 2: Heatmap of top 10 columns most correlated with SalePrice
top10_corr = train.corr(numeric_only=True)["SalePrice"].sort_values(ascending=False).head(10).index
plt.figure(figsize=(10, 8))
sns.heatmap(train[top10_corr].corr(), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Top 10 Features Correlated with SalePrice")
plt.tight_layout()
plt.show()


# =========================================================
# Step 4: Simple Data Cleaning
# =========================================================
# Save the target column and test Ids for submission
y = train["SalePrice"]
test_ids = test["Id"]

# Combine train and test features (dropping Id and SalePrice) so both get identical columns
train_features = train.drop(columns=["Id", "SalePrice"])
test_features = test.drop(columns=["Id"])
combined_data = pd.concat([train_features, test_features], axis=0, ignore_index=True)

# 1. Fill missing numeric values with the median
num_cols = combined_data.select_dtypes(include=[np.number]).columns
for col in num_cols:
    combined_data[col] = combined_data[col].fillna(combined_data[col].median())

# 2. Fill missing text (categorical) values with the mode (most frequent value)
cat_cols = combined_data.select_dtypes(exclude=[np.number]).columns
for col in cat_cols:
    combined_data[col] = combined_data[col].fillna(combined_data[col].mode()[0])

# 3. Convert categorical columns to numeric using one-hot encoding (get_dummies)
combined_data = pd.get_dummies(combined_data, drop_first=True)


# =========================================================
# Step 5: Split train data into Train and Validation sets (80% / 20%)
# =========================================================
# Separate combined data back into training and test sets
X = combined_data.iloc[: len(train)]
X_test = combined_data.iloc[len(train) :]

# Split into 80% train and 20% validation
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

print("\n--- Step 5: Train and Validation Split ---")
print(f"X_train shape: {X_train.shape}, X_val shape: {X_val.shape}")


# =========================================================
# Step 6: Train a LinearRegression model
# =========================================================
model = LinearRegression()
model.fit(X_train, y_train)
print("\n--- Step 6: Linear Regression Model Trained Successfully ---")


# =========================================================
# Step 7: Check model on validation data (R2 Score and RMSE)
# =========================================================
y_val_pred = model.predict(X_val)

r2 = r2_score(y_val, y_val_pred)
rmse = np.sqrt(mean_squared_error(y_val, y_val_pred))

print("\n--- Step 7: Validation Metrics ---")
print(f"Validation R2 Score : {r2:.4f}")
print(f"Validation RMSE     : {rmse:.2f}")


# =========================================================
# Step 8: Scatter plot of Actual vs Predicted prices
# =========================================================
plt.figure(figsize=(8, 6))
plt.scatter(y_val, y_val_pred, alpha=0.5, color="teal")
plt.plot([y_val.min(), y_val.max()], [y_val.min(), y_val.max()], "r--", lw=2)  # Ideal line
plt.title("Actual vs Predicted House Prices")
plt.xlabel("Actual SalePrice")
plt.ylabel("Predicted SalePrice")
plt.tight_layout()
plt.show()


# =========================================================
# Step 9: Predict on test.csv and save as submission.csv
# =========================================================
test_predictions = model.predict(X_test)

submission = pd.DataFrame({
    "Id": test_ids,
    "SalePrice": test_predictions
})

submission.to_csv("submission.csv", index=False)
print("\n--- Step 9: submission.csv Saved Successfully ---")
print("First 5 rows of submission:")
print(submission.head())


# =========================================================
# Step 10: 3-line simple conclusion
# =========================================================
print("\n--- Step 10: Conclusion ---")
print("1. Missing values were cleaned and categorical features were converted using one-hot encoding.")
print(f"2. A Linear Regression model was fitted, achieving an R2 score of {r2:.4f} and RMSE of {rmse:.2f} on the validation data.")
print("3. Predictions for the test set were generated and exported to submission.csv in the required format.")
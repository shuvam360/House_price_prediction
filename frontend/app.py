import os
import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# =========================================================
# Step 1: Cache and train the Linear Regression model
# =========================================================
@st.cache_resource
def load_and_train_model():
    # Use relative path to train.csv in the parent directory
    train_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "train.csv")
    
    # Check if train.csv exists
    if not os.path.exists(train_path):
        return None, None
    
    # Load the training dataset
    df = pd.read_csv(train_path)
    
    # Use the 6 chosen numeric features and target column
    features = ['OverallQual', 'GrLivArea', 'TotalBsmtSF', 'GarageCars', 'FullBath', 'YearBuilt']
    target = 'SalePrice'
    
    X = df[features].copy()
    y = df[target]
    
    # Fill missing values with column median
    X = X.fillna(X.median())
    
    # Split into 80% train and 20% validation sets
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train the Linear Regression model
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    # Calculate validation R2 score
    val_preds = model.predict(X_val)
    val_r2 = r2_score(y_val, val_preds)
    
    return model, val_r2


# =========================================================
# Step 2: Load the model and check for errors
# =========================================================
model, val_r2 = load_and_train_model()

if model is None:
    st.error("Error: 'train.csv' was not found in the parent folder. Please make sure the file exists.")
    st.stop()


# =========================================================
# Step 3: Page title and simple description
# =========================================================
st.title("House Price Predictor")
st.write("Enter house details below to estimate its market sale price.")


# =========================================================
# Step 4: Input form in 2 columns
# =========================================================
col1, col2 = st.columns(2)

with col1:
    # Overall Quality slider (1 to 10)
    overall_qual = st.slider("Overall Quality (1-10)", min_value=1, max_value=10, value=6)
    
    # Above ground living area in square feet
    gr_liv_area = st.number_input("Living Area (sq ft)", min_value=300, max_value=10000, value=1500, step=50)
    
    # Total basement area in square feet
    total_bsmt_sf = st.number_input("Basement Area (sq ft)", min_value=0, max_value=5000, value=1000, step=50)

with col2:
    # Garage capacity in number of cars (0 to 4)
    garage_cars = st.slider("Garage Capacity (cars)", min_value=0, max_value=4, value=2)
    
    # Full bathrooms count (0 to 4)
    full_bath = st.slider("Full Bathrooms", min_value=0, max_value=4, value=2)
    
    # Construction year (1870 to 2025)
    year_built = st.number_input("Year Built", min_value=1870, max_value=2025, value=2000, step=1)


# =========================================================
# Step 5: Predict button and price output
# =========================================================
if st.button("Predict price"):
    # Create input DataFrame with the 6 feature names matching training data
    user_input = pd.DataFrame([[overall_qual, gr_liv_area, total_bsmt_sf, garage_cars, full_bath, year_built]],
                              columns=['OverallQual', 'GrLivArea', 'TotalBsmtSF', 'GarageCars', 'FullBath', 'YearBuilt'])
    
    # Predict price using trained LinearRegression model
    raw_price = model.predict(user_input)[0]
    
    # Ensure price is not negative
    final_price = max(0, raw_price)
    
    # Display formatted price in a green success box
    st.success(f"Estimated Price: ${final_price:,.0f}")


# =========================================================
# Step 6: Caption showing model details and validation score
# =========================================================
st.caption(f"Model: LinearRegression | Validation R² Score: {val_r2:.4f}")

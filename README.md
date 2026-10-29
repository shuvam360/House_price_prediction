# 🏡 House Price Prediction System

An end-to-end Machine Learning web application that predicts residential house prices based on property characteristics using Multiple Linear Regression trained on the Ames Housing Dataset.

---

## 🌟 Architecture Overview

The project is structured into three clean layers:
1. **Machine Learning Model**: Multiple Linear Regression trained on key housing attributes (`OverallQual`, `GrLivArea`, `TotalBsmtSF`, `GarageCars`, `FullBath`, `YearBuilt`), achieving an $R^2$ score of `0.7944` on validation.
2. **Backend API (`backend/`)**: Python Flask REST API running with Gunicorn, ready for instant deployment to [Render](https://render.com).
3. **Frontend Web App (`index.html`, `style.css`, `app.js`)**: Modern, responsive interface deployable via [GitHub Pages](https://pages.github.com) or [Vercel](https://vercel.com).

---

## 📂 Project Structure

```text
House_price_prediction/
├── backend/
│   ├── app.py                     # Flask REST API endpoints (/api/predict, /api/health, /api/metrics)
│   ├── train.py                   # Script to train and serialize the model
│   ├── test_api.py                # Automated API unit tests
│   ├── requirements.txt           # Production dependencies for Render
│   ├── Procfile                   # Gunicorn process definition
│   └── models/
│       ├── house_price_model.joblib # Serialized scikit-learn model
│       └── model_metrics.json       # Model weights, intercept, and validation scores
├── frontend/
│   ├── index.html                 # Frontend user interface
│   ├── style.css                  # Modern UI styles
│   ├── app.js                     # Frontend API client with fallback calculation
│   └── app.py                     # Alternative Streamlit web interface
├── index.html                     # Root entry point for GitHub Pages deployment
├── style.css                      # Root stylesheet
├── app.js                         # Root frontend script
├── house_price.py                 # Full Kaggle training and submission pipeline
├── server.py                      # Local lightweight Python HTTP server
├── train.csv                      # Training dataset
├── test.csv                       # Kaggle test dataset
├── submission.csv                 # Generated predictions
└── README.md
```

---

## 🚀 Running Locally

### 1. Run the Backend API
```bash
cd backend
pip install -r requirements.txt
python app.py
```
The API starts at `http://127.0.0.1:5000`.

### 2. Run the Frontend
You can simply open `index.html` in your browser, or start a local server:
```bash
python server.py
```
Open `http://localhost:8000/index.html`.

---

## ☁️ Deployment Instructions

### A. Deploy Backend to Render

1. Log in to your [Render Dashboard](https://dashboard.render.com).
2. Click **New +** $\rightarrow$ **Web Service**.
3. Select your GitHub repository: `shuvam360/House_price_prediction`.
4. Configure the Web Service:
   * **Name**: `house-price-backend` (or your choice)
   * **Root Directory**: `backend`
   * **Runtime**: `Python 3`
   * **Build Command**: `pip install -r requirements.txt`
   * **Start Command**: `gunicorn app:app`
   * **Instance Type**: `Free`
5. Click **Deploy Web Service**.
6. When deployment finishes, copy your Render URL (e.g. `https://house-price-backend.onrender.com`).

---

### B. Connect Frontend to Render & Deploy to GitHub Pages

1. In `app.js` (and `frontend/app.js`), set your Render URL on line 12:
   ```javascript
   const RENDER_BACKEND_URL = "https://house-price-prediction-2h2l.onrender.com";
   ```
2. Commit and push the change to GitHub:
   ```bash
   git add app.js frontend/app.js
   git commit -m "Configure live Render backend URL"
   git push origin main
   ```
3. Enable **GitHub Pages**:
   * On GitHub, navigate to **Settings** $\rightarrow$ **Pages**.
   * Under **Branch**, select `main` and `/ (root)`.
   * Click **Save**.
   * Your frontend will be live at `https://shuvam360.github.io/House_price_prediction/` within a minute!

---

## 📊 API Reference

### `GET /api/health`
Checks API status and confirms the model is loaded.

### `POST /api/predict`
Calculates predicted house price.

**Request Body:**
```json
{
  "OverallQual": 6,
  "GrLivArea": 1500,
  "TotalBsmtSF": 1000,
  "GarageCars": 2,
  "FullBath": 2,
  "YearBuilt": 2000
}
```

**Response:**
```json
{
  "status": "success",
  "predicted_price": 187254,
  "formatted_price": "$187,254",
  "currency": "USD",
  "model": "LinearRegression",
  "r2_score": 0.7944
}
```

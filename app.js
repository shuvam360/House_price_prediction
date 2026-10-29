/* ========================================================
   House Price Predictor - JavaScript Engine
   Connected to Flask / Render Backend API
   Model: Linear Regression (Trained on Kaggle Ames Dataset)
   Features: OverallQual, GrLivArea, TotalBsmtSF, GarageCars, FullBath, YearBuilt
   Validation R2 Score: 0.7944
   ======================================================== */

// Backend API URL:
// 1. If running locally on localhost/127.0.0.1, defaults to http://127.0.0.1:5000
// 2. Once deployed on Render, put your Render URL below (e.g. "https://house-price-backend.onrender.com")
const RENDER_BACKEND_URL = "https://house-price-prediction-2h2l.onrender.com"; 

const API_BASE_URL = RENDER_BACKEND_URL || (
  window.location.hostname === "localhost" || window.location.hostname === "127.0.0.1"
    ? "http://127.0.0.1:5000"
    : ""
);

// Fallback Linear Regression weights (ensures app works offline or while Render wakes up)
const MODEL_WEIGHTS = {
  intercept: -759894.3244,
  OverallQual: 20507.2013,
  GrLivArea: 51.9934,
  TotalBsmtSF: 24.4754,
  GarageCars: 15329.4085,
  FullBath: -5421.2103,
  YearBuilt: 350.9117
};

function calculateClientSidePrice(qual, grLiv, bsmt, garage, bath, year) {
  let rawPrice = MODEL_WEIGHTS.intercept
    + (MODEL_WEIGHTS.OverallQual * qual)
    + (MODEL_WEIGHTS.GrLivArea * grLiv)
    + (MODEL_WEIGHTS.TotalBsmtSF * bsmt)
    + (MODEL_WEIGHTS.GarageCars * garage)
    + (MODEL_WEIGHTS.FullBath * bath)
    + (MODEL_WEIGHTS.YearBuilt * year);
  return Math.max(0, Math.round(rawPrice));
}

document.addEventListener("DOMContentLoaded", () => {
  const qualInput = document.getElementById("overall_qual");
  const qualBadge = document.getElementById("badge_qual");

  const garageInput = document.getElementById("garage_cars");
  const garageBadge = document.getElementById("badge_garage");

  const bathInput = document.getElementById("full_bath");
  const bathBadge = document.getElementById("badge_bath");

  const predictBtn = document.getElementById("btn_predict");
  const resultBox = document.getElementById("result_box");
  const resultPrice = document.getElementById("result_price");

  // Sync sliders with their display badges
  if (qualInput && qualBadge) {
    qualInput.addEventListener("input", () => {
      qualBadge.textContent = `${qualInput.value} / 10`;
    });
  }

  if (garageInput && garageBadge) {
    garageInput.addEventListener("input", () => {
      garageBadge.textContent = `${garageInput.value} cars`;
    });
  }

  if (bathInput && bathBadge) {
    bathInput.addEventListener("input", () => {
      bathBadge.textContent = `${bathInput.value} full baths`;
    });
  }

  // Calculate prediction on button click
  predictBtn.addEventListener("click", async () => {
    const qual = parseFloat(qualInput.value) || 0;
    const grLiv = parseFloat(document.getElementById("gr_liv_area").value) || 0;
    const bsmt = parseFloat(document.getElementById("total_bsmt_sf").value) || 0;
    const garage = parseFloat(garageInput.value) || 0;
    const bath = parseFloat(bathInput.value) || 0;
    const year = parseFloat(document.getElementById("year_built").value) || 0;

    const payload = {
      OverallQual: qual,
      GrLivArea: grLiv,
      TotalBsmtSF: bsmt,
      GarageCars: garage,
      FullBath: bath,
      YearBuilt: year
    };

    // UI Loading state
    const originalText = predictBtn.innerHTML;
    predictBtn.disabled = true;
    predictBtn.innerHTML = `<span>⏳</span> Calculating...`;

    let finalPrice = null;

    // Attempt to query Backend API if available
    if (API_BASE_URL) {
      try {
        const controller = new AbortController();
        const timeoutId = setTimeout(() => controller.abort(), 6000); // 6s timeout

        const res = await fetch(`${API_BASE_URL}/api/predict`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(payload),
          signal: controller.signal
        });
        clearTimeout(timeoutId);

        if (res.ok) {
          const data = await res.json();
          if (data && data.predicted_price !== undefined) {
            finalPrice = data.predicted_price;
          }
        }
      } catch (err) {
        console.warn("Backend API call failed, falling back to client-side model calculation:", err);
      }
    }

    // Fallback to client-side calculation if API is unavailable or offline
    if (finalPrice === null) {
      finalPrice = calculateClientSidePrice(qual, grLiv, bsmt, garage, bath, year);
    }

    // Restore button
    predictBtn.disabled = false;
    predictBtn.innerHTML = originalText;

    // Display formatted price
    resultPrice.textContent = `$${finalPrice.toLocaleString()}`;
    resultBox.classList.add("show");

    // Smooth scroll down to result
    resultBox.scrollIntoView({ behavior: "smooth", block: "nearest" });
  });
});

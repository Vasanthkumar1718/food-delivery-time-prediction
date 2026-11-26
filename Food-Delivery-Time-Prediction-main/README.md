# Food Delivery Time Prediction

Predict how many **minutes** a food delivery will take, given the order details —
who is delivering it, how far it has to travel, the weather, traffic, time of day,
and more. Framed as a **supervised regression** problem and solved end-to-end, from
raw messy data to a saved model ready for inference.

---

## Results

Four models were trained and compared. XGBoost (with tuned hyperparameters) came out on top.

| Model | Test R² | Test RMSE (min) | Train R² | Notes |
|---|---|---|---|---|
| Linear Regression | 0.57 | 6.21 | 0.58 | Baseline — relationships aren't linear |
| Decision Tree | 0.67 | 5.46 | 1.00 | Overfits (memorises the training set) |
| Random Forest | 0.82 | 4.03 | 0.97 | Big gain — averaging tames overfitting |
| **XGBoost (tuned)** | **0.83** | **3.91** | 0.85 | Best score *and* healthiest train/test gap |

*RMSE is in minutes, so the final model is off by roughly 4 minutes on average.*

---

## Approach

1. **Load & inspect** the raw data
2. **Clean** messy text and missing values (e.g. `"conditions Sunny"` → `Sunny`, `"(min) 24"` → `24`)
3. **EDA** — find which features actually move delivery time
4. **Feature engineering**
   - `distance_km` from restaurant/customer coordinates via the **Haversine formula**
   - `order_hour` extracted from the order timestamp
5. **Encode** categoricals (ordinal for traffic density, one-hot for the rest) and **scale** features
6. **Train & compare** Linear Regression, Decision Tree, Random Forest, XGBoost
7. **Tune** the best model (RandomizedSearchCV), **evaluate** (RMSE, R², Adjusted R²), and **save** it for inference

The maths and reasoning behind each step are explained inline in the notebook.

---

## Dataset

The dataset (`45,593` orders across `20` features) contains historical food-delivery orders with details such as the delivery person's age and rating, restaurant and delivery coordinates, weather conditions, road-traffic density, vehicle type, order type, festival flag, and city type. The target variable is `Time_taken(min)`.

---

## Tech Stack

Python · pandas · NumPy · Matplotlib · seaborn · scikit-learn · XGBoost · joblib

---

## How to Run

```bash
# 1. Clone the repository
git clone [https://github.com/Vasanthkumar1718/food-delivery-time-prediction.git](https://github.com/Vasanthkumar1718/food-delivery-time-prediction.git)
cd food-delivery-time-prediction

# 2. Create and activate a virtual environment
python -m venv .venv
.venv\Scripts\activate          # macOS/Linux: source .venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Launch the notebook
jupyter notebook Food_Delivery_Time_Prediction.ipynb

Repository Structure

food-delivery-time-prediction/
├── README.md
├── LICENSE
├── requirements.txt
├── .gitignore
├── Food_Delivery_Time_Prediction.ipynb   # Full documented EDA & ML pipeline
├── food_delivery.csv                      # Raw dataset (45,593 records)
├── xgb_best.pkl                           # Saved tuned XGBoost model
└── standard_scaler.pkl                    # Saved feature scaler

# 🌡️ Monthly Mean Temperature Time Series Analysis  

## 📌 Project Overview  
This project analyzes **monthly mean temperature data (1920–1939)** using classical **time series models** (ARMA, ARIMA).  
The objective is to **forecast temperature trends**, assess **model accuracy**, and establish a baseline for more advanced forecasting techniques.  

---

## 📂 Dataset  
- **Source File:** `monthly-mean-temp.csv`  
- **Time Period:** Jan 1920 – Dec 1939 (240 months)  
- **Features:**  
  - `Month` → Datetime index  
  - `Temperature` → Monthly mean temperature (°F)  

---

## ⚙️ Methodology  

1. **Data Preprocessing**  
   - Converted `Month` into a datetime index.  
   - Verified there were **no missing values**.  
   - Generated summary statistics for exploratory insights.  

2. **Stationarity Check**  
   - **ADF Test:** p-value < 0.05 → data is stationary.  
   - **ACF Plot:** confirmed stationary behavior.  

3. **Model Development**  
   - **AR (AutoReg):** Lags selected → [1, 4, 11, 13].  
   - **ARIMA:** Exhaustive grid search across `(p,d,q)` values.  
   - **Best Model:** ARIMA(2,0,3) with lowest AIC = **675.65**.  

4. **Forecasting & Evaluation**  
   - Forecasted **25 months** ahead.  
   - Compared predicted vs actual values.  
   - Evaluated performance with error metrics.  

---

## 📊 Results & Insights  

| Metric | Value |
|--------|-------|
| **RMSE** | 2.84 |
| **MAE**  | 2.36 |
| **MSE**  | 8.09 |

📌 **Key Insights:**  
- Model achieved **good accuracy** with relatively low errors.  
- **RMSE > MAE** → means model is good.
- ARIMA captured **short-term patterns**  

---

## 📈 Visuals (Examples)  

- 📊 **Temperature Trend (1920–1939)**  
- 🔁 **ACF/PACF plots for lag analysis**  
- 🔮 **Forecast vs Actual for 25 months**  


---

## 🛠️ Tech Stack  

- **Python**  
- **pandas, numpy** → data processing  
- **matplotlib, seaborn** → visualization  
- **statsmodels** →  ARMA, ARIMA models  
- **scikit-learn** → evaluation metrics  

---

   --------------  Thank you    -----------------


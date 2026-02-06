# 🏙️ Milan Housing Dashboard

Interactive Streamlit dashboard for exploring Milan housing listings, visualizing spatial patterns, and explaining an XGBoost regression model with SHAP.

---

## 🚀 Live Demo
(After deployment, your Streamlit link will go here)

---

## 📊 Features

- Interactive filters (Area, Bedrooms, Energy Score, Transport)
- Dynamic dataset preview
- Folium map visualization (exported HTML)
- XGBoost regression model
- Model evaluation (RMSE, MAE, R²)
- SHAP explainability (Summary bar, Beeswarm, Dependence plot)
- Download filtered dataset (CSV)

---

## 🧠 Model

- Algorithm: XGBoost Regressor
- Train/Test split: 80/20
- Target: `ln_price`
- Explainability: SHAP (TreeExplainer)

---

## 🛠 Tech Stack

- Streamlit
- Pandas
- XGBoost
- SHAP
- Folium
- Scikit-learn
- Matplotlib

---

## 📁 Project Structure

app.py
data_final.xlsx
milan_area_map_lnprice.html
model_metrics.json
requirements.txt

---

## 🎯 Purpose

This project was built as a portfolio-ready data science dashboard demonstrating:

- Data analysis
- Machine learning modeling
- Model explainability
- Dashboard development
- Deployment-ready architecture

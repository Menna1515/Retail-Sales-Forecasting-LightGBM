# Retail Sales Forecasting using LightGBM

An advanced machine learning project focused on predicting retail store sales using **LightGBM** and comprehensive feature engineering.

## 🚀 Project Overview
This repository contains an end-to-end data science pipeline designed to clean, process, and forecast retail demand across multiple stores and items. 

## 🛠️ Key Steps & Features
1. **Data Preprocessing & Noise Reduction:** Comprehensive checks for missing values, masked nulls (`NaN`, `?`, empty spaces), and duplicate records.
2. **Feature Engineering:** 
   - Extracted temporal features (Year, Quarter, Weekend indicators).
   - Created powerful lag features (`sales_lag_1`, `sales_lag_7`) grouped by store and item.
3. **Model Training:** Utilized **LightGBM** optimized for large-scale regression tasks, leveraging categorical feature handling and early stopping.
4. **Evaluation:** Measured performance using **RMSE** and **MAE** metrics.

## 📊 Tech Stack
* **Python**
* **Pandas & NumPy** (Data Manipulation)
* **LightGBM** (Machine Learning Model)
* **Scikit-Learn** (Evaluation Metrics)
* **Matplotlib** (Data Visualization)

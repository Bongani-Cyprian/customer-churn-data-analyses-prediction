#  Churn Prediction App

[![Python](https://img.shields.io/badge/python-3.9%2B-blue)](https://www.python.org/)  
[![Streamlit Community](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://share.streamlit.io/)  
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](./LICENSE)

##  Overview  
A Streamlit dashboard that predicts customer churn using an XGBoost model.

## Features  
- Adjustable probability threshold  
- Plotly-powered feature importance  
- Upload your own CSV & download scored output  

## Tech Stack  
Python 3.9+, Streamlit, XGBoost, joblib, Pandas, Plotly

## Getting Started

**Clone & install**  
```bash
git clone https://github.com/YourUsername/churn-prediction-app.git
cd churn-prediction-app
python -m venv .venv
.\.venv\Scripts\activate    # Windows
pip install -r requirements.txt

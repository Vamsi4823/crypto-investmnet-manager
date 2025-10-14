# 🪙 Crypto Portfolio Investment Manager

An automated tool built to **analyze, predict, and manage cryptocurrency investments** efficiently.  
This project was developed as part of an **internship** to demonstrate real-time data handling, predictive modeling, and portfolio management using **Python** and **Streamlit**.

---

## 🚀 Features

- 📊 **Real-Time Crypto Data Fetching** – Using `yfinance` to fetch live cryptocurrency prices.  
- 📈 **Portfolio Performance Tracking** – Calculate and visualize both individual and total portfolio returns.  
- ⚙️ **Momentum-Based Strategy** – Allocate investment weights based on recent performance trends.  
- 🧮 **Predictive Modeling** – Apply **Simple Linear Regression** to forecast crypto returns.  
- 📉 **Model Evaluation** – Measure accuracy using **Mean Squared Error (MSE)** and **R-squared (R²)**.  
- 💌 **Email Alerts** – Automated notifications for rule or risk violations.  
- 🌪️ **Volatile Market Simulation** – Perform stress tests under high volatility.  
- 🧠 **Interactive Dashboard** – Built using **Streamlit** for local execution and visualization.

---

## 🧰 Tech Stack

- **Programming Language:** Python  
- **Libraries:** yfinance, pandas, numpy, matplotlib, scikit-learn, streamlit, sqlite3, smtplib  
- **Database:** SQLite  
- **IDE:** Visual Studio Code  
- **Deployment:** Localhost (Streamlit App)

---

## 📂 Project Structure

📦 Crypto-Portfolio-Investment-Manager
┣ 📁 data/ # Stores downloaded crypto data
┣ 📁 models/ # Stores regression models (optional)
┣ 📁 database/ # SQLite database file
┣ 📜 app.py # Main Streamlit dashboard
┣ 📜 portfolio_analysis.py # Functions for return & metric calculations
┣ 📜 requirements.txt # Dependencies
┗ 📜 README.md # Project documentation

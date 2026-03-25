#  Tesla Stock Price Forecaster: Deep Learning with LSTM

##  Project Overview

This project tackles the complex challenge of **time-series financial forecasting** by developing a robust deep learning pipeline to predict **Tesla Inc. (TSLA)** stock prices. Given Tesla’s historical market volatility and extreme price breakouts, traditional predictive models often fail to capture sudden momentum shifts. 

The primary objective of this project was to architect, optimize, and deploy a neural network capable of accurately forecasting **1-day, 5-day, and 10-day future closing prices** to aid in algorithmic trading strategies, liquidity planning, and automated risk management.

##  Key Features

* **Advanced Deep Learning Architecture:** Utilizes a Long Short-Term Memory (LSTM) network, specifically chosen to overcome the vanishing gradient problem found in traditional RNNs, allowing the model to capture long-term macro-momentum.
* **Rigorous Data Preprocessing:** Engineered a 60-day sliding window mechanism for sequential data ingestion and applied `MinMaxScaler` for stable model convergence.
* **Strict Temporal Validation:** Hyperparameter tuning was conducted using `GridSearchCV` integrated with `TimeSeriesSplit` to strictly prevent temporal data leakage (look-ahead bias).
* **Interactive Web Deployment:** The fully optimized model is deployed as a live, interactive dashboard using **Streamlit**, allowing users to generate custom forecasting horizons on demand.

##  Technology Stack

* **Language:** Python
* **Deep Learning:** TensorFlow, Keras
* **Data Manipulation:** Pandas, NumPy, Scikit-Learn
* **Visualization:** Matplotlib
* **Deployment Interface:** Streamlit

## Model Architecture & Performance

A comparative architectural analysis was conducted between a baseline SimpleRNN and the final LSTM. The LSTM successfully retained critical historical context across the 60-day sequences, filtering out daily market noise while seamlessly capturing major structural breakouts.

**Optimized LSTM Hyperparameters:**
* **Layers:** 2 Stacked LSTM Layers (50 units each)
* **Regularization:** 20% Dropout (to explicitly prevent overfitting)
* **Optimizer:** Adam (Learning Rate: 0.01)

**Evaluation Metric: Mean Squared Error (MSE)**
MSE was deliberately chosen over MAE because its quadratic nature heavily penalizes large predictive deviations, actively forcing the model to prioritize capital preservation and baseline stability.
* **Final Model MSE:** `0.0005`

```bash
git clone [https://github.com/your-username/tesla-stock-lstm-forecaster.git](https://github.com/your-username/tesla-stock-lstm-forecaster.git)
cd tesla-stock-lstm-forecaster

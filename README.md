# CS4375_Term_Project
<div align="center">

# University of Texas at Dallas  
**Computer Science Department**

**Jeffrey Chou · Josetta Reyes · Anish Saravanan · Bright Gao**  
**CS 4375.5U1 — Introduction to Machine Learning — Su25**  
**Submission Date: August 8, 2025**

</div>

# LSTM From Scratch: Predicting Time Series Stock Prices

## Overview
This project implements a **from-scratch LSTM in NumPy** (no deep-learning libraries) to forecast the **next day’s closing price** of AAPL from recent history.  
It builds a sliding-window dataset, trains a NumPy LSTM + small MLP head, and evaluates on **time-ordered** Train/Validation/Test splits.

---

## Features
- ✅ Pure **NumPy** implementation of LSTM (gates + BPTT) and a small Dense head  
- ✅ **Chronological split** (no leakage): 80% Train, 10% Val, 10% Test  
- ✅ **Experiment log** appended to `experiment_log.csv` (RMSE, R²)  
- ✅ **Matplotlib plots** for Train / Val / Test predictions  
- ✅ Clean code with clearly marked places to change **date window** and **lookback length n**

---

## Prerequisites
- **Python** 3.9+  
- Python packages:
  ```bash
  pip install numpy pandas matplotlib scikit-learn

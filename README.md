# CS4375_Term_Project
<div align="center">
   
# The University of Texas at Dallas  
**Computer Science Department**

**Jeffrey Chou · Josetta Reyes · Anish Saravanan**  
**CS 4375.5U1 — Introduction to Machine Learning — Su25**  
**Submission Date: August 8, 2025**

# LSTM From Scratch: Predicting Time Series Stock Prices
</div>

## Overview
This project implements a **from-scratch Long Short-Term Memory (LSTM)** network in **NumPy** to predict the **next day’s closing price** of Apple Inc. (AAPL) stock using historical daily prices.  
The implementation does not use deep learning libraries such as TensorFlow or PyTorch. All gating equations, backpropagation through time, and parameter updates are coded manually.  
This version includes:
- Stable gradient clipping to avoid exploding gradients
- Configurable training parameters
- Automatic figure saving to `figs/` for Train, Validation, and Test predictions
- Detailed experiment logging to `experiment_log.csv`

---

## Features
- ✅ Pure **NumPy** LSTM cell and dense head  
- ✅ Sequential sliding-window dataset generation  
- ✅ Chronological Train / Validation / Test split (80% / 10% / 10%)  
- ✅ Experiment logging with RMSE and R² metrics  
- ✅ Auto-generated plots saved to `figs/`

---

## Prerequisites
- **Python** 3.9+
- Install dependencies:
  ```bash
  pip install numpy pandas matplotlib scikit-learn
  ```

---
  ## How to Run

1. Place your dataset (e.g., `AAPL.csv`) in the `data/` folder or use the online source in the script.  
2. Open a terminal in the project folder.  
3. Run:
   ```bash
   python "lstm2.py"
   ```
4. The script will:
   - Load and preprocess the dataset
   - Create sliding-window samples for LSTM
   - Train the LSTM model
   - Save figures to `figs/`
   - Append metrics to `experiment_log.csv`

---

## Parameters

Edit these in `lstm2.py` to adjust experiments:

| Parameter        | Description                          | Example Value |
|------------------|--------------------------------------|---------------|
| `first_date_str` | Start date for dataset               | `"2015-03-25"`|
| `last_date_str`  | End date for dataset                 | `"2019-03-25"`|
| `n`              | Lookback window length               | `3`           |
| `hidden`         | Number of LSTM hidden units          | `64`          |
| `epochs`         | Number of training epochs            | `100`         |
| `lr`             | Learning rate                        | `0.001`       |
| `clip`           | Gradient clipping value              | `1.0`         |

---

## Outputs

| Output File / Folder   | Description |
|------------------------|-------------|
| `experiment_log.csv`   | CSV log of experiments (RMSE, R², parameters) |
| `figs/train.png`       | Predictions vs actual for training set |
| `figs/val.png`         | Predictions vs actual for validation set |
| `figs/test.png`        | Predictions vs actual for test set |

Example experiment log entry:
```
1, epochs=100, lr=0.001, lookback n=3, hidden=64, Train RMSE=6.62, Test RMSE=0.22, Train R²=0.91, Test R²=0.22
```

---

## Project Structure

```
.
├── lstm2.py               # Updated NumPy LSTM implementation
├── experiment_log.csv     # Experiment results
├── figs/                  # Saved prediction plots
├── data/                  # Dataset CSV files
└── README.md              # Project documentation
```

---

## Troubleshooting

- **R² is low or negative** → Increase `epochs`, adjust `lr`, or try a larger `hidden` size  
- **NaN values in metrics** → Lower `lr` or check for missing values in dataset  
- **No plots generated** → Ensure `figs/` folder exists or let script create it automatically  

---

## References

- Apple Inc. (AAPL) Daily Closing Price Dataset — Public GitHub source  
- Hochreiter, S., Schmidhuber, J., *Long Short-Term Memory*, Neural Computation, 1997  
- Goodfellow, I., Bengio, Y., Courville, A., *Deep Learning*, MIT Press, 2016

---

## Important Note
All code in this project, including the LSTM implementation, was written from scratch using NumPy. No deep learning frameworks were used.

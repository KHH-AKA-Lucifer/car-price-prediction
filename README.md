# Car Price Prediction

This project predicts the selling price of cars from the Chaky Company based on features like year, kilometers driven, engine size, mileage, max power, seats, fuel type, seller type, transmission, and car name.

The workflow:
1. Clean the dataset (remove outliers, handle missing values).
2. Engineer features (extract numbers, one-hot encode categories).
3. Train models with log-transformed target prices.
4. Save the best model (Random Forest pipeline).
5. Serve predictions through a simple web app built with [Dash](https://dash.plotly.com/).

---

## Run locally (without Docker)

Make sure you have Python 3.10 installed.

```bash
cd app
pip install -r requirements.txt
python code/app.py
```

Open your browser at: [http://localhost:8050](http://localhost:8050)

You can create your own environment. 

---

## 🐳 Run with Docker (recommended)

Build and start the app:

```bash
cd app
docker compose up --build
```

Then go to: [http://localhost:8050](http://localhost:8050)

---

## Project structure

```
car-price-prediction/
│
├─ app/
│   ├─ artifacts/          # saved model (joblib)
│   ├─ code/               # Dash app code (app.py)
│   ├─ requirements.txt    # Python dependencies
│   ├─ Dockerfile          # Docker build instructions
│   └─ docker-compose.yml  # Run app with Docker
│
├─ data/                   # (if any raw data)
├─ model/                  # (not used in container, training artifacts)
└─ car_price_prediction.ipynb  # Jupyter notebook (data cleaning + training)
```

---

## Notes

- The model is a Random Forest Regressor wrapped in a scikit-learn Pipeline.  
- It handles categorical features with `OneHotEncoder(handle_unknown="ignore")`.  
- Prices are predicted on the original scale (target was log-transformed during training).  
- Missing values are filled with median (numeric) or most frequent (categorical).  
- You can try any car name; unseen names are simply ignored by the encoder.

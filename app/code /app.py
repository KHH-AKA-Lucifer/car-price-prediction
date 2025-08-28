# app/code/app.py
# import pickle
import joblib
import pandas as pd
from pathlib import Path
from dash import Dash, html, dcc, Input, Output, State

ART = Path(__file__).resolve().parents[1] / "artifacts"
print(ART)

def load_model():
    """Load either a GridSearchCV or a fitted estimator/pipeline."""
    # model_path_pickle = ART / "car-price-prediction.model"
    model_path_joblib = ART / "car-price-prediction.joblib"
    obj = None
    # if model_path_pickle.exists():
    #     with open(model_path_pickle, "rb") as f:
    #         obj = pickle.load(f)
    if model_path_joblib.exists():
        obj = joblib.load(model_path_joblib)
    else:
        raise FileNotFoundError(
            "Model not found. Expecting 'car-price-prediction.joblib' or 'car-price-prediction.model' in app/artifacts/"
        )
    # If this is a GridSearchCV, use best_estimator_
    return getattr(obj, "best_estimator_", obj)

MODEL = load_model()

# If your model was trained on log(y) and predicts log-prices, set this True.
# If you trained directly on raw selling_price (your current GridSearch code), keep False.
USE_LOG_TARGET = False

# Categorical choices (dataset cleaned to Petrol/Diesel per the assignment)
FUEL_OPTS   = ["Petrol", "Diesel"]
SELLER_OPTS = ["Dealer", "Individual", "Trustmark Dealer"]
TRANS_OPTS  = ["Manual", "Automatic"]

app = Dash(__name__)
app.title = "Car Price Predictor"

def build_row(values: dict) -> pd.DataFrame:
    """
    Build a single-row DataFrame that matches your training schema.
    IMPORTANT:
      - If your saved MODEL is a Pipeline (with OneHotEncoder on 'name', etc.), just pass raw fields.
      - If your saved MODEL is a bare RandomForest trained on pre-encoded columns, you MUST
        replicate the exact encoding here before calling predict.
    """
    row = {
        'name'        : values.get('name') or None,      # free text; OHE(handle_unknown='ignore') will skip unseen names
        'year'        : values.get('year'),
        'km_driven'   : values.get('km'),
        'owner'       : values.get('owner'),
        'mileage'     : values.get('mileage'),
        'engine'      : values.get('engine'),
        'max_power'   : values.get('power'),
        'seats'       : values.get('seats'),
        'fuel'        : values.get('fuel'),
        'seller_type' : values.get('seller'),
        'transmission': values.get('trans'),
    }
    return pd.DataFrame([row])

app.layout = html.Div(style={"maxWidth": "900px", "margin": "40px auto"}, children=[
    html.H2("Car Price Prediction"),
    html.P("Enter what you know; missing fields are fine — the model imputes them if the Pipeline includes Imputers."),

    html.Div(style={"display":"grid", "gridTemplateColumns":"1fr 1fr", "gap":"12px"}, children=[
        html.Label("Car Name (full model name)"),
        dcc.Input(id="name", type="text", placeholder="e.g., Maruti Swift VDI"),

        html.Label("Year"),
        dcc.Input(id="year", type="number", placeholder="e.g., 2016"),

        html.Label("KM Driven"),
        dcc.Input(id="km", type="number", placeholder="e.g., 65000"),

        html.Label("Owner (1=First, 2=Second, 3=Third, 4=Fourth+)"),
        dcc.Input(id="owner", type="number", min=1, max=4, placeholder="1–4"),

        html.Label("Mileage (kmpl)"),
        dcc.Input(id="mileage", type="number", step="any", placeholder="e.g., 18.5"),

        html.Label("Engine (CC)"),
        dcc.Input(id="engine", type="number", step="any", placeholder="e.g., 1498"),

        html.Label("Max Power (bhp)"),
        dcc.Input(id="power", type="number", step="any", placeholder="e.g., 98.6"),

        html.Label("Seats"),
        dcc.Input(id="seats", type="number", placeholder="e.g., 5"),
    ]),

    html.Div(style={"display":"grid", "gridTemplateColumns":"1fr 1fr 1fr", "gap":"12px", "marginTop":"12px"}, children=[
        html.Div([html.Label("Fuel"), dcc.Dropdown(FUEL_OPTS, id="fuel", placeholder="Select fuel")]),
        html.Div([html.Label("Seller Type"), dcc.Dropdown(SELLER_OPTS, id="seller", placeholder="Select seller")]),
        html.Div([html.Label("Transmission"), dcc.Dropdown(TRANS_OPTS, id="trans", placeholder="Select transmission")]),
    ]),

    html.Button("Predict Price", id="go", n_clicks=0, style={"marginTop":"16px"}),
    html.H3(id="out", style={"marginTop":"16px"}),
    html.Pre(id="dbg", style={"opacity":0.6}),
])

@app.callback(
    Output("out", "children"),
    Output("dbg", "children"),
    Input("go", "n_clicks"),
    State("name","value"),
    State("year","value"),
    State("km","value"),
    State("owner","value"),
    State("mileage","value"),
    State("engine","value"),
    State("power","value"),
    State("seats","value"),
    State("fuel","value"),
    State("seller","value"),
    State("trans","value"),
    prevent_initial_call=True
)
def predict(_, name, year, km, owner, mileage, engine, power, seats, fuel, seller, trans):
    try:
        X = build_row({
            "name": name, "year": year, "km": km, "owner": owner,
            "mileage": mileage, "engine": engine, "power": power, "seats": seats,
            "fuel": fuel, "seller": seller, "trans": trans
        })

        y_pred = MODEL.predict(X)
        # If your model predicts log-price (you trained on log(y)), flip this flag to True.
        if USE_LOG_TARGET:
            import numpy as np
            y_pred = np.exp(y_pred)

        pred_val = float(y_pred[0])
        return f"Estimated selling price: {pred_val:,.0f}", X.to_json(orient="records", indent=2)

    except Exception as e:
        return "Prediction failed. Please review inputs.", f"Error: {e}"

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8050, debug=False)

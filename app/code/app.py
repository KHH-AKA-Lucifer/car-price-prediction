import sys
import joblib
import pandas as pd
import numpy as np
from pathlib import Path
from dash import Dash, html, dcc, Input, Output, State

from LinearRegression import LinearRegression
from LinearRegression import LassoPenalty, RidgePenalty
main_mod = sys.modules.get("__main__")
if main_mod:
    setattr(main_mod, "LassoPenalty", LassoPenalty)
    setattr(main_mod, "RidgePenalty", RidgePenalty)

ART = Path(__file__).resolve().parents[1] / "artifacts"

def _load_one(filename: str):
    p = ART / filename
    if not p.exists():
        return None
    obj = joblib.load(p)
    # If it's our runtime-light dict (transformer + runtime)
    if isinstance(obj, dict) and "runtime" in obj and "transformer" in obj:
        return obj
    # Otherwise (e.g., old sklearn pipeline) return as is
    return getattr(obj, "best_estimator_", obj)

def load_models():
    old_model = _load_one("car-price-prediction.joblib")
    if old_model is None:
        raise FileNotFoundError("Expected app/artifacts/car-price-prediction.joblib")
    new_model = _load_one("car-price-prediction-new.joblib") or old_model
    return old_model, new_model

MODEL_OLD, MODEL_NEW = load_models()

# Columns V2 used during training (order matters only a little, but keep it consistent)
V2_FEATURES = [
    "year", "km_driven", "mileage", "engine", "max_power", "seats",
    "fuel", "seller_type", "transmission", "owner",
]


def encode_v2_categories(df: pd.DataFrame) -> pd.DataFrame:
    """Replicate LabelEncoder behavior used in V2 training."""
    # LabelEncoder maps classes in sorted order.
    # Based on your dataset/choices this is deterministic:
    fuel_map = {"Diesel": 0, "Petrol": 1}
    seller_map = {"Dealer": 0, "Individual": 1, "Trustmark Dealer": 2}
    trans_map = {"Automatic": 0, "Manual": 1}

    if "fuel" in df:
        df["fuel"] = df["fuel"].map(fuel_map).astype(float)
    if "seller_type" in df:
        df["seller_type"] = df["seller_type"].map(seller_map).astype(float)
    if "transmission" in df:
        df["transmission"] = df["transmission"].map(trans_map).astype(float)

    # You label-encoded owner after converting to str; sorted(['1','2','3','4']) -> 0..3
    # So owner_numerical_code = int(owner) - 1
    if "owner" in df and pd.notna(df["owner"].iloc[0]):
        try:
            df["owner"] = (pd.to_numeric(df["owner"], errors="coerce") - 1).astype(float)
        except Exception:
            df["owner"] = np.nan

    return df

def prepare_v2_input(input_df: pd.DataFrame) -> pd.DataFrame:
    """
    Drop fields V2 didn't use (like 'name') and align to the exact feature set.
    Unknown columns are dropped; missing ones are added as NaN (imputers/scaler will handle).
    """
    # Drop any columns V2 didn't train on (e.g., 'name')
    input_df = input_df[[c for c in input_df.columns if c in V2_FEATURES]].copy()

    # Ensure all expected columns exist and in the right order
    input_df = input_df.reindex(columns=V2_FEATURES)

    # Coerce empty strings to NaN so imputers can work (if you have them)
    input_df.replace({"": np.nan}, inplace=True)

    return input_df


# If your model was trained on log(y) and predicts log-prices, set this True.
USE_LOG_TARGET = False

# Categorical choices (dataset cleaned to Petrol/Diesel per the assignment)
FUEL_OPTS   = ["Petrol", "Diesel"]
SELLER_OPTS = ["Dealer", "Individual", "Trustmark Dealer"]
TRANS_OPTS  = ["Manual", "Automatic"]

app = Dash(__name__, suppress_callback_exceptions=True, title="Car Price Predictor")
server = app.server  # for Docker/Gunicorn if needed

def build_row(values: dict) -> pd.DataFrame:
    # map form keys -> model feature names
    key_to_col = {
        "name": "name",
        "year": "year",
        "km": "km_driven",
        "owner": "owner",
        "mileage": "mileage",
        "engine": "engine",
        "power": "max_power",
        "seats": "seats",
        "fuel": "fuel",
        "seller": "seller_type",
        "trans": "transmission",
    }
    row = {}
    for k, v in values.items():
        col = key_to_col.get(k)
        if col and v not in (None, ""):
            row[col] = v
    return pd.DataFrame([row])

# -----------------------------
# UI pieces (keep your old look)
# -----------------------------
def navbar():
    link_style = {
        "padding": "8px 12px",
        "textDecoration": "none",
        "color": "white",
        "borderRadius": "6px",
        "marginRight": "8px",
        "background": "#2b7cff",
        "display": "inline-block",
    }
    return html.Div(
        [
            dcc.Link("Home", href="/", style={**link_style, "background": "#333"}),
        ],
        style={"padding": "10px 16px", "background": "#111", "position": "sticky", "top": 0, "zIndex": 999},
    )

def form_block(prefix: str, include_name: bool = True):
    grid = []

    if include_name:
        grid += [
            html.Label("Car Name (full model name)"),
            dcc.Input(id=f"{prefix}name", type="text", placeholder="e.g., Maruti Swift VDI"),
        ]

    grid += [
        html.Label("Year"),              dcc.Input(id=f"{prefix}year", type="number", placeholder="e.g., 2016"),
        html.Label("KM Driven"),         dcc.Input(id=f"{prefix}km", type="number", placeholder="e.g., 65000"),
        html.Label("Owner (1=First, 2=Second, 3=Third, 4=Fourth+)"),
                                         dcc.Input(id=f"{prefix}owner", type="number", min=1, max=4, placeholder="1–4"),
        html.Label("Mileage (kmpl)"),    dcc.Input(id=f"{prefix}mileage", type="number", step="any", placeholder="e.g., 18.5"),
        html.Label("Engine (CC)"),       dcc.Input(id=f"{prefix}engine", type="number", step="any", placeholder="e.g., 1498"),
        html.Label("Max Power (bhp)"),   dcc.Input(id=f"{prefix}power", type="number", step="any", placeholder="e.g., 98.6"),
        html.Label("Seats"),             dcc.Input(id=f"{prefix}seats", type="number", placeholder="e.g., 5"),
    ]

    return html.Div(style={"maxWidth":"900px","margin":"40px auto"}, children=[
        html.H2("Car Price Prediction"),
        html.P("Please, enter details below and click Predict Price. Missing out something will not give you the price."),
        html.Div(style={"display":"grid","gridTemplateColumns":"1fr 1fr","gap":"12px"}, children=grid),
        html.Div(style={"display":"grid","gridTemplateColumns":"1fr 1fr 1fr","gap":"12px","marginTop":"12px"}, children=[
            html.Div([html.Label("Fuel"), dcc.Dropdown(FUEL_OPTS, id=f"{prefix}fuel", placeholder="Select fuel")]),
            html.Div([html.Label("Seller Type"), dcc.Dropdown(SELLER_OPTS, id=f"{prefix}seller", placeholder="Select seller")]),
            html.Div([html.Label("Transmission"), dcc.Dropdown(TRANS_OPTS, id=f"{prefix}trans", placeholder="Select transmission")]),
        ]),
        html.Button("Predict Price", id=f"{prefix}go", n_clicks=0, style={"marginTop":"16px"}),
        html.H3(id=f"{prefix}out", style={"marginTop":"16px"}),
        html.Pre(id=f"{prefix}dbg", style={"opacity":0.6}),
    ])

def home_page():
    card = {
        "border": "1px solid #e5e5e5",
        "borderRadius": 12,
        "padding": 16,
        "flex": 1,
        "minWidth": 260,
        "background": "white",
        "boxShadow": "0 4px 16px rgba(0,0,0,0.06)",
    }
    btn = {
        "display": "inline-block",
        "padding": "10px 14px",
        "background": "#2b7cff",
        "color": "white",
        "textDecoration": "none",
        "borderRadius": 8,
        "marginTop": 10,
    }
    return html.Div(
        [
            html.Div(style={"maxWidth": "900px", "margin": "24px auto"}, children=[
                html.H2("Welcome!", style={"textAlign": "center", "margin": "24px 0 8px"}),
                html.P("Choose which model you’d like to use.", style={"textAlign": "center", "color": "#555", "marginBottom": 30}),
                html.Div(
                    [
                        html.Div([html.H3("Old Model"),
                                  html.P("Use this to compare results and behavior with the upgraded model."),
                                  dcc.Link("Go to Old Model", href="/old", style=btn)], style=card),
                        html.Div([html.H3("New Model"),
                                  html.P("Same interface, upgraded model pipeline (if provided)."),
                                  dcc.Link("Try New Model", href="/new", style=btn)], style=card),
                    ],
                    style={"display": "flex", "gap": 16, "flexWrap": "wrap", "justifyContent": "center"},
                ),
            ])
        ]
    )

def old_page():
    return form_block("old-", include_name=True)

def new_page():
    return html.Div([
        html.Div(style={"maxWidth":"900px","margin":"24px auto 0"}, children=[
            html.H2("New Model"),
            dcc.Markdown(
                "**How to use**\n"
                "1) Fill in the fields below.\n"
                "2) Click **Predict Price** and wait for the result.\n\n"

                "This model may have a different pipeline than the old model but with advanced model, but the input fields are the same.\n\n"
            ),
        ]),
        form_block("new-", include_name=False)
    ])

# -------------
# Main layout
# -------------
app.layout = html.Div([
    dcc.Location(id="url"),
    navbar(),
    html.Div(id="page-content"),
])

@app.callback(Output("page-content", "children"), Input("url", "pathname"))
def route(pathname):
    if pathname == "/old":
        return old_page()
    if pathname == "/new":
        return new_page()
    return home_page()

# -----------------
# Predict callbacks
# -----------------
@app.callback(
    Output("old-out", "children"),
    Output("old-dbg", "children"),
    Input("old-go", "n_clicks"),
    State("old-name","value"),
    State("old-year","value"),
    State("old-km","value"),
    State("old-owner","value"),
    State("old-mileage","value"),
    State("old-engine","value"),
    State("old-power","value"),
    State("old-seats","value"),
    State("old-fuel","value"),
    State("old-seller","value"),
    State("old-trans","value"),
    prevent_initial_call=True
)
def predict_old(_, name, year, km, owner, mileage, engine, power, seats, fuel, seller, trans):
    try:
        X = build_row({
            "name": name, "year": year, "km": km, "owner": owner,
            "mileage": mileage, "engine": engine, "power": power, "seats": seats,
            "fuel": fuel, "seller": seller, "trans": trans, "owner": owner
        })
        y_pred = MODEL_OLD.predict(X)
        if USE_LOG_TARGET:
            import numpy as np
            y_pred = np.exp(y_pred)
        pred_val = float(y_pred[0])
        return f"Estimated selling price: ₹ {pred_val:,.0f}", X.to_json(orient="records", indent=2)
    except Exception as e:
        return "Prediction failed. Please review inputs.", f"Error: {e}"

@app.callback(
    Output("new-out", "children"),
    Output("new-dbg", "children"),
    Input("new-go", "n_clicks"),
    State("new-year","value"),
    State("new-km","value"),
    State("new-mileage","value"),
    State("new-engine","value"),
    State("new-power","value"),
    State("new-seats","value"),
    State("new-fuel","value"),
    State("new-seller","value"),
    State("new-trans","value"),
    State("new-owner","value"),
    prevent_initial_call=True
)
def predict_new(_, year, km, mileage, engine, power, seats, fuel, seller, trans, owner):
    try:
        X = build_row({
            "year": year, "km": km,
            "mileage": mileage, "engine": engine, "power": power, "seats": seats,
            "fuel": fuel, "seller": seller, "trans": trans, "owner": owner
        })
        # print("V2 Columns:", list(X.columns))
        X = prepare_v2_input(X)  # Align to V2 features if needed
        X = encode_v2_categories(X)  # Replicate LabelEncoder behavior
        y_pred = MODEL_NEW.predict(X)
        if USE_LOG_TARGET:
            import numpy as np
            y_pred = np.exp(y_pred)
        pred_val = float(y_pred[0])
        return f"Estimated selling price: ₹ {pred_val:,.0f}", X.to_json(orient="records", indent=2)
    except Exception as e:
        return "Prediction failed. Please review inputs.", f"Error: {e}"

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8050, debug=False)

import dash
from dash import Dash, html, dcc, callback, Input, Output, State, no_update
import pandas as pd
import numpy as np
import joblib

# --------------------------------------------------------------------------------------
# (1) App setup
# --------------------------------------------------------------------------------------
app: Dash = dash.Dash(__name__, suppress_callback_exceptions=True, title="Car Price App")
server = app.server  # for Docker/Gunicorn if needed

# --------------------------------------------------------------------------------------
# (2) Shared form builder (same interface for Old and New pages)
#     - Reuse this exact UI for both models to satisfy "same interface"
#     - Replace inputs here to match your real form fields
# --------------------------------------------------------------------------------------
def build_model_form(page_mode: str) -> html.Div:
    """
    Build the input form used by both Old/New pages.
    page_mode: 'old' or 'new' (affects which model is called on submit)
    """
    prefix = f"{page_mode}-"   # ensure unique component IDs per page
    return html.Div(
        [
            html.Div(
                [
                    html.Label("Year"),
                    dcc.Input(id=f"{prefix}year", type="number", placeholder="e.g., 2019", style={"width": "100%"}),
                ],
                style={"marginBottom": 10},
            ),
            html.Div(
                [
                    html.Label("Mileage (km)"),
                    dcc.Input(id=f"{prefix}mileage", type="number", placeholder="e.g., 45000", style={"width": "100%"}),
                ],
                style={"marginBottom": 10},
            ),
            html.Div(
                [
                    html.Label("Engine Size (L)"),
                    dcc.Input(id=f"{prefix}engine", type="number", step="0.1", placeholder="e.g., 1.6", style={"width": "100%"}),
                ],
                style={"marginBottom": 10},
            ),
            html.Div(
                [
                    html.Label("Transmission"),
                    dcc.Dropdown(
                        id=f"{prefix}trans",
                        options=[{"label": "Automatic", "value": "auto"}, {"label": "Manual", "value": "man"}],
                        placeholder="Select transmission",
                        clearable=True,
                    ),
                ],
                style={"marginBottom": 10},
            ),

            html.Button("Submit", id=f"{prefix}submit", n_clicks=0, style={"width": "100%", "padding": "10px"}),
            html.Div(
                id=f"{prefix}output-box",
                className="output-box",
                style={
                    "marginTop": 15,
                    "padding": 12,
                    "border": "1px solid #ddd",
                    "borderRadius": 6,
                    "minHeight": 48,
                    "backgroundColor": "#fafafa",
                },
            ),
        ],
        style={"maxWidth": 520, "margin": "0 auto"},
    )

# --------------------------------------------------------------------------------------
# (3) Navbar (appears on every page)
# --------------------------------------------------------------------------------------
def navbar() -> html.Div:
    link_style = {
        "padding": "10px 14px",
        "textDecoration": "none",
        "color": "white",
        "borderRadius": "6px",
        "marginRight": "10px",
        "background": "#2b7cff",
        "display": "inline-block",
    }
    return html.Div(
        [
            dcc.Link("Home", href="/", style={**link_style, "background": "#333"}),
            dcc.Link("Old Model", href="/old", style=link_style),
            dcc.Link("New Model", href="/new", style=link_style),
        ],
        style={
            "display": "flex",
            "alignItems": "center",
            "gap": "10px",
            "padding": "10px 16px",
            "background": "#111",
            "position": "sticky",
            "top": 0,
            "zIndex": 999,
        },
    )

# --------------------------------------------------------------------------------------
# (4) Pages
# --------------------------------------------------------------------------------------
def home_page() -> html.Div:
    card_style = {
        "border": "1px solid #e5e5e5",
        "borderRadius": 12,
        "padding": 16,
        "flex": 1,
        "minWidth": 260,
        "background": "white",
        "boxShadow": "0 4px 16px rgba(0,0,0,0.06)",
    }
    btn_style = {
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
            html.H2("Welcome!", style={"textAlign": "center", "margin": "24px 0 8px"}),
            html.P(
                "Choose which model you’d like to use.",
                style={"textAlign": "center", "color": "#555", "marginBottom": 30},
            ),
            html.Div(
                [
                    html.Div(
                        [
                            html.H3("Old Model"),
                            html.P(
                                "The original version. Use this to compare results and behavior with the upgraded model."
                            ),
                            dcc.Link("Go to Old Model", href="/old", style=btn_style),
                        ],
                        style=card_style,
                    ),
                    html.Div(
                        [
                            html.H3("New Model"),
                            html.P(
                                "The improved version. Faster preprocessing, better feature handling, and more robust predictions."
                            ),
                            dcc.Link("Try New Model", href="/new", style=btn_style),
                        ],
                        style=card_style,
                    ),
                ],
                style={"display": "flex", "gap": 16, "flexWrap": "wrap", "justifyContent": "center"},
            ),
        ],
        style={"padding": "16px"},
    )

def old_model_page() -> html.Div:
    return html.Div(
        [
            html.H2("Old Model"),
            html.P("Use the form below to get a prediction."),
            build_model_form("old"),
        ],
        style={"padding": "16px"},
    )

def new_model_page() -> html.Div:
    return html.Div(
        [
            html.H2("New Model"),
            html.Div(
                dcc.Markdown(
                    """
**How to use**  
1. Fill in the fields below (Year, Mileage, Engine Size, Transmission).  
2. Click **Submit** and wait for the prediction in the output box.

**What’s better than the old model?**  
- More stable feature preprocessing (handles missing/edge values more gracefully)  
- Improved generalization on recent data  
- Faster inference path and clearer error messages
                    """
                ),
                style={
                    "maxWidth": 720,
                    "margin": "12px auto 20px",
                    "padding": 12,
                    "border": "1px solid #e5e5e5",
                    "borderRadius": 8,
                    "background": "#fcfcff",
                },
            ),
            build_model_form("new"),
        ],
        style={"padding": "16px"},
    )

# --------------------------------------------------------------------------------------
# (5) Router layout
# --------------------------------------------------------------------------------------
app.layout = html.Div(
    [
        dcc.Location(id="url"),
        navbar(),
        html.Div(id="page-content"),
    ]
)

@callback(Output("page-content", "children"), Input("url", "pathname"))
def render_page(pathname: str):
    if pathname == "/old":
        return old_model_page()
    if pathname == "/new":
        return new_model_page()
    # default
    return home_page()

# --------------------------------------------------------------------------------------
# (6) Dummy predictors (replace with your real model code)
#     - If you already have loaded pipelines/models, import and call them here.
# --------------------------------------------------------------------------------------
def predict_old(year, mileage, engine, trans):
    # TODO: replace with real old predictor
    if None in (year, mileage, engine) or trans is None:
        return "Please fill all fields."
    base = 20000
    price = base + (2025 - int(year)) * -800 + (float(engine) * 1200) + (0 if trans == "man" else 500)
    price -= (int(mileage) / 1000) * 30
    return f"Estimated price (Old Model): ${max(1000, round(price, 2))}"

def predict_new(year, mileage, engine, trans):
    # TODO: replace with real new predictor
    if None in (year, mileage, engine) or trans is None:
        return "Please fill all fields."
    base = 21000
    price = base + (2025 - int(year)) * -900 + (float(engine) * 1400) + (0 if trans == "man" else 700)
    price -= (int(mileage) / 1000) * 28
    return f"Estimated price (New Model): ${max(1200, round(price, 2))}"

# --------------------------------------------------------------------------------------
# (7) Callbacks for Old/New pages (identical UI, different predictor)
# --------------------------------------------------------------------------------------
@callback(
    Output("old-output-box", "children"),
    Input("old-submit", "n_clicks"),
    State("old-year", "value"),
    State("old-mileage", "value"),
    State("old-engine", "value"),
    State("old-trans", "value"),
    prevent_initial_call=True,
)
def on_submit_old(n, year, mileage, engine, trans):
    return predict_old(year, mileage, engine, trans)

@callback(
    Output("new-output-box", "children"),
    Input("new-submit", "n_clicks"),
    State("new-year", "value"),
    State("new-mileage", "value"),
    State("new-engine", "value"),
    State("new-trans", "value"),
    prevent_initial_call=True,
)
def on_submit_new(n, year, mileage, engine, trans):
    return predict_new(year, mileage, engine, trans)

# --------------------------------------------------------------------------------------
# (8) Run
# --------------------------------------------------------------------------------------
if __name__ == "__main__":
    # Dash default host/port align with your docker-compose (8050)
    new_design.run_server(host="0.0.0.0", port=8050, debug=True)

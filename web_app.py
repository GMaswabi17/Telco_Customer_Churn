import dash
from dash import dcc, html, Input, Output, State
import dash_bootstrap_components as dbc
import pandas as pd
import joblib
import plotly.express as px
import os
 
print("Working directory:", os.getcwd())
 
# Initialize Dash app
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
server = app.server
 
# Load models
rf_model = joblib.load("random_forest_model.pkl")
xgb_model = joblib.load("xgb_model.pkl")
logisticalregression_model = joblib.load("logisticalregression_model.pkl")
 
# Load feature names
X_sample = pd.read_csv("_data/cleaned_X_test.csv")
input_features = X_sample.columns
 
# Layout
app.layout = dbc.Container([
    html.H1("Customer Churn Predictor", className="text-center my-4"),
 
    html.P("Enter customer data below to predict whether they will churn using one of the ML models."),
 
    dbc.Row([
        dbc.Col([
            html.Label(f"{feature.replace('_', ' ')}:"),
            dcc.Input(id=f"input-{feature}", type="text", className="mb-2", style={"width": "100%"})
        ]) for feature in input_features
    ], className="mb-4"),
 
    dbc.Row([
        dbc.Col([
            html.Label("Select Model:"),
            dcc.Dropdown(
                id='model-select',
                options=[
                    {'label': 'Random Forest', 'value': 'rf'},
                    {'label': 'XGBoost', 'value': 'xgb'},
                    {'label': 'Logistic Regression', 'value': 'lr'}
                ],
                value='rf',
                clearable=False
            )
        ], width=4)
    ], className="mb-3"),
 
    dbc.Row([
        dbc.Col([
            dbc.Button("Predict", id="predict-btn", color="primary", className="mb-3"),
            html.Div(id="prediction-output", className="h5")
        ])
    ])
], fluid=True)
 
# Callback
@app.callback(
    Output("prediction-output", "children"),
    Input("predict-btn", "n_clicks"),
    State("model-select", "value"),
    [State(f"input-{feature}", "value") for feature in input_features]
)
def predict_churn(n_clicks, model_choice, *input_values):
    if not n_clicks:
        return ""
 
    try:
        input_df = pd.DataFrame([input_values], columns=input_features)
 
        # Convert types if necessary (optional, depends on your preprocessing pipeline)
        # Example: input_df = input_df.astype(X_sample.dtypes.to_dict())
 
        if model_choice == 'rf':
            pred = rf_model.predict(input_df)[0]
        elif model_choice == 'xgb':
            pred = xgb_model.predict(input_df)[0]
        else:
            pred = logisticalregression_model.predict(input_df)[0]
 
        return f"Predicted Result: {'Churn' if pred == 1 else 'No Churn'}"
 
    except Exception as e:
        return f"Error in prediction: {e}"
 
# Run app
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8050))
    app.run(host="0.0.0.0", port=port, debug=False)
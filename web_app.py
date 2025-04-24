import dash
from dash import dcc, html, Input, Output, State
import dash_bootstrap_components as dbc
import pandas as pd
import joblib
import plotly.express as px
import os

# Initialize Dash app
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
server = app.server

# Load models
rf_model = joblib.load("random_forest_model.pkl")
xgb_model = joblib.load("xgb_model.pkl")
logisticalregression_model = joblib.load("logisticalregression_model.pkl")

input_fields = [
    ('gender', ['Male', 'Female']),
    ('SeniorCitizen', ['Yes', 'No']),
    ('Partner', ['Yes', 'No']),
    ('Dependents', [str(i) for i in range(1, 6)]),
    ('tenure', 'number'),
    ('PhoneService', ['Yes', 'No']),
    ('MultipleLines', ['Yes', 'No', 'No phone service']),
    ('InternetService', ['DSL', 'Fiber optic', 'No']),
    ('OnlineSecurity', ['Yes', 'No', 'No internet service']),
    ('OnlineBackup', ['Yes', 'No', 'No internet service']),
    ('DeviceProtection', ['Yes', 'No', 'No internet service']),
    ('TechSupport', ['Yes', 'No', 'No internet service']),
    ('StreamingTV', ['Yes', 'No', 'No internet service']),
    ('StreamingMovies', ['Yes', 'No', 'No internet service']),
    ('Contract', ['Month-to-month', 'One year', 'Two year']),
    ('PaperlessBilling', ['Yes', 'No']),
    ('PaymentMethod', ['Electronic check', 'Mailed check', 'Credit card', 'Bank transfer']),
    ('MonthlyCharges', 'number'),
    ('TotalCharges', 'number'),
]

# Layout
app.layout = dbc.Container([
    html.H1("Telco Customer Churn Predictor", className="text-center my-4"),

    dbc.Row([
        dbc.Col([
            html.H4("Input Customer Data"),
            *[html.Div([
                html.Label(field[0], className="form-label"),
                dcc.Dropdown(
                    id=field[0],
                    options=[{'label': val, 'value': val} for val in field[1]],
                    placeholder=f"Select {field[0]}",
                    className="mb-3"
                ) if isinstance(field[1], list) else dcc.Input(
                    id=field[0],
                    type='number',
                    placeholder=f"Enter {field[0]}",
                    className='form-control mb-3')
            ]) for field in input_fields],

            html.Br(),
            html.Label("Select Model for Prediction", className="form-label"),
            dcc.Dropdown(
                id='model-choice',
                options=[
                    {'label': 'Random Forest', 'value': 'rf'},
                    {'label': 'XGBoost', 'value': 'xgb'},
                    {'label': 'Logistical Regression', 'value': 'model'}
                ],
                placeholder="Select a model",
                className="mb-3"
            ),
            dbc.Button("Predict Churn", id='predict-btn', color='primary', className='w-100'),
            html.Br(),
            html.Label("Prediction Result", className="form-label mt-3"),
            dcc.Input(id='prediction-result', type='text', readOnly=True, className='form-control mb-3')
        ], width=6)
    ])
], fluid=True)

# Callback for prediction
@app.callback(
    Output('prediction-result', 'value'),
    Input('predict-btn', 'n_clicks'),
    State('model-choice', 'value'),
    *[State(field[0], 'value') for field in input_fields]
)
def predict(n_clicks, model_choice, *values):
    if n_clicks is None:
        return ''

    input_dict = {field[0]: val for field, val in zip(input_fields, values)}
    input_df = pd.DataFrame([input_dict])

    if model_choice == 'rf':
        model = rf_model
    elif model_choice == 'xgb':
        model = xgb_model
    else:
        model = logisticalregression_model

    pred = model.predict(input_df)[0]
    result = 'Yes' if pred == 1 else 'No'
    return f"Predicted Churn: {result}"

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8050))
    app.run(host="0.0.0.0", port=port, debug=False)
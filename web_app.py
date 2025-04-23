
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

# Load data and models
X = pd.read_csv("_data/cleaned_X_test.csv")
y_true = pd.read_csv("_data/cleaned_y_test.csv").squeeze()

grade_map = {0: "A", 1: "B", 2: "C", 3: "D", 4: "F"}
y_true_labels = y_true.map(grade_map)

# Load models
rf_model = joblib.load("random_forest_model.pkl")
xgb_model = joblib.load("xgb_model.pkl")
logisticalregression_model = joblib.load("logisticalregression_model.pkl")

rf_preds = pd.Series(rf_model.predict(X)).map(grade_map)
xgb_preds = pd.Series(xgb_model.predict(X)).map(grade_map)
logisticalregression_preds = pd.Series(logisticalregression_model.predict(X)).map(grade_map)

# Layout
app.layout = dbc.Container([
    html.H1("Telco Customer Churn Predictor", className="text-center my-4"),

    html.P("This dashboard allows comparison of predictions made by the Random Forest, XGBoost and Logistical Regression models."),

    dbc.Row([
        dbc.Col([
            html.Label("Select Model:"),
            dcc.Dropdown(
                id='model-select',
                options=[
                    {'label': 'Random Forest', 'value': 'rf'},
                    {'label': 'XGBoost', 'value': 'xgb'},
                    {'label': 'Logistical Regression', 'value': 'model'}
                ],
                value='rf',
                clearable=False
            )
        ], width=4)
    ], className="mb-4"),

    dbc.Row([
        dbc.Col([
            dcc.Graph(id='prediction-table')
        ])
    ]),

    dbc.Row([
        dbc.Col([
            html.H5("Model Performance:"),
            html.Div(id='accuracy-output', className='mb-2'),
            html.Img(id='conf-matrix-img', style={"width": "100%"})
        ])
    ]),

    dbc.Row([
        dbc.Col([
            html.H5("Prediction Distribution:"),
            dcc.Graph(id='prediction-distribution')
        ])
    ])
], fluid=True)

# Callbacks
@app.callback(
    [Output('prediction-table', 'figure'),
     Output('accuracy-output', 'children'),
     Output('conf-matrix-img', 'src'),
     Output('prediction-distribution', 'figure')],
    Input('model-select', 'value')
)
def update_output(model_name):
    if model_name == 'rf':
        preds = rf_preds
        acc = (y_true_labels == preds).mean()
        img_src = "/assets/RandomForest_Confusion-matrix_Heat.png"
    elif model_name == 'xgb':
        preds = xgb_preds
        acc = (y_true_labels == preds).mean()
        img_src = "/assets/XGBoost_Confusion-matrix_Heat.png"
    else:
        preds = logisticalregression_preds
        acc = (y_true_labels == preds).mean()
        img_src = "/assets/XGBoost_Confusion-matrix_Heat.png"

    # Table figure
    table_fig = {
        'data': [ {
            'type': 'table',
            'header': {
                'values': list(X.head().columns) + ['Actual', 'Predicted'],
                'fill': {'color': 'orange'},
                'align': 'left'
            },
            'cells': {
                'values': [X[col].head().tolist() for col in X.columns] +
                          [y_true_labels.head().tolist(), preds.head().tolist()],
                'align': 'left'
            }
        }]
    }

    # Bar chart
    df_plot = pd.DataFrame({
        "Actual": y_true_labels,
        "Predicted": preds
    })

    dist_fig = px.histogram(df_plot.melt(var_name='Type', value_name='Grade'),
                            x='Grade', color='Type', barmode='group',
                            title="Grade Distribution: Actual vs Predicted",
                            color_discrete_map={"Actual": "green", "Predicted": "orange"})

    return table_fig, f"Accuracy: {acc:.2f}", img_src, dist_fig

# Run app
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8050))  
    app.run(host="0.0.0.0", port=port, debug=False)
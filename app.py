import pandas as pd
from dash import Dash, html, dcc, callback, Output, Input
import plotly.express as px
from synthesizing import *

model_name_cache = None
dataset_name_cache = None
real_data_cache = None
synthetic_datasets_cache = None
models = ['ct_gan_synthetic_data', 'tvae_synthetic_data', 'fast_ml_synthetic_data', 'gaussian_synthetic_data']
real_dataset_names = ["linear", "sin"]

app = Dash(__name__)

app.layout = html.Div([
    html.H1(children='Dash & SVD demo app', style={'textAlign': 'center'}),
    dcc.Dropdown(models, models[0], id='model-option'),
    dcc.Dropdown(real_dataset_names, real_dataset_names[0], id='dataset-option'),
    dcc.Graph(id='graph-content')
])


@callback(
    Output('graph-content', 'figure'),
    Input('dataset-option', 'value'),
    Input('model-option', 'value')
)
def update_graph_from_dataset(dataset_value, model_value):
    data_getter_mapping = {
        "linear": get_linear_dataframe,
        "sin": get_sin_dataframe
    }
    global model_name_cache, dataset_name_cache, real_data_cache, synthetic_datasets_cache

    if dataset_name_cache != dataset_value:
        real_data = data_getter_mapping[dataset_value]()
        real_data_cache = real_data

        synthetic_datasets = get_synthetic_datasets(real_data, epochs=20)
        synthetic_datasets_cache = synthetic_datasets

        real_data_df = pd.DataFrame({'X': real_data.X, 'Y': real_data.Y, 'model': 'real'})
        synthetic_data = synthetic_datasets[model_value]
        synthetic_data_df = pd.DataFrame({'X': synthetic_data.X, 'Y': synthetic_data.Y, 'model': model_value})
        df = pd.concat([real_data_df, synthetic_data_df])

        dataset_name_cache = dataset_value

        return px.scatter(df, x='X', y='Y', color='model')

    if model_name_cache != model_value:
        real_data = real_data_cache
        synthetic_datasets = synthetic_datasets_cache

        real_data_df = pd.DataFrame({'X': real_data.X, 'Y': real_data.Y, 'model': 'real'})
        synthetic_data = synthetic_datasets[model_value]
        synthetic_data_df = pd.DataFrame({'X': synthetic_data.X, 'Y': synthetic_data.Y, 'model': model_value})
        df = pd.concat([real_data_df, synthetic_data_df])

        model_name_cache = model_value

        return px.scatter(df, x='X', y='Y', color='model')


if __name__ == '__main__':
    app.run_server(debug=True)

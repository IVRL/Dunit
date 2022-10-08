import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import numpy as np
import torch
from src.models.losses import AVAILABLE_IOU

external_stylesheets = [
    "https://stackpath.bootstrapcdn.com/bootstrap/4.3.1/css/bootstrap-grid.min.css"]

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

GT = "[(10, 10, 20, 20), (80, 80, 90, 90)]"
PREDICTION = ("[(10, 10, 20, 20), (" +
    "index * 10 + (1 - index) * 80, " +
    "index * 10 + (1 - index) * 80, " +
    "index * 20 + (1 - index) * 90, " +
    "index * 20 + (1 - index) * 90" +
    ")]")
INDEXES = np.arange(-.1, 1.1, 0.1)

def _reload(gt, prediction):
    ground_truth = torch.tensor(eval(gt), dtype=torch.float32)
    boxes_list = [
        torch.tensor(eval(prediction))
        for index in INDEXES]
    values = {}
    for key, value in AVAILABLE_IOU.items():
        values[key] = [1 - value(ground_truth, boxes)
                    for boxes in boxes_list]
    return values

VALUES = _reload(GT, PREDICTION)

MARKS = {i if not i.is_integer() else int(i) : f'{i:.1f}' for i in INDEXES}

app.layout = html.Div([
    dcc.Input(id="input_gt", type="text", debounce=True, className="col-12",
              value=GT),
    dcc.Input(id="input_p", type="text", debounce=True, className="col-12",
              value=PREDICTION),
    html.Div([
        html.Div([
            dcc.Graph(id='curve', figure={}),
            dcc.Slider(id='slider', min=-.1, max=1.1, step=0.1, value=0,
                    marks=MARKS)
            ], className="col-6"),
        dcc.Graph(id='square', className="col-6", figure={})
        ], className="d-flex"),
    ])

def _get_x(box):
    return [box[0].item(), box[2].item(), box[2].item(), box[0].item(),
            box[0].item()]

def _get_y(box):
    return [box[1].item(), box[1].item(), box[3].item(), box[3].item(),
            box[1].item()]

@app.callback(
    Output(component_id='square', component_property='figure'),
    [Input(component_id='slider', component_property='value'),
     Input(component_id='input_gt', component_property='value'),
     Input(component_id='input_p', component_property='value')]
)
def update_output_div(index, gt, p):
    ground_truth = torch.tensor(eval(gt), dtype=torch.float32)
    boxes_list = torch.tensor(eval(p))
    return {
        'data': [
            {'x': _get_x(box), 'y': _get_y(box), 'type': 'line', 'name': 'ground truth'}
            for box in ground_truth] + [
            {'x': _get_x(box), 'y': _get_y(box), 'type': 'line', 'name': 'prediction'}
            for box in boxes_list] + [
            ],
        'layout': {'title': 'Boxes'}
        }

@app.callback(
    Output(component_id='curve', component_property='figure'),
    [Input(component_id='slider', component_property='value'),
     Input(component_id='input_gt', component_property='value'),
     Input(component_id='input_p', component_property='value')]
)
def update_line(i, gt, p):
    values = _reload(gt, p)
    return {
        'data': [
            {'x': INDEXES, 'y': value, 'type': 'line', "name": f"{key.title()} IoU"}
            for key, value in values.items()
            ] + [
                {'x': [i for _ in range(2500)],
                 'y': np.arange(0., max([
                     v for value in values.values() for v in value]), 1e-3),
                 'type': 'line', "showlegend": False},
            ],
        'layout': {'title': 'Losses'}
        }

if __name__ == '__main__':
    app.run_server(debug=True)

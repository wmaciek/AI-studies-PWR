import base64
import io
from typing import Dict, List, Union

import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px
import torch
from dash import dcc, html, Input, Output, no_update
from jupyter_dash import JupyterDash
from PIL import Image

from src.similarity import get_most_similar


def np_image_to_base64(im_matrix):
    im = Image.fromarray(im_matrix)
    buffer = io.BytesIO()
    im.save(buffer, format="png")
    encoded_image = base64.b64encode(buffer.getvalue()).decode()
    im_url = "data:image/jpeg;base64, " + encoded_image
    return im_url


def make_interactive_scatter_plot(
    title: str,
    z_2d: torch.Tensor,
    df: pd.DataFrame,
):
    fig = px.scatter(
        x=z_2d[:, 0],
        y=z_2d[:, 1],
        title=title,
        #color=[str(label) for label in data["label"].tolist()],
        #color_discrete_map={"1": "red", "0": "green"},
    )

    app = JupyterDash(__name__)

    app.layout = html.Div(
        className="container",
        children=[
            dcc.Graph(id="graph-5", figure=fig, clear_on_unhover=True),
            dcc.Tooltip(id="graph-tooltip-5", direction="bottom"),
        ],
    )

    @app.callback(
        Output("graph-tooltip-5", "show"),
        Output("graph-tooltip-5", "bbox"),
        Output("graph-tooltip-5", "children"),
        Input("graph-5", "hoverData"),
    )
    def display_hover(hoverData):
        if hoverData is None:
            return False, no_update, no_update

        hover_data = hoverData["points"][0]
        bbox = hover_data["bbox"]
        num = hover_data["pointNumber"]

        im_matrix = df["image"].iloc[num].permute(1, 2, 0).numpy().astype("uint8")
        im_url = np_image_to_base64(im_matrix)
        children = [
            html.Div([
                html.P(
                    f"Text: {df['text'].iloc[num]}",
                    style={
                        "width": "224px",
                        "fontSize": "10px",
                        "whiteSpace": "pre-wrap",
                    },
                ),
                html.Img(
                    src=im_url,
                    style={
                        "width": "224px",
                        "display": "block",
                        "margin": "0 auto",
                    },
                ),
            ])
        ]

        return True, bbox, children

    app.run_server(mode="inline", debug=True)


def visualize_most_similar(
    title: str,
    anchor_index: int,
    z: torch.Tensor,
    df: pd.DataFrame,
    metric: str = "l2",
    num_similar: int = 5,
) -> plt.Figure:
    fig, axs = plt.subplots(figsize=(15, 5), ncols=num_similar + 1)

    similarities, indices = get_most_similar(
        x=z,
        anchor=z[anchor_index],
        metric=metric,
        num_neighbors=num_similar,
    )

    axs[0].imshow(
        df["image"].iloc[anchor_index]
        .permute(1, 2, 0)
        .numpy()
        .astype("uint8")
    )
    axs[0].set(title="Anchor", xticks=[], yticks=[])

    for sim, index, ax in zip(similarities, indices, axs[1:]):
        ax.imshow(df["image"].iloc[index].permute(1, 2, 0).numpy().astype("uint8"))
        ax.set(title=f"Dist: {sim:.3f}", xticks=[], yticks=[])

    fig.suptitle(title)
    fig.tight_layout()

    return fig

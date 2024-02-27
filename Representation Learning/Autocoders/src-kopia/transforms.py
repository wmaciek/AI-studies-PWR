from typing import Callable, Union

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import torch

from .ae import Autoencoder


def plot_transformation(
    image: torch.Tensor,
    model: Autoencoder,
    transformation_fn: Callable[[torch.Tensor, Union[float, int]], torch.Tensor],
    min_param_value: int = 0,
    max_param_value: int = 28,
    step: int = 1,
    keep_channel_dim: bool = False,
):
    # Apply image transformations and gather latent vectors and reconstructions
    transformed_images = []
    latents = []
    reconstructions = []

    param_space = range(min_param_value, max_param_value + 1, step)

    for param in param_space:
        t_img = transformation_fn(image[0], param).unsqueeze(dim=0)

        if keep_channel_dim:
            t_img = t_img.unsqueeze(dim=0)

        with torch.no_grad():
            z = model(t_img)
            rec_t_img = model.reconstruct(t_img)[0]

        transformed_images.append(t_img if not keep_channel_dim else t_img[0])
        latents.append(z)
        reconstructions.append(rec_t_img)

    # Make transformation animation
    fig, axs = plt.subplots(figsize=(10, 3), ncols=4)

    def update_fn(idx: int):
        axs[0].imshow(image[0], cmap="gray")
        axs[0].set(title="Original", xticks=[], yticks=[])

        axs[1].imshow(transformed_images[idx][0], cmap="gray")
        axs[1].set(title="Transformed", xticks=[], yticks=[])

        z_curr = torch.cat(latents[:idx + 1], dim=0)
        axs[2].cla()
        axs[2].scatter(
            z_curr[:, 0],
            z_curr[:, 1],
            s=5,
        )
        axs[2].scatter(z_curr[0, 0], z_curr[0, 1], s=10, color="red")
        axs[2].set(title="Latent dim")

        axs[3].imshow(reconstructions[idx][0], cmap="gray")
        axs[3].set(title="Reconstructed", xticks=[], yticks=[])

    return FuncAnimation(
        fig=fig,
        func=update_fn,
        frames=len(param_space),
        interval=500,
    )

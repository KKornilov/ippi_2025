"""Bilinear interpolation implementation for numerical functions."""

import os

import click
import numpy as np
from PIL import Image


def bilinear_interpolation(x: float, y: float, image: np.ndarray) -> list[float]:
    """
    Perform bilinear interpolation for a given (x, y) coordinate in an image.

    Args:
        x, y: Floating-point coordinates in the image space.
        image: Input image as a numpy array (height, width, channels).

    Returns
    -------
        Interpolated pixel value [R, G, B(, A)].
    """
    x1, y1 = int(x), int(y)
    x2, y2 = min(x1 + 1, image.shape[1] - 1), min(y1 + 1, image.shape[0] - 1)

    q11 = image[y1, x1]
    q12 = image[y2, x1]
    q21 = image[y1, x2]
    q22 = image[y2, x2]

    dx = x - x1
    dy = y - y1

    interpolated = []
    for c in range(image.shape[2]):
        value = q11[c] * (1 - dx) * (1 - dy) + q21[c] * dx * (1 - dy) + q12[c] * (1 - dx) * dy + q22[c] * dx * dy
        interpolated.append(round(value))
    return interpolated


def resize_image(image: np.ndarray, scale: float) -> np.ndarray:
    """
    Resize image using bilinear interpolation.

    Args:
        image: Input image as numpy array
        scale: Scaling factor (e.g., 2.0 for 2x enlargement)

    Returns
    -------
        Resized image as numpy array
    """
    h, w = image.shape[:2]
    new_h, new_w = int(h * scale), int(w * scale)
    resized = np.zeros((new_h, new_w, image.shape[2]), dtype=np.uint8)

    for i in range(new_h):
        for j in range(new_w):
            # Map new coordinates to original image space
            x = j / scale
            y = i / scale
            resized[i, j] = bilinear_interpolation(x, y, image)
    return resized


@click.command()
@click.option(
    "--input",
    default="/Users/konstantinkornilov/Desktop/Ippi_python_2025/src/interpolation_project/interpolation/biliniar_algorithms/input_img/cat.jpeg",
    required=True,
    help="Input image path",
)
@click.option(
    "--output",
    default="/Users/konstantinkornilov/Desktop/Ippi_python_2025/src/interpolation_project/interpolation/biliniar_algorithms/output_img",
    required=True,
    help="Output image path",
)
@click.option("--scale", default=2.0, type=float, help="Scaling factor")
def main(input: str, output: str, scale: float) -> None:
    """Bilinear image resizing tool."""
    try:
        # Handle output path
        if os.path.isdir(output):
            input_filename = os.path.basename(input)
            output = os.path.join(output, f"resized_{input_filename}")

        if not output.lower().endswith((".png", ".jpg", ".jpeg")):
            output += ".jpg"

        img = np.array(Image.open(input))
        result = resize_image(img, scale)
        Image.fromarray(result).save(output)

        print(f"Success! Resized image saved to: {output}")

    except Exception as e:
        raise click.UsageError(str(e)) from e


if __name__ == "__main__":
    main()

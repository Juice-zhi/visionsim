from __future__ import annotations

import os
from pathlib import Path
from typing import cast

import numpy as np
import numpy.typing as npt
import torch

LINEAR_EXTENSIONS: tuple[str, ...] = (".exr", ".hdr")
"""File extensions of scene-referred formats, which store linear intensities."""


def srgb_to_linearrgb(img: torch.Tensor | npt.NDArray[np.floating]) -> torch.Tensor | npt.NDArray[np.floating]:
    """Performs sRGB to linear RGB color space conversion by reversing gamma
    correction and obtaining values that represent the scene's intensities.

    Args:
        img (torch.Tensor | npt.NDArray): Image to un-tonemap.

    Returns:
        linear rgb image.
    """
    # https://github.com/blender/blender/blob/master/source/blender/blenlib/intern/math_color.c
    module, img = (torch, img.clone()) if torch.is_tensor(img) else (np, np.copy(img))
    mask = img < 0.04045
    img[mask] = module.clip(img[mask], 0.0, module.inf) / 12.92
    img[~mask] = ((img[~mask] + 0.055) / 1.055) ** 2.4  # type: ignore
    return img


def linearrgb_to_srgb(img: torch.Tensor | npt.NDArray) -> torch.Tensor | npt.NDArray:
    """Performs linear RGB to sRGB color space conversion to apply gamma correction for display purposes.

    Args:
        img (torch.Tensor | npt.NDArray): Image to tonemap.

    Returns:
        tonemapped rgb image.
    """
    # https://github.com/blender/blender/blob/master/source/blender/blenlib/intern/math_color.c
    module, img = (torch, img.clone()) if torch.is_tensor(img) else (np, np.copy(img))
    mask = img < 0.0031308
    img[img < 0.0] = 0.0
    img[mask] = img[mask] * 12.92  # type: ignore
    img[~mask] = module.clip(1.055 * img[~mask] ** (1.0 / 2.4) - 0.055, 0.0, 1.0)
    return img


def to_linearrgb(img: npt.ArrayLike, path: str | os.PathLike) -> npt.NDArray[np.floating]:
    """Convert image data, as loaded from ``path``, to linear intensities.

    Scene-referred formats (EXR/HDR) already store linear intensities and are returned as-is,
    including values above one. Display-referred images (PNG, JPEG, etc.) are first normalized
    by the range of their integer datatype (e.g. 255 for 8-bit or 65535 for 16-bit images) and
    then un-tonemapped using the sRGB transfer function. This ensures that the same scene saved
    in either kind of format results in the same linear intensities, up to quantization.

    Args:
        img (npt.ArrayLike): Image data to convert.
        path (str | os.PathLike): Path from which the data was loaded, only its extension is used.

    Returns:
        npt.NDArray[np.floating]: Linear intensity image, where 1.0 corresponds to a display-referred white.
    """
    img = np.asarray(img)

    if Path(path).suffix.lower() in LINEAR_EXTENSIONS:
        return img.astype(float)
    if np.issubdtype(img.dtype, np.integer):
        img = img / np.iinfo(img.dtype).max
    return cast(npt.NDArray[np.floating], srgb_to_linearrgb(img.astype(float)))

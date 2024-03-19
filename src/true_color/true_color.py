import numpy as np
import xarray as xr

# Source: https://custom-scripts.sentinel-hub.com/sentinel-2/l2a_optimized/
"""
Here's a breakdown of the script using the entry point function evaluatePixel, which 
     takes in RGB bands:

1. Initial Setup: The script sets up parameters. The entry point is evaluatePixel,
     taking in RGB bands.
2. Highlight Compression and Constrast Enhancement (sAdj Function): highlight compression is applied
     to avoid maxing-out of clouds or snow. Contrast enhancement is achieved by gamma correction.
3. Saturation Adjustment (satEnh Function): The script enhances saturation by calculating
    the average saturation (avgS) and then adjusting the red, green, and blue components accordingly.
    This step is crucial for enhancing the vividness of the colors.
5. Conversion to sRGB (sRGB Function): Finally, the linear RGB values are converted to sRGB values.
    sRGB encoding is applied for no extra darkening of shadows.

Final Output: The processed values are then returned, which are the sRGB values.
"""

# Parameters
maxR = 3.0
midR = 0.13
sat = 1.2
gamma = 1.8
gOff = 0.01
gOffPow = gOff ** gamma
gOffRange = (1 + gOff) ** gamma - gOffPow


def _adjGamma(b: xr.DataArray):
    return ((b + gOff) ** gamma - gOffPow) / gOffRange


def _clip(s: xr.DataArray):
    return s.clip(min=0, max=1)


def _adj(a: xr.DataArray, tx, ty, maxC):
    ar = _clip(a / maxC)
    return ar * (ar * (tx / maxC + ty - 1) - ty) / (ar * (2 * tx / maxC - 1) - tx / maxC)


def _satEnh(r: xr.DataArray, g: xr.DataArray, b: xr.DataArray, avgS):
    return np.stack([_clip(avgS + r * sat), _clip(avgS + g * sat), _clip(avgS + b * sat)])


def _sAdj(a: xr.DataArray):
    return _adjGamma(_adj(a, midR, 1, maxR))


def _sRGB(c: xr.DataArray):
    return xr.where(c <= 0.0031308, 12.92 * c, 1.055 * c ** (1 / 2.4) - 0.055)


def avgS(r: xr.DataArray, g: xr.DataArray, b: xr.DataArray) -> xr.DataArray:
    return np.mean((r + g + b) / 3.0 * (1 - sat))

def enhanceImage(xarr: xr.DataArray, average_saturation: xr.DataArray|None=None):
    """
    Enhances the given image represented by the xarr DataArray.

    Args:
        xarr (xr.DataArray): The input image data array, with bands red, green and blue.
                                Data is assumed to be scaled between 0 and 1.
        average_saturation (xr.DataArray, optional): The average saturation value.
                                Defaults to None.

    Returns:
        xr.DataArray: The enhanced image data array.
    """
    r = _sAdj(xarr.sel(band='red'))
    g = _sAdj(xarr.sel(band='green'))
    b = _sAdj(xarr.sel(band='blue'))
    if average_saturation is None:
        average_saturation = avgS(r,g,b)
    rgbLin = _satEnh(r, g, b, average_saturation)
    rgbLin_xr = xr.DataArray(rgbLin, coords=xarr.coords, dims=xarr.dims)
    rgbLin_xr['avgS'] = average_saturation.values #NOQA
    return _sRGB(rgbLin_xr)

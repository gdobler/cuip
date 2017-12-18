from __future__ import print_function

import numpy as np
import matplotlib.pyplot as plt
from skimage.data import chelsea, coffee

plt.style.use("ggplot")


def single_channel_histogram(channel, bins=256):
    """Calculate channels hist params.
    Args:
        Channel (array) - 2D image channel.
    Returns:
        params (list)
    """
    shape = channel.shape
    # -- Summarize channel.
    chan_vals, chan_bins, chan_cnts = np.unique(channel.ravel(),
        return_inverse=True, return_counts=True)
    # -- Find quantiles.
    chan_quantiles = np.cumsum(chan_cnts).astype(float)
    chan_quantiles /= chan_quantiles[-1]

    return [chan_vals, chan_bins, chan_cnts, chan_quantiles, shape]


def single_channel_match(img_params, ref_params):
    """Match channels from image to reference.
    Args:
        img_params (list) - image params.
        ref_params (list) - reference params.
    Returns:
        (array) - 2D array of of matched channel.
    """
    # -- Unpack params.
    img_vals, img_bins, img_cnts, img_quantiles, img_shape = img_params
    ref_vals, ref_bins, ref_cnts, ref_quantiles, ref_shape = ref_params
    # -- Interpolate images quantiles to match reference.
    interp_vals = np.interp(img_quantiles, ref_quantiles, ref_vals)
    # -- Set interpolated values.
    interp_channel = interp_vals[img_bins].reshape(img_shape).astype(int)

    return interp_channel.astype(float)


def rgb_params(im):
    """Calculate RGB histograms.
    Args:
        im (array) - RGB image.
    Returns:
        (array) - RGB params.
    """
    # -- Calculate RGB params.
    return [single_channel_histogram(im[:, :, ii]) for ii in range(3)]


def rgb_match(img_params, ref_params):
    """Perform histogram matching for a 3 channel RGB image.
    Args:
        img (array) - RGB image.
        ref (array) - RGB reference image.
    Returns:
        new_channels (array) - RGB histogram matched channels.
    """
    # -- Match each channel.
    params = zip(img_params, ref_params)
    new_channels = np.array([single_channel_match(im, rr) for im, rr in params])
    # -- Inverse colors.
    new_channels = np.array([256, 256, 256]) - np.moveaxis(new_channels, 0, -1)

    return new_channels


def plot_match(img, ref, new_channels, figsize=(6, 8)):
    """Plot image, reference, and resulting matches.
    Args:
        img (array) - RGB image.
        ref (array) - RGB reference image.
        new_channels (array) - RGB histogram matched channels.
    """
    # -- Create figure.
    fig, [r1, r2, r3, r4] = plt.subplots(nrows=4, ncols=3, figsize=figsize)
    # -- Plot all reference and image channels.
    for ii, (ref_ax, img_ax, new_ax) in enumerate([r1, r2, r3, r4]):
        if ii < 3:
            ref_ax.imshow(ref[:, :, ii], cmap="gray")
            img_ax.imshow(img[:, :, ii], cmap="gray")
            new_ax.imshow(256 - new_channels[:, :, ii], cmap="gray")
        else:
            ref_ax.imshow(ref)
            img_ax.imshow(img)
            new_ax.imshow(new_channels)
    # -- Axes labels.
    for ax, label in zip([r1[0], r2[0], r3[0], r4[0]], ["R", "G", "B", "Color"]):
        ax.set_ylabel(label)
    for ax, label in zip(r1, ["Reference", "Image", "Match"]):
        ax.set_title(label)
    # -- Formatting.
    for ii in fig.axes:
        ii.set_xticks([])
        ii.set_yticks([])
    plt.tight_layout(h_pad=0.0001, w_pad=0.1)
    plt.show(block=True)


def demo(img=coffee(), ref=chelsea(), figsize=(6, 6)):
    """Demo the histogram matching.
    Args:
        img (array) - RGB image.
        ref (array) - RGB image to use as reference.
        figsize (tup) - plotting size.
    """
    img_params = rgb_params(img)
    ref_params = rgb_params(ref)
    new_channels = rgb_match(img_params, ref_params)
    plot_match(img, ref, new_channels, figsize=figsize)


if __name__ == "__main__":
    demo()

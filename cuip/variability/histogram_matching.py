from __future__ import print_function

import sys
import numpy as np
import matplotlib.pyplot as plt
from skimage.data import chelsea, coffee

plt.style.use("ggplot")


def single_channel_cdf(channel):
    """Calculate channel's cdf.
    Args:
        Channel (array) - 2D image channel.
    Returns:
        cdf (array) - Normalized cdf (256 bins).
    """
    # -- Calculate cdf.
    cdf = np.cumsum(np.histogram(channel.flatten(), 256, [0, 255])[0])
    # -- Normalize.
    cdf = cdf.astype(float) / cdf.max()

    return cdf


def single_channel_vals(img_cdf, ref_cdf):
    """Map values from img_cdf to ref_cdf.
    Args:
        ref_cdf (array) - reference channel cdf.
        img_cdf (array) - target channel cdf.
    Returns:
        vals (array) - mapped values for 256 values.
    """
    # -- Map img values to ref.
    vals = [list(ref_cdf >= img_cdf[ii]).index(True) for ii in range(len(img_cdf))]

    return np.array(vals).astype(float)


def single_channel_match(channel, vals):
    """Replace values in channel with mapped values.
    Args:
        channel (array) - image channel.
        vals (array) - mapped values for 256 values.
    Returns:
        match (array) - image with remapped values.
    """
    # -- Empty array to fill with mapped values.
    match = np.zeros(channel.shape, dtype=np.uint8)
    # -- Map all values in channel to matched values.
    for ii in range(256):
        match[channel == ii] = vals[ii]

    return match


def rgb_match(img, ref):
    """Histogram matching for 3 color images.
    Args:
        img (array) - target image.
        ref (array) - reference image.
    Returns:
        match (array) - histogram matched target image.
    """
    # -- Find cdfs.
    img_cdfs = [single_channel_cdf(img[:, :, ii]) for ii in range(3)]
    ref_cdfs = [single_channel_cdf(ref[:, :, ii]) for ii in range(3)]
    # -- Find value map.
    vals = [single_channel_vals(ii, rr) for ii, rr in zip(img_cdfs, ref_cdfs)]
    # -- Map image values.
    match = [single_channel_match(img[:, :, ii], vals[ii]) for ii in range(3)]

    return np.moveaxis(match, 0, -1)


def plot_match(img, ref, match, figsize=(6, 8)):
    """Plot image, reference, and resulting matches.
    Args:
        img (array) - RGB image.
        ref (array) - RGB reference image.
        match (array) - RGB histogram matched image
    """
    # -- Create figure.
    fig, [r1, r2, r3, r4] = plt.subplots(nrows=4, ncols=3, figsize=figsize)
    # -- Plot all reference and image channels.
    for ii, (ref_ax, img_ax, new_ax) in enumerate([r1, r2, r3, r4]):
        if ii < 3:
            ref_ax.imshow(ref[:, :, ii], cmap="gray")
            img_ax.imshow(img[:, :, ii], cmap="gray")
            new_ax.imshow(match[:, :, ii], cmap="gray")
        else:
            ref_ax.imshow(ref)
            img_ax.imshow(img)
            new_ax.imshow(match)
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
    match = rgb_match(img, ref)
    plot_match(img, ref, match, figsize=figsize)


def match_lightcurves(lc):
    """Perform histogram matching for all lightcurve files.
    Args:
        lc (obj) - LightCurve object.
    """
    # -- Load registration files to df.
    fnames = sorted(os.listdir(lc.path_regis))
    fpaths = [os.path.join(lc.path_regis, fname) for fname in fnames]
    reg = pd.concat([pd.read_csv(path) for path in fpaths])
    # -- Get lightcurve paths.
    fnames = sorted(os.listdir(lc.path_light))
    fpaths = [os.path.join(lc.path_light, fname) for fname in fnames]
    # -- Load reference "image" and set reference params.
    tmp = np.load(fpaths[-1])
    ref = np.ma.array(tmp[-5, :, :], mask=tmp[-5, :, :] == -9999)
    ref_params = rgb_params(ref)
    # -- For all lightcurves, histogram match each night to a reference time.
    for ii, fpath in enumerate(fpaths):
        bname = os.path.basename(fpath)
        print("LIGHTCURVES: Histogram matching {} ({}/{})                    " \
            .format(bname, ii + 1, len(fpaths)), end="\r")
        sys.stdout.flush()
        # -- Load lightcurves.
        lightc = np.load(fpath)
        malightc = np.ma.array(lightc, mask=lightc == -9999)
        # -- Histogram matching.
        mlight = np.array([rgb_match(rgb_params(src), ref_params) for src in malightc])
        # -- Save new lightcurve array.
        hpath = os.path.join(lc.path_outpu, "histogram_matching", bname)
        np.save(hpath, mlight)


if __name__ == "__main__":
    demo()
    # match_lightcurves(lc)

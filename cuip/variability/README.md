# cuip/cuip/variability 
# Description
Included code aims to resolve the following tasks after extracting the time series:
1. Provide a container for loading light curve data.
    1. `lightcurve.py` include the `LightCurve` class, which will load all lightcurves, ons/offs, bigoffs, format metadata, and create data dictionaries that are used elsewhere in the codebase and in plotting.
1. Perform histogram matching of the extracted light curves to limit time dependent variability in extracted mean RGB values.
    1. This is accomplished using `histogram_matching.py`, which performs histogram matching across all source for a given time to a reference time.
1. Preprocess light curves, extract ons and offs, and determine the big off for all sources in all nights.
    1. This is accomplished in `onsoffs.py`. The light curves are smoothed to minimize noise, masked, standardized, detrended, and undergo high pass subtraction. 
    1. Ons and offs are found by taking the guassian differences of the resulting light curves, under going 10 iterations of 2 sigma outlier rejection, and then performing a cross check to ensure that the recognized ons/offs are not a product of noise.
    1. Bigoffs are found by looking at the recognized offs that maximize--for the night--the difference in mean luminosity before and after the off time.
1. Predict residential v. non-residential buildings in scene using the output light curves.
    1. Using the lightcurve random forests were trained and tested to predicted source level building classification, this is included in `prediction.py`.
1. Lastly, all plotting code is included in `plot.py`.

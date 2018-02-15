
# -- This file outlines the workflow utilized to:
# --    1) Load lightcurves in their various forms.
# --    2) Execute histogram matching.
# --    3) Calculate ons/offs/bigoffs.
# --    4) Predict building classification.
# -- The following code will not run as an executable script. Instead, steps are
# -- identified followed by code blocks that reflect the steps taken to
# -- accomplish the tasks listed above.

# -- 1) Loading lightcurves.
# -- When referring to lightcurves there are 3 forms:
# --    i) Original RGB intensities within window appertures.
# --    ii) Histogram matched RGB intensities.
# --    iii) Median detrended luminosity intensities.
# -- Files for both i and ii are for the full range of data and split into
# -- seperate files at non-meaningful times. iii has seperate files for each
# -- night.

# -- Create the lc object. This will save the provided paths, create a metadata
# -- df, load window appertures, create relevant data dicts, and load a test
# -- night.
path_lig = "PATH TO LIGHTCURVES DIR"
path_var = "PATH TO ONS/OFFS DIR"
path_reg = "PATH TO REGISTRATION DIR"
path_sup = "PATH TO SUPPLEMENTAL DATA DIR"
path_out = "PATH TO OUTPUT DIR"

# -- If path_lig points to the original RGB dir that's it, the lightcurves are
# -- accessible under lc.lcs.; lc.lc_ons and lc.lc_offs will be empty arrays.
from cuip.variability.lightcurve import *
lc = LightCurves(path_lig, path_var, path_reg, path_sup, path_out) # -- 5 secs.

# -- To load other nights you can use:
lc.loadnight(lc.meta.index[5], lc_mean=True, load_all=False, lc_dtrend=False) # -- 2 secs.
# -- Here, using lc_mean will take the mean across RGB channels. If loading the
# -- original lightcurves, you will not be loading ons/offs or using the
# --detrended lcs.

# -- By changing path_lig, lc_mean, load_all, lc_dtrend, you can change the data
# -- sources and how they are formatted.

# -- 2) Execute histogram matching.
# -- Load original lightcurves as above and run:
from cuip.variability.histogram_matching import *
match_lightcurves(lc, "OUTPUT_PATH") # -- several hours...

# -- 3) Calculate ons/offs/bigoffs.
# -- Can be done quickly in the python interpreter, will need lc defined in the
# -- environment, and the path_lig pointing to your desired lightcurves, then...
from cuip.variability.onsoffs import *
CLI(lc) # -- Will launch a CLI to either do a one off calc or write to file.

# -- 4) Prediction.
# -- Use cuip.variability.prediction.main() to train and test individual
# -- classifiers.

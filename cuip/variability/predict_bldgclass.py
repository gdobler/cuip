from __future__ import print_function

import numpy as np
import pandas as pd
# from fastdtw import fastdtw
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import silhouette_score, confusion_matrix
from scipy.ndimage.filters import gaussian_filter1d, gaussian_filter
# from tsfresh import extract_features
# from tsfresh.feature_extraction.settings import EfficientFCParameters


def stack_nights(lc):
    """Load complete lightcurve data for weeknights. De-trend, filter noise,
    and min max each lightcurve.
    Args:
        lc (obj) - LightCurve object.
    Returns:
        data (array) - data cube.
    """

    holidates = [pd.datetime(2013, 12, 25).date(),
                 pd.datetime(2013, 12, 26).date()]
    meta_gb = lc.meta.groupby(lc.meta.index).mean()
    ndates = meta_gb[(meta_gb.null_percent < 0.05) &
                     (meta_gb.timesteps > 2875) &
                     ([ii.weekday() < 5 for ii in meta_gb.index]) &
                     ([ii not in holidates for ii in meta_gb.index])].index

    rel_bigoffs = lc.bigoffs.loc[ndates] > 0

    data = []
    # -- Stack multiple nights.
    for date in ndates:
        lc.loadnight(date)
        # -- Pull nightly lightcurves for same range of timesteps.
        lightc = pd.DataFrame(lc.src_lightc[:2876, :])
        lightc.columns = np.array(lightc.columns) + 1
        # -- Select lightcurves with a bigoff.
        bidx = [idx for idx, val in rel_bigoffs.loc[lc.night].items() if val == True]
        lightc = lightc.loc[:, np.array(bidx)]
        # -- Detrend (i.e., subtract the median).
        dlightc = (lightc.T - np.median(lightc, axis=1)).T
        # -- Filter noise.
        dlightc_gauss = pd.DataFrame(gaussian_filter(dlightc.T, [0, 10])).T
        dlightc_gauss.columns = dlightc.columns
        # -- Min max each source, and append lightcurve array to data.
        data.append(MinMaxScaler().fit_transform(dlightc_gauss.T).T)

    data = np.array(data).T

    return [data, rel_bigoffs]


def split_bbls_left_right(lc, training_size):
    """"""

    # -- Get the centroid for all of bbls sources.
    src_bbls = pd.DataFrame.from_dict(lc.coords_bbls, orient="index") \
        .rename(columns={0: "bbls"})
    src_coords = pd.DataFrame.from_dict(lc.coords, orient="index") \
        .rename(columns={0: "x", 1: "y"})
    bbl_coords = src_bbls.merge(src_coords, left_index=True, right_index=True) \
        .groupby(["bbls"]).median()

    # -- Split bbls into left and right, based on the centroid.
    splitv = int(4096 * training_size)
    lbbls = bbl_coords[bbl_coords.x <= splitv].index
    rbbls = bbl_coords[bbl_coords.x > splitv].index

    # -- Get sources corresponding to left and right bbl split.
    lsrcs = np.concatenate([lc.dd_bbl_srcs[bbl] for bbl in lbbls]).ravel()
    rsrcs = np.concatenate([lc.dd_bbl_srcs[bbl] for bbl in rbbls]).ravel()

    # -- Create dict of {key: class} for each left right split.
    keys = lc.coords_cls.keys()
    lclass = {key: lc.coords_cls[key] for key in lsrcs if key in keys}
    rclass = {key: lc.coords_cls[key] for key in rsrcs if key in keys}

    # -- How many Res v. Non Res in each set.
    lres = sum(np.array(lclass.values()) == 1)
    lnonres = sum(np.array(lclass.values()) > 1)
    rres = sum(np.array(rclass.values()) == 1)
    rnonres = sum(np.array(rclass.values()) > 1)
    print("LIGHTCURVES: Left  -- Res {}, NonRes {}                           " \
        .format(lres, lnonres))
    print("LIGHTCURVES: Right -- Res {}, NonRes {}                           " \
        .format(rres, rnonres))

    return [lclass, rclass]


def class_sources(lc):
    """"""

    res = np.array([pair[0] for pair in lc.coords_cls.items() if pair[1] == 1])
    com = np.array([pair[0] for pair in lc.coords_cls.items() if pair[1] == 2])
    mix = np.array([pair[0] for pair in lc.coords_cls.items() if pair[1] == 3])
    mis = np.array([pair[0] for pair in lc.coords_cls.items() if pair[1] == 5])

    return [res, com, mix, mis]



def train_test_datacube(data, lclass, rclass):
    """"""

    # -- Select all sources for training/testing.
    test = data[:, np.array(rclass.keys()) - 1, :]
    train = data[:, np.array(lclass.keys()) - 1, :]

    # -- Reshape into 2d array.
    test_data = np.vstack(test.T)
    train_data = np.vstack(train.T)

    # -- Reshape labels.
    test_labels = (np.array(rclass.values() * data.shape[-1]).ravel() == 1).astype(int)
    train_labels = (np.array(lclass.values() * data.shape[-1]).ravel() == 1).astype(int)

    return [train_data, train_labels, test_data, test_labels]


def dumb_classifier():
    # -- Predict buildings as residential or non-residential.
    clf = RandomForestClassifier(n_estimators=1000, random_state=0,
        class_weight="balanced")
    clf.fit(train_data, train_labels)
    testpred = clf.predict(test_data)
    print(confusion_matrix(test_labels, testpred))


# -- CODE HAS BEEN REVISED...
def dtw_cluster(mean_gaus_mm, ts_mm):
    """"""
    Ntest = len(mean_gaus_mm)
    preds = []
    for idx, src in enumerate(mean_gaus_mm):
        print("L2M: Calculating DTW Distance ({}/{})                         " \
            .format(idx + 1, Ntest), end="\r")
        sys.stdout.flush()
        dtw_scr = [fastdtw(src, ts.reshape(1, -1)[0])[0] for ts in ts_mm]
        val = np.argmin(dtw_scr)
        preds.append(val)

    print("\n", confusion_matrix((np.array(lc.coords_cls.values()) > 1) \
        .astype(int), np.array(preds)[np.array(lc.coords_cls.keys()) - 1]))


def use_tsfresh():
    """"""

    param_dict = EfficientFCParameters()
    for key in ["fft_coefficient", "change_quantiles", "cwt_coefficients"]:
        del param_dict[key]

    data = []
    for idx, src in enumerate(mean_gaus_mm):
        data.append([(idx, val) for val in src])
    data = [val for sublist in data for val in sublist]
    df = pd.DataFrame(data, columns=["idx", "val"])

    df = pd.DataFrame(mean_gaus_mm[np.array(lc.coords_cls.keys()) - 1])
    df["id"] = lc.coords_cls.values()
    df["id"] = (df["id"] == 1).astype(int)

    if os.path.isfile("./tsfresh_mean_gaus_mm.csv"):
        exfeatures = pd.read_csv("./tsfresh_mean_gaus_mm.csv")
    else:
        exfeatures = extract_features(df, column_id="idx", default_fc_parameters=param_dict)
        exfeature.to_csv("./tsfresh_mean_gaus_mm.csv")

    df1 = exfeatures.loc[np.array(lc.coords_cls.keys()) - 1]
    df1["bldgclass"] = lc.coords_cls.values()
    df1["residential"] = (df1["bldgclass"] == 1).astype(int)
    df1.fillna(0, inplace=True)

    y = df1.iloc[:, -2:]
    x = df1.iloc[:, :-2]

    fpreds = ['val__number_peaks__n_3', 'val__ar_coefficient__k_10__coeff_2',
        'val__energy_ratio_by_chunks__num_segments_10__segment_focus_6',
        'val__agg_linear_trend__f_agg_"var"__chunk_len_50__attr_"stderr"']
    fpreds = x.columns

    xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.3, random_state=0)
    clf = RandomForestClassifier(n_estimators=1000, random_state=0)
    clf.fit(xtrain[fpreds], ytrain["residential"])
    testpred = clf.predict(xtest[fpreds])
    print(confusion_matrix(ytest["residential"], testpred))

    clf = LogisticRegression(C=100, penalty="l1")
    clf.fit(xtrain[fpreds], ytrain["residential"])
    testpred = clf.predict(xtest[fpreds])
    print(confusion_matrix(ytest["residential"], testpred))


def bottom_left_right_split(lc):
    """Pull coords that are in the bottom half of the frame and split into
    left and right sets.
    Args:
        lc (obj) - LightCurves object.
    Returns:
        (list) - [lclass, rclass] dictionaries of {coord: class}.
    """

    # -- Select all sources in the bottom half of the image.
    bcoords = filter(lambda x: x[1][1] >= 1020, lc.coords.items())
    # -- Split the bottom half into left and right coords.
    lcoords = filter(lambda x: x[1][0] <= 2048, bcoords)
    rcoords = filter(lambda x: x[1][0] > 2048, bcoords)
    # -- Get keys for the left and right coords.
    lkeys = [val[0] for val in lcoords]
    rkeys = [val[0] for val in rcoords]
    # -- Create dict of {key: class} for each left right split.
    keys = lc.coords_cls.keys()
    lclass = {key: lc.coords_cls[key] for key in lkeys if key in keys}
    rclass = {key: lc.coords_cls[key] for key in rkeys if key in keys}
    # -- How many Res v. Non Res in each set.
    lres = sum(np.array(lclass.values()) == 1)
    lnonres = sum(np.array(lclass.values()) > 1)
    rres = sum(np.array(rclass.values()) == 1)
    rnonres = sum(np.array(rclass.values()) > 1)
    print("LIGHTCURVES: Left  -- Res {}, NonRes {}                           " \
        .format(lres, lnonres))
    print("LIGHTCURVES: Right -- Res {}, NonRes {}                           " \
        .format(rres, rnonres))

    return [lclass, rclass]


def left_right_pred(lclass, rclass):
    """"""

    fpreds = ['val__number_peaks__n_3', 'val__ar_coefficient__k_10__coeff_2',
        'val__energy_ratio_by_chunks__num_segments_10__segment_focus_6',
        'val__agg_linear_trend__f_agg_"var"__chunk_len_50__attr_"stderr"']

    # -- Load tsfresh data for all sources.
    if os.path.isfile("./tsfresh_mean_gaus_mm.csv"):
        exfeatures = pd.read_csv("./tsfresh_mean_gaus_mm.csv")

    # -- Create xx and yy data for left (train) and right (test) sources.
    lxx = exfeatures.loc[np.array(lclass.keys()) - 1].fillna(0)
    lyy = np.array(lclass.values())
    rxx = exfeatures.loc[np.array(rclass.keys()) - 1].fillna(0)
    ryy = np.array(rclass.values())

    # -- Predict sources as residential or non-residential.
    clf = RandomForestClassifier(n_estimators=1000, random_state=0,
        class_weight="balanced")
    clf.fit(lxx[fpreds].iloc[:50, :], (lyy == 1).astype(int)[:50])
    testpred = clf.predict(rxx[fpreds])
    print(confusion_matrix((ryy == 1).astype(int), testpred))


def sources_to_bbls(lc):
    """"""

    # -- Split bbls into train/test set.
    bbls = lc.dd_bbl_srcs.keys()
    train_bbl, test_bbl = train_test_split(bbls, test_size=0.4, random_state=0)

    # -- Load tsfresh data for all sources.
    try:
        if os.path.isfile("./tsfresh_mean_gaus_mm.csv"):
            exfeatures = pd.read_csv("./tsfresh_mean_gaus_mm.csv")
    except:
        raise("MISSING TSFRESH DATA!!")

    # -- Get x-data for training and testing.
    xtrain = [exfeatures.loc[lc.dd_bbl_srcs[bbl]].fillna(0).mean(axis=0).values
        for bbl in train_bbl]
    xtest = [exfeatures.loc[lc.dd_bbl_srcs[bbl]].fillna(0).mean(axis=0).values
        for bbl in test_bbl]

    # -- Get y-data for training and testing.
    ytrain = (np.array([lc.dd_bbl_bldg2[bbl] for bbl in train_bbl]) == 1).astype(int)
    ytest = (np.array([lc.dd_bbl_bldg2[bbl] for bbl in test_bbl]) == 1).astype(int)

    # -- Predict buildings as residential or non-residential.
    clf = RandomForestClassifier(n_estimators=1000, random_state=0,
        class_weight="balanced")
    clf.fit(xtrain, ytrain)
    testpred = clf.predict(xtest)
    print(confusion_matrix(ytest, testpred))

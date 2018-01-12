from __future__ import print_function

import cPickle
import numpy as np
import pandas as pd
from collections import Counter
# from fastdtw import fastdtw
from sklearn.cluster import KMeans
from sklearn.externals import joblib
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import silhouette_score, confusion_matrix
from scipy.ndimage.filters import gaussian_filter1d, gaussian_filter
from tsfresh import extract_features
from tsfresh.feature_extraction.settings import EfficientFCParameters


def traintestsplit(lc, train_split=0.7):
    """Split sources into training and testing sets, where each bbl is only
    assigned to either the training or test set.
    Args:
        lc (object) - LightCurve object.
        train_split (float) - proportion of training set.
    Returns:
        train (list) - List of sources' idx.
        test (list) - List of sources' idx.
    """
    np.random.seed(1)
    # -- Count the number of sources for each bbl.
    bbl = Counter(lc.coords_bbls.values())
    key = bbl.keys()
    np.random.shuffle(key)
    # -- Dictionaries to store src count and list of bbls.
    trn = {"count": 0., "bbls": []}
    tst = {"count": 0., "bbls": []}
    # -- For each bbl key add to training or test set.
    for kk in key:
        vv = bbl[kk]
        # -- If it's the first record add a records to the train set.
        if trn["count"] == 0.:
            trn["bbls"].append(kk)
            trn["count"] = trn["count"] + vv
        # -- If train set is above 0.7 of total, add to test set.
        elif trn["count"] / (trn["count"] + tst["count"]) > train_split:
            tst["bbls"].append(kk)
            tst["count"] = tst["count"] + vv
        # -- Else add to train set.
        else:
            trn["bbls"].append(kk)
            trn["count"] = trn["count"] + vv
    # -- Map source indexes based on bbls.
    train = [src for src, bbl in lc.coords_bbls.items() if bbl in trn["bbls"]]
    test  = [src for src, bbl in lc.coords_bbls.items() if bbl in tst["bbls"]]
    return [train, test]


def stack_nights_np(lc):
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

    data = []
    # -- Stack multiple nights.
    for date in ndates:
        lc.loadnight(date)
        # -- Pull nightly lightcurves for same range of timesteps.
        lightc = lc.src_lightc[:2876, :]
        # -- Detrend (i.e., subtract the median).
        dlightc = (lightc.T - np.median(lightc, axis=1)).T
        # -- Filter noise.
        dlightc_gauss = gaussian_filter(dlightc.T, [0, 10])
        # -- Min max each source, and append lightcurve array to data.
        data.append(MinMaxScaler().fit_transform(dlightc_gauss.T).T)

    data = np.array(data).T

    return [data, rel_bigoffs]


def stack_nights_df(lc):
    """"""
    # -- List nights with complete data.
    holidates = [pd.datetime(2013, 12, 25).date(), # -- Anomaly.
                 pd.datetime(2013, 12, 26).date(), # -- Anomaly.
                 pd.datetime(2013, 11, 27).date(), # -- Only 28 bigoffs.
                 pd.datetime(2013, 12, 23).date()] # -- No bigoffs.
    # -- Groupby to combine days that were spread over multiple files.
    meta_gb = lc.meta.groupby(lc.meta.index).mean()
    # -- Select weekday nights with full data.
    ndates = meta_gb[(meta_gb.null_percent < 0.05) &
                     (meta_gb.timesteps > 2875) &
                     ([ii.weekday() < 5 for ii in meta_gb.index]) &
                     ([ii not in holidates for ii in meta_gb.index])].index
    # -- Create a mask for True bigoffs.
    bigoffs_mask = lc.bigoffs.loc[ndates] > 0

    data = []
    # -- Stack multiple nights.
    for date in ndates:
        lc.loadnight(date)
        # -- Pull nightly lightcurves for same range of timesteps.
        lightc = pd.DataFrame(lc.src_lightc[:2876, :])
        lightc.columns = np.array(lightc.columns) + 1
        # -- Select lightcurves with a bigoff.
        bidx = [idx for idx, val in bigoffs_mask.loc[lc.night].items() if val == True]
        lightc = lightc.loc[:, np.array(bidx)]
        # -- Detrend (i.e., subtract the median).
        dlightc = (lightc.T - np.median(lightc, axis=1)).T
        # -- Filter noise.
        dlightc_gauss = pd.DataFrame(gaussian_filter(dlightc.T, [0, 10])).T
        dlightc_gauss.columns = dlightc.columns
        # -- Min max each source, and append lightcurve array to data.
        dlightc_mm = pd.DataFrame(MinMaxScaler().fit_transform(dlightc_gauss))
        dlightc_mm.columns = dlightc.columns
        data.append(dlightc_mm.T)

    return [data, bigoffs_mask, ndates]


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


def split_bbls_center_strip(lc):
    """"""

    # -- Get the centroid for all of bbls sources.
    src_bbls = pd.DataFrame.from_dict(lc.coords_bbls, orient="index") \
        .rename(columns={0: "bbls"})
    src_coords = pd.DataFrame.from_dict(lc.coords, orient="index") \
        .rename(columns={0: "x", 1: "y"})
    bbl_coords = src_bbls.merge(src_coords, left_index=True, right_index=True) \
        .groupby(["bbls"]).mean()

    # -- Select bbls in center strip.
    center_bbls = bbl_coords[(bbl_coords.y > 800) & (bbl_coords.y < 1100)]
    center_bbls = np.array(center_bbls.index)
    center_bbls = center_bbls[center_bbls > 1]

    return center_bbls


def random_split_bbls_train_test(lc, bbl_list, training_size):
    """"""

    train_bbls, test_bbls = train_test_split(bbl_list, train_size=training_size,
        random_state=0)

    train_srcs = np.concatenate([lc.dd_bbl_srcs[bbl] for bbl in train_bbls]).ravel()
    test_srcs = np.concatenate([lc.dd_bbl_srcs[bbl] for bbl in test_bbls]).ravel()

    # -- Create dict of {key: class} for each left right split.
    keys = lc.coords_cls.keys()
    train_class = {key: lc.coords_cls[key] for key in train_srcs if key in keys}
    test_class = {key: lc.coords_cls[key] for key in test_srcs if key in keys}

    return [train_class, test_class]


def class_sources(lc):
    """"""

    res = np.array([pair[0] for pair in lc.coords_cls.items() if pair[1] == 1])
    com = np.array([pair[0] for pair in lc.coords_cls.items() if pair[1] == 2])
    mix = np.array([pair[0] for pair in lc.coords_cls.items() if pair[1] == 3])
    mis = np.array([pair[0] for pair in lc.coords_cls.items() if pair[1] == 5])

    return [res, com, mix, mis]


def train_test_datacube(data, lclass, rclass):
    """"""

    # -- Selet all sources for training/testing.
    test = pd.concat([df[df.index.isin(np.array(rclass.keys()))] for df in data])
    train = pd.concat([df[df.index.isin(np.array(lclass.keys()))] for df in data])

    # -- Create array of training and testing labels.
    test_labels = [lc.coords_cls[idx] == 1 for idx in test.index]
    train_labels = [lc.coords_cls[idx] == 1 for idx in train.index]

    return [train, train_labels, test, test_labels]


def use_tsfresh(train, test, outputf):
    """"""

    test_fname = os.path.join(outputf, "tsfresh_leftright_7030_test.csv")
    train_fname = os.path.join(outputf, "tsfresh_leftright_7030_train.csv")

    param_dict = {'length': None, 'maximum': None, 'mean': None, 'median': None,
                  'minimum': None, 'standard_deviation': None, 'variance': None,
                  'sum_values': None, 'number_peaks': [{'n': 1}, {'n': 3},
                  {'n': 5}, {'n': 10}, {'n': 50}]}

    if os.path.isfile(test_fname):
        test_exfeatures = pd.read_csv(test_fname)
    else:
        print("LIGHTCURVES: Creating tsfresh features for test set.           ")
        test_temp = test.reset_index().rename(columns={"index": "src"})
        test_temp.index = test_temp.src.astype(str) + "-" + test_temp.index.astype(str)
        test_temp.drop("src", axis=1, inplace=True)
        test_temp = test_temp.stack()
        # -- tsfresh test data.
        test_exfeatures = extract_features(test_temp.reset_index(),
            column_id="src", column_sort="level_1",
            default_fc_parameters=param_dict)
        # -- Write to file.
        test_exfeatures.to_csv(test_fname)

    if os.path.isfile(train_fname):
        train_exfeatures = pd.read_csv(train_fname)
    else:
        print("LIGHTCURVES: Creating tsfresh features for train set.          ")
        train_temp = train.reset_index().rename(columns={"index": "src"})
        train_temp.index = train_temp.src.astype(str) + "-" + train_temp.index.astype(str)
        train_temp.drop("src", axis=1, inplace=True)
        train_temp = train_temp.stack()
        # -- tsfresh train data.
        train_exfeatures = extract_features(train_temp.reset_index(),
            column_id="src", column_sort="level_1",
            default_fc_parameters=param_dict)
        # -- Write to file.
        train_exfeature.to_csv(train_fname)

    for df in [test_exfeatures, train_exfeatures]:
        df["src"] = [int(str(idx).split("-")[0]) for idx in df["id"]]
        df["num"] = [int(str(idx).split("-")[1]) for idx in df["id"]]
        df.sort_values("num", inplace=True)
        df.set_index("src", inplace=True)
        df.drop(["id", "num"], axis=1, inplace=True)

    return [train_exfeatures, test_exfeatures]


def dumb_classifier(fname, train, train_labels, test, test_labels):
    if os.path.isfile(fname):
        clf = joblib.load(fname)
    else:
        # -- Predict buildings as residential or non-residential.
        clf = RandomForestClassifier(n_estimators=1000, random_state=0,
            class_weight="balanced")
        clf.fit(train, train_labels)
        joblib.dump(clf, fname)

    testpred = clf.predict(test)
    print("LIGHTCURVES: Confusion matrix of individual nights.")
    print(confusion_matrix(test_labels, testpred))

    df = pd.DataFrame(zip(test.index, testpred)) \
        .rename(columns={0: "idx", 1: "pred"})
    df["bbl"] = [lc.coords_bbls[idx] for idx in df["idx"]]
    df_size = df.groupby("bbl").size()
    df = df.groupby("bbl").mean()
    df["N"] = df_size
    df["pred20"] = df["pred"] > 0.2
    df["pred30"] = df["pred"] > 0.3
    df["pred40"] = df["pred"] > 0.4
    df["pred45"] = df["pred"] > 0.45
    df["pred50"] = df["pred"] > 0.5
    df["pred60"] = df["pred"] > 0.6
    df["actual"] = [lc.dd_bbl_bldg2[idx] == 1 for idx in df.index]
    df["correct"] = df["pred50"] == df["actual"]
    print("LIGHTCURVES: Confusion matrix of bbls.")
    print(confusion_matrix(df.actual, df.pred50))

    return df


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


if __name__ == "__main__":

    data, rel_bigoffs = stack_nights_df(lc)
    lclass, rclass = split_bbls_left_right(lc, 0.7)
    train, train_labels, test, test_labels = train_test_datacube(data, lclass, rclass)
    train_exfeatures, test_exfeatures = use_tsfresh(train, test, "./")

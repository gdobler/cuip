from __future__ import print_function

import os
import time
import cPickle
import itertools
import numpy as np
import pandas as pd
import multiprocessing
import matplotlib.pyplot as plt
from collections import Counter
# from fastdtw import fastdtw
from sklearn.cluster import KMeans
from itertools import combinations
from sklearn.externals import joblib
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import silhouette_score, confusion_matrix
from scipy.ndimage.filters import gaussian_filter1d, gaussian_filter
# from tsfresh import extract_features
# from tsfresh.feature_extraction.settings import EfficientFCParameters
plt.style.use("ggplot")


def _start(text):
    """"""
    print("LIGHTCURVES: {}".format(text))
    sys.stdout.flush()
    return time.time()


def _finish(tstart):
    """"""
    print("LIGHTCURVES: Complete ({:.2f}s)".format(time.time() - tstart))


def traintestsplit(lc, train_split=0.7, seed=5):
    """Split sources into training and testing sets, where each bbl is only
    assigned to either the training or test set.
    Args:
        lc (object) - LightCurve object.
        train_split (float) - proportion of training set.
    Returns:
        train (list) - List of sources' idx.
        test (list) - List of sources' idx.
    """
    # -- Print status.
    tstart = _start("Splitting keys into training/testing set.")
    # --
    np.random.seed(seed)
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
    traink = [src for src, bbl in lc.coords_bbls.items() if bbl in trn["bbls"]]
    traink = filter(lambda x: x in lc.coords_cls.keys(), traink)
    trainv = [lc.coords_cls[ii] for ii in traink]
    testk  = [src for src, bbl in lc.coords_bbls.items() if bbl in tst["bbls"]]
    testk  = filter(lambda x: x in lc.coords_cls.keys(), testk)
    testv  = [lc.coords_cls[ii] for ii in testk]
    # -- Print status.
    print("LIGHTCURVES:     Train Set: {} res, {} non-res".format(
        sum(np.array(trainv) == 1), sum(np.array(trainv) != 1)))
    print("LIGHTCURVES:     Test Set: {} res, {} non-res".format(
        sum(np.array(testv) == 1), sum(np.array(testv) != 1)))
    # --
    _finish(tstart)
    return [[traink, trainv], [testk, testv]]


def stack_nights(path):
    """Load detrended ligtcurves and stack if there are no masked values.
    Args:
        path (str) - folder with detrended .npy files.
    Returns:
        data (dict) - dict of {pd.datetime: np.ma.array} pairs.
    """
    # -- Print status.
    tstart = _start("Creating data dict.")
    # -- Collect filenames at provided path.
    dtrend_fnames = filter(lambda x: x.startswith("detrend"), os.listdir(path))
    # -- Stack dtrended lightcurves to
    data = {}
    for fname in dtrend_fnames:
        tmp = np.load(os.path.join(path, fname))
        if (tmp.mask.sum() == 0) & (tmp.shape[0] > 2690):
            date = pd.datetime.strptime(fname[10:-4], "%Y-%m-%d")
            data[date] = tmp
    # --
    _finish(tstart)
    return data


def split_data(data, traink, testk, ndays=74):
    """Stack data and split into train and test sets.
    Args:
        data (dict) - dict of {pd.datetime: np.ma.array} pairs.
        traink (list) - list of train idxs.
        testk (list) - list of test idxs.
    Return:
        train (array) - 2D array, each training sources time series.
        test (array) - 2D array, each test sources time series.
        ndays (int) - number of days split into train/test sets.
    """
    # -- Print status.
    tstart = _start("Splitting data into training/testing sets.")
    # -- Stack the data in numpy cube.
    arr_len = 2692
    stack = np.dstack([arr.data[:arr_len] for arr in data.values()])[:, :, :ndays]
    # -- Split data into training and testing sets.
    train = np.vstack(stack[:, np.array(traink) - 1].T)
    test  = np.vstack(stack[:, np.array(testk) -1].T)
    # -- Define the number of days in the dataset.
    ndays = stack.shape[-1]
    # --
    _finish(tstart)
    return [train, test, ndays]


def downsample(arr, size):
    """"""
    # --
    tstart = _start("Downsampling data.")
    # --
    arr_len = arr.shape[1]
    # -- Define the padding size required.
    pad   = int(np.ceil(arr.shape[1] / float(size)) * size - arr.shape[1])
    # -- Add padding to train/test and take the mean over N values.
    tmp = np.append(arr, np.zeros((arr.shape[0], pad)) * np.NaN, axis=1)
    tmp = np.nanmean(tmp.reshape(-1, size), axis=1).reshape(-1, arr_len / size + 1)
    # --
    _finish(tstart)
    return tmp


def rf_classifier(fpath, train, trainv, test, testv, ndays, bool_label=True,
                  njobs=multiprocessing.cpu_count() - 2, load=True, grad=False,
                  append_coords=False):
    """"""
    # --
    tstart = _start("Training or loading classifier.")
    # -- Training and testing labels for each source.
    train_labels = np.array(trainv * ndays)
    # -- If bool_label, convert to True for residential and False else.
    if bool_label:
        train_labels = (train_labels == 1).astype(int)
    if append_coords:
        coords = np.array([lc.coords[ii] for ii in traink * ndays])
        train = np.concatenate([train, coords], axis=1)
    if load: # -- Load existing classifier.
        clf = joblib.load(fpath)
    else: # -- Fit a classifier.
        clf = RandomForestClassifier(n_estimators=1000, random_state=0,
            class_weight="balanced", n_jobs=njobs, verbose=5)
        if grad:
            clf = GradientBoostingClassifier(learning_rate=0.05,
                n_estimators=200, random_state=0, max_depth=1, verbose=5)
        # params = {"class_weight": ("balanced")}
        # clf = GridSearchCV(clf, params)
        clf.fit(train, train_labels)
        joblib.dump(clf, fpath)
    # --
    _finish(tstart)
    return clf


def votes_comparison(preds, testv, ndays, split=0.5, pprint=True):
    """"""
    # -- Take the mean over each source and bool split.
    votes = np.array(np.split(preds, ndays)).mean(axis=0) > split
    # -- Compare actual labels to voted labels.
    comp = zip(np.array(testv) == 1, votes)
    # -- Calculate overall, residential, and non-residential accuracy.
    acc = sum([k == v for k, v in comp]) * 1. / len(comp)
    res_acc = (sum([k == v for k, v in comp if k == True]) * 1. /
               len(filter(lambda x: x[0] == True, comp)))
    non_res_acc = (sum([k == v for k, v in comp if k == False]) * 1. /
                   len(filter(lambda x: x[0] == False, comp)))
    # -- Print results.
    if pprint:
        print("Split: {}, Acc.: {:.2f}, Res. Acc.: {:.2f}, Non-Res. Acc.: {:.2f}" \
            .format(split, acc * 100, res_acc * 100, non_res_acc * 100))
    # --
    return([acc, res_acc, non_res_acc])


def resample_vals(results):
    """"""
    np.random.seed(5)
    data = []
    # -- Reshape predictions into 2D array.
    vals = results["preds"].reshape(results["ndays"], -1)
    # -- Sample nn days of preds.
    for nn in np.array(range(vals.shape[0])) + 1:
        # -- Do so 1000 times.
        for _ in range(100):
            # -- Randomly select nn idxs,
            idx = np.random.randint(vals.shape[0], size=nn)
            # -- Select rows corresponding to idx.
            val = vals[idx, :]
            # -- Test all 0.05 breaks and save most accurate split.
            max_res = [0, 0, 0]
            split = 0
            # -- Print status.
            print("Sampling {} days ({}/{}).                                 " \
                .format(nn, _ + 1, 100), end="\r")
            sys.stdout.flush()
            for ii in np.array(range(20)) / 20.:
                res = votes_comparison(val.reshape(-1), results["testv"], nn,
                    ii, pprint=False)
                if res[0] > max_res[0]:
                    max_res = res
                    split = ii
            # -- Append values for max accuracy to data.
            acc, res_acc, non_res_acc = max_res
            data.append([nn, split, acc, res_acc, non_res_acc])
    return data


def plot_sampling_result(npy_path):
    """"""
    # -- Load sampling data in DataFrame.
    cols = ["N", "split", "acc", "res_acc", "nres_acc"]
    df = pd.DataFrame(np.load(npy_path), columns=cols)
    # -- Create figure.
    fig, axes = plt.subplots(ncols=2, nrows=2, figsize=(10, 5), sharex=True)
    # -- Plot mean overall accuracy.
    fig.axes[0].plot(df.groupby("N").mean()["acc"])
    fig.axes[0].set_ylabel("Overall Accuracy")
    # -- Plot mean residential accuracy.
    fig.axes[1].plot(df.groupby("N").mean()["res_acc"])
    fig.axes[1].set_ylabel("Residential Accuracy")
    # -- Plot mean non-residential accuracy.
    fig.axes[2].plot(df.groupby("N").mean()["nres_acc"])
    fig.axes[2].set_ylabel("Non-Residential Accuracy")
    fig.axes[2].set_xlabel("Nights")
    # -- Plot the mean best voting split.
    fig.axes[3].plot(df.groupby("N").mean()["split"])
    fig.axes[3].set_ylabel("Best Voting Split")
    fig.axes[3].set_xlabel("Nights")
    plt.tight_layout()
    plt.show()


def fft_pull_weekday_data(data, wd_int, arr_len=2692):
    """"""
    day = np.concatenate([v[:arr_len, :] for k, v in data.items()
                          if k.weekday() == wd_int], axis=0).T
    return day


def fft_weekday_combo(data, wd_int, arr_len=2692):
    """"""
    wd = [v[:arr_len, :] for k, v in data.items() if k.weekday() == wd_int]
    wd = np.stack(wd).T
    nmon = wd.shape[-1]
    inds = np.array(list(itertools.combinations(range(nmon), 2))).T
    cmbs = wd[..., inds]
    xx, yy, zz, ww = cmbs.shape
    combo = np.swapaxes(cmbs, 1, -1).reshape(xx * ww, yy * zz)
    return combo


def fft_split(data, traink, testk, ds_size=False, fft_combo=False):
    """"""
    # -- Concatenate all weekdays into single source vectors.
    days = [fft_pull_weekday_data(data, dd) for dd in range(5)]
    # -- If a downsample size is provided, downsample.
    if type(ds_size) == int:
        days = [downsample(dd, ds_size) for dd in days]
    # -- Calculate ffts.
    ffts = np.array([np.fft.fft(dd) for dd in days])
    # --
    rows, cols = zip(*[fft.shape for fft in ffts])
    # --
    train = np.vstack([fft[np.array(traink) - 1, :min(cols)] for fft in ffts])
    test  = np.vstack([fft[np.array(testk ) - 1, :min(cols)] for fft in ffts])
    return [train, test]


def fft_combination_split(data, traink, trainv, testk, testv, ds_size=False):
    """"""
    print("WARNING: This block of code must be run in chunks.")
    print("WARNING: It will otherwise cause memory errors.")
    sys.stdout.flush()
    # -- Concatenate weekdays into source pair vectors.
    days = [fft_weekday_combo(data, dd) for dd in range(5)]
    # -- If a downsample size is provided, downsample.
    if type(ds_size) == int:
        days = [downsample(dd, ds_size) for dd in days]
    # -- Create fft function.
    def fft(x):
        return np.fft.fft(x)
    # -- Multiprocess fft calculation.
    p = multiprocessing.Pool((multiprocessing.cpu_count() - 2) * 2)
    mon = np.array(p.map(fft, days[0]))
    tue = np.array(p.map(fft, days[1]))
    wed = np.array(p.map(fft, days[2]))
    thu = np.array(p.map(fft, days[3]))
    fri = np.array(p.map(fft, days[4]))
    p.close()
    p.join()
    ffts = [mon, tue, wed, thu, fri]
    # -- Create coord keys for each day.
    keys = [np.array([lc.coords.keys()] * (fft.shape[0] / 4147)).T.ravel()
            for fft in ffts]
    # -- Identify training keys.
    train_keys = [np.isin(key, traink) for key in keys]
    test_keys = [np.isin(key, testk) for key in keys]
    # -- Collect training/testing observations from ffts.
    train_ffts = [fft[key] for fft, key in zip(ffts, train_keys)]
    test_ffts = [fft[key] for fft, key in zip(ffts, test_keys)]
    # -- Collect training/testing labels.
    train_labels = [np.array([trainv] * (fft.shape[0] / len(trainv))).T.ravel()
                    for fft in train_ffts]
    test_labels = [np.array([testv] * (fft.shape[0] / len(testv))).T.ravel()
                   for fft in test_ffts]
    # -- Flatten observations and labels.
    train_ffts   = np.concatenate(train_ffts, axis=0)
    train_labels = np.concatenate(train_labels, axis=0)
    test_ffts    = np.concatenate(test_ffts, axis=0)
    test_labels  = np.concatenate(test_labels, axis=0)
    return [train_ffts, train_labels, test_ffts, test_labels]

   #  In [6]: for nn, fft in zip(["mon", "tue", "wed", "thu", "fri"], train_ffts):
   # ...:     np.save("train_fft_{}.npy".format(nn), fft)
   # train_ffts = [np.load(os.path.join("train_test_npy", fname))
   #               for fname in sorted(filter(lambda x: x.startswith("train"),
   #               os.listdir("train_test_npy")))]
   # train_labels = [np.array([trainv] * (fft.shape[0] / len(trainv))).T.ravel()
   #                 for fft in train_ffts]
   # train_ffts = np.concatenate(train_ffts, axis=0)
   # train_labels = np.concatenate(train_labels, axis=0)
   # train_labels = (train_labels == 1).astype(int)
   # njobs = multiprocessing.cpu_count() - 2
   # clf = RandomForestClassifier(n_estimators=1000, random_state=0,
   #    class_weight="balanced", n_jobs=njobs)
   # test_shapes = [tt.size for tt in test_labels]
   # vals = [0] + list(np.cumsum(test_shapes))
   # test_idx = np.concatenate([np.array([testk] * (fft.size / len(testk))).T.ravel() for fft in test_labels])
   # labs = [(lc.coords_cls[ii] == 1) * 1 for ii in test_idx]
   # df = pd.DataFrame(np.array([test_idx, test_preds, labs]).T, columns=["idx", "preds", "actual"])
   # df["bbl"] = [lc.coords_bbls[ii] for ii in df.idx]
   # tst = df.groupby("idx").mean()
   # tst["vote"] = (tst["preds"] > 0.3) * 1.
   # tst = df.groupby("bbl").mean()
   # tst["vote"] = (tst["preds"] > 0.25) * 1.

def combo_main(lc, ds_size=False, whiten=True):
    """"""
    # --
    tstart = _start("Classify with full ts and fft.")
    # -- Split keys into train and test set,
    [traink, trainv], [testk, testv] = traintestsplit(lc)
    # -- Load data into and 3D numpy array.
    data = stack_nights(os.path.join(lc.path_out, "onsoffs"))
    # -- Split data into train and test (optionally downsample).
    train, test, ndays = split_data(data, traink, testk)
    if ds_size:
        train = downsample(train, ds_size)
        test = downsample(test, ds_size)
    # -- Calculate the fourier transform.
    fft_train = np.fft.fft(train)
    fft_test = np.fft.fft(test)
    # -- Concatenate the data.
    combo_train = np.concatenate([train, fft_train], axis=1)
    combo_test = np.concatenate([test, fft_test], axis=1)
    # --
    if whiten:
        std = np.concatenate([combo_train, combo_test], axis=0).std(axis=0)
        combo_train = combo_train / std
        combo_test = combo_test / std
    # --
    clf = rf_classifier("./whiten_tmp.pkl", combo_train, trainv, combo_test, testv, ndays, load=False)
    preds = clf.predict(combo_test)
    # --
    print("Vote Comparison:")
    for ii in np.array(range(20)) / 20.:
        votes_comparison(preds, testv, ndays, ii)
    print(confusion_matrix((np.array(testv * ndays) == 1).astype(int), preds))
    # --
    _finish(tstart)


def fft_main(lc, ds_size=False):
    """"""
    # --
    tstart = _start("Classify with fft.")
    # -- Split keys into train and test set,
    [traink, trainv], [testk, testv] = traintestsplit(lc)
    # -- Load data into and 3D numpy array.
    data = stack_nights(os.path.join(lc.path_out, "onsoffs"))
    # -- Split data into train and test (optionally downsample).
    if ds_size:
        train, test = fft_split(data, traink, testk, ds_size)
    else:
        train, test = fft_split(data, traink, testk)
    # --
    clf = rf_classifier("./fft_clf.pkl", train, trainv, test, testv, 5, load=False)
    preds = clf.predict(test)
    # -- Check if voting changes the results with various break points.
    print("Vote Comparison:")
    for ii in np.array(range(20)) / 20.:
        votes_comparison(preds, testv, 5, ii)
    print(confusion_matrix((np.array(testv * ndays) == 1).astype(int), preds))
    # --
    _finish(tstart)
    # --
    return({"traink": traink, "trainv": trainv, "testk": testk, "testv": testv,
            "data": data, "train": train, "test": test, "ndays": ndays,
            "clf": clf, "preds": preds})


def main(lc, rf_file, ds_size=False, load=True, whiten=True, grad=False,
         append_coords=False):
    """"""
    # --
    tstart = _start("Classify with full fft.")
    # -- Split keys into train and test set,
    [traink, trainv], [testk, testv] = traintestsplit(lc)
    # -- Load data into and 3D numpy array.
    data = stack_nights(os.path.join(lc.path_out, "onsoffs"))
    # -- Split data into train and test (optionally downsample).
    train, test, ndays = split_data(data, traink, testk)
    # -- Downsample if vals is passed.
    if ds_size:
        train = downsample(train, ds_size)
        test = downsample(test, ds_size)
    # -- Whiten data.
    if whiten:
        std = np.concatenate([train, test], axis=0).std(axis=0)
        train /= std
        test /= std
    if append_coords:
        coords = np.array([lc.coords[ii] for ii in testk * ndays])
        test = np.concatenate([test, coords], axis=1)
    # -- Train random forest and predict.
    clf = rf_classifier(rf_file, train, trainv, test, testv, ndays, load=load,
        grad=grad, append_coords=append_coords)
    preds = clf.predict(test)
    # -- Check if voting changes the results with various break points.
    print("Vote Comparison:")
    for ii in np.array(range(20)) / 20.:
        votes_comparison(preds, testv, ndays, ii)
    print(confusion_matrix((np.array(testv * ndays) == 1).astype(int), preds))
    # --
    _finish(tstart)
    # --
    return({"traink": traink, "trainv": trainv, "testk": testk, "testv": testv,
            "data": data, "train": train, "test": test, "ndays": ndays,
            "clf": clf, "preds": preds})


if __name__ == "__main__":
    rf_file = os.path.join(lc.path_out, "estimators", "rf_clf_src_7030_1000est_20180116.pkl")
    results = main(lc, rf_file, False, True)

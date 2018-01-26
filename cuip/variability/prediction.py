from __future__ import print_function

import os
import sys
import time
import numpy as np
import pandas as pd
import multiprocessing
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.externals import joblib
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier

plt.style.use("ggplot")


def _start(text, same_line=False):
    """Print status.
    Args:
        text (str) - text to print.
        same_line (bool) - print with or without return.
    Returns:
        time.time() - time at start of process.
    """
    string = "LIGHTCURVES: {}                                                  "
    if same_line:
        print(string.format(text), end="\r")
    else:
        print(string.format(text))
    sys.stdout.flush()
    return time.time()


def _finish(tstart):
    """Print elapsed time from start of process.
    Args:
        tstart - time.time()
    """
    string = "LIGHTCURVES: Complete ({:.2f}s)                                  "
    print(string.format(time.time() - tstart))


def load_data(lc, path, arr_len=2692):
    """Load detrended lightcurves, ons, and offs.
    Args:
        path (str) - folder with detrended .npy files.
        arr_len (int) - array length cut-off.
    Returns:
        data[:, 0] (array) - dates corresponding to the data.
        crds (array) - Source idx [1-4147]
        lcs (array) - Stacked array of lightcurves.
        ons (array) - Stacked array of ons.
        offs (array) - Stacked array of offs.
    """
    # -- Print status.
    tstart = _start("Loading data from {}.".format(path))
    # -- Collect detrended file names
    fnames = filter(lambda x: x.startswith("detrended_"), os.listdir(path))
    # -- For each file in fnames load the corresponding files.
    data = []
    ll   = len(fnames)
    for ii, fname in enumerate(sorted(fnames)):
        # -- Print loading status.
        _ = _start("Loading {} ({}/{})".format(fname, ii + 1, ll), True)
        # -- Load detrended lightcurve file.
        lcs = np.load(os.path.join(path, fname))
        # -- Check if the lightcurves are complete and load data.
        if (lcs.mask.sum() == 0) & (lcs.shape[0] > 2690):
            dd   = fname[10:-4]
            date = pd.datetime.strptime(dd, "%Y-%m-%d")
            ons  = np.load(os.path.join(path, "good_ons_{}.npy".format(dd)))
            offs = np.load(os.path.join(path, "good_offs_{}.npy".format(dd)))
            data.append([date, lcs[:arr_len], ons[:arr_len], offs[:arr_len]])
    # -- Stack data.
    data = np.array(data)
    lcs  = np.hstack(data[:, 1]).T
    ons  = np.hstack(data[:, 2]).T
    offs = np.hstack(data[:, 3]).T
    crds = np.array(lc.coords.keys() * len(data[:, 0]))
    # -- Print status & return data.
    _finish(tstart)
    return [data[:, 0], crds, lcs, ons, offs]


def bbl_split(lc, crds, data, train_size=0.7, seed=1, excl_bbl=False):
    """Split coordinates and data into train/test split, keeping sources from
    the same bbl in the same set.
    Args:
        lc (obj) - LightCurves object.
        crds (array) - Array of source idxs [1-4147].
        data (array) - Stacked array of lcs, ons, or offs to be split.
        train_size (float) - Training set proportion to approximate.
        seed (int) - Random seed for splitting.
        excl_bbl (bool/array) - Can pass array of BBLs to exclude.
    Returns:
        (list)
    """
    # -- Print status.
    tstart = _start("Splitting data by bbls")
    # -- Set random seed.
    np.random.seed(seed)
    # -- Only use coords with corresponding class.
    crdC = crds[np.isin(crds, np.array(lc.coords_cls.keys()))]
    # -- Create list of labels.
    labels = (np.array([lc.coords_cls[crd] for crd in crdC]) == 1) * 1
    # -- Shuffle list of (bbl, srcs) pairs.
    crdD = {crd: lc.coords_bbls[crd] for crd in crdC}
    bblN = np.array(Counter(crdD.values()).items())
    np.random.shuffle(bblN)
    # -- Remove bbl from sample if excl_bbl provided and in list of bbls.
    if np.isin(excl_bbl, bblN[:, 0]).sum() > 0:
        _ = _start("Excluding sources from BBL: {}".format(excl_bbl))
        bblN = bblN[~np.isin(bblN[:, 0], excl_bbl)]
    # -- Find index to split bbls
    splt = np.argmax(1. * np.cumsum(bblN[:, 1]) / sum(bblN[:, 1]) > train_size)
    trn_bbls = bblN[:splt, 0]
    tst_bbls = bblN[splt:, 0]
    # -- Find coords that correspond to training and testing bbls.
    trn = np.array([cc for cc, bb in lc.coords_bbls.items() if bb in trn_bbls])
    tst = np.array([cc for cc, bb in lc.coords_bbls.items() if bb in tst_bbls])
    # -- Split the coordinates into training and testing.
    trn_coords = crds[np.isin(crds, trn)]
    tst_coords = crds[np.isin(crds, tst)]
    # -- Split your data into training and testing data.
    trn_data = data[np.isin(crds, trn)]
    tst_data = data[np.isin(crds, tst)]
    # -- Split labels into training and testing.
    trn_labs = labels[np.isin(crdC, trn)]
    tst_labs = labels[np.isin(crdC, tst)]
    # -- Print status.
    _finish(tstart)
    _ = _start("Train N: {}, Test N: {}".format(trn_labs.size, tst_labs.size))
    actual_split = 1. * trn_labs.size / (trn_labs.size + tst_labs.size)
    _ = _start("Actual Split: {:.2f}".format(actual_split))
    return [trn_coords, trn_data, trn_labs, tst_coords, tst_data, tst_labs]


def score(preds, tst_labs, conf=True, split=False):
    """Print accuracy scores.
    Args:
        preds (array) - Predicted labels.
        tst_labs (array) - Actual labels.
        conf (bool) - Print out confusion matrix.
        split (float) - Split value.
    """
    # -- Calculate accuracy scores.
    cf     = confusion_matrix(preds, tst_labs)
    acc    = 100. * (cf[0][0] + cf[1][1]) / cf.sum()
    r_acc  = 100. * cf[1][1] / (cf[1].sum() + (cf[1].sum() == 0))
    nr_acc = 100. * cf[0][0] / (cf[0].sum() + (cf[0].sum() == 0))
    # -- Define text to print.
    txt = "Acc: {:.2f}, Non-Res Acc: {:.2f}, Res Acc: {:.2f}" \
        .format(acc, r_acc, nr_acc)
    if type(split) != bool: # -- If a split value is supplied, add to existing text.
        txt = "Split: {:.2f} ".format(split) + txt
    if conf == True:   # -- If chosen, print confusion matrix.
        _ = _start("Confusion matrix: {}".format([list(cf[0]), list(cf[1])]))
    # -- Print scores.
    _ = _start(txt)


def votescore(preds, tst_labs, ndays=74):
    """Print scores predictions.
    Args:
        preds (array) - predicted labels.
        tst_labs (array) - actual labels.
    """
    # -- Print scores for individual sources
    score(preds, tst_labs)
    # -- Calculate results for voting on sources accross days.
    _ = _start("Vote Comparison")
    # -- Take the mean prediction across days for each source.
    src_mn = preds.reshape(ndays, preds.size / ndays).mean(0)
    for ii in np.array(range(20)) / 20.:
        votes = (src_mn > ii).astype(int)
        score(votes, tst_labs[:len(votes)], False, ii)


def preprocess(trn, trn_data, tst, tst_data, whiten=True, append_coords=True):
    """Options to whiten and append coordinates to training/testing data.
    Args:
        trn (array) - Training coords.
        trn_data (array) - Training feature vector.
        tst (array) - Testing coords.
        tst_data (array) - Testing feature vector.
        whiten (bool) - Whiten feature vectors.
        append_coords (bool) - Append coordinates to feature vectors.
    Returns:
        trn_data (array) - preprocessed training feature vectors.
        tst_data (array) - preprocessed testing feature vectors.
    """
    # -- Flow controls to modify the training/testing data.
    if whiten: # -- Whiten the data. Calculate the std over tst and trn.
        std = np.concatenate([trn_data, tst_data], axis=0).std(axis=0)
        trn_data /= std
        tst_data /= std
    if append_coords: # -- Append the srcs coordinates to the vectors.
        trn_coords = np.array([lc.coords[crd] for crd in trn])
        tst_coords = np.array([lc.coords[crd] for crd in tst])
        trn_data   = np.concatenate([trn_data, trn_coords], axis=1)
        tst_data   = np.concatenate([tst_data, tst_coords], axis=1)
    return [trn_data, tst_data]


def plot_feature_importance(clf, ordinal=False):
    """Plot the feature importance of the feature vector, in either real values
    or ranked order.
    Args:
        clf (obj) - sklearn object with .feature_importances_ attribute.
        ordinal (bool) - Plot real values or rank.
    """
    fimp = clf.feature_importances_
    fig, ax = plt.subplots(figsize=(12, 4))
    if ordinal:
        ax.scatter(fimp.argsort(), range(len(fimp)), s=5)
        ax.set_yticks([])
        ax.set_ylabel("Ordinal Feature Importance")
        ax.set_title("Ordinal Feature Importance In RF Classification")
    else:
        ax.scatter(range(len(fimp)), fimp, s=5)
        ax.set_ylabel("Feature Importance")
        ax.set_title("Feature Importance In RF Classification")
    ax.set_xticks(np.array(range(8)) * 360)
    ax.set_xticklabels([21, 22, 23, 24, 1, 2, 3, 4])
    ax.set_xlabel("Hour")
    ax.set_xlim(0, len(fimp) + 10)
    plt.show()


def recursive_rf(lc, path, whiten, append_coords, verbose=0, iters=10,
    excl_bbl=False):
    """"""
    days, crds, lcs, ons, offs = load_data(lc, path)
    # -- Train/test split keeping BBLs in the same set.
    trn, trn_data, trn_labs, val, val_data, val_labs = bbl_split(
        lc, crds, lcs, train_size=0.8, seed=ii, excl_bbl=excl_bbl)
    # -- Whiten and append coords if chosen.
    trn_data, tst_data = preprocess(trn, trn_data, tst, tst_data,
        whiten=whiten, append_coords=append_coords)
    # --


    # -- Define random forest classifier.
    clf = RandomForestClassifier(n_estimators=100, random_state=0,
        class_weight="balanced", n_jobs=multiprocessing.cpu_count() - 2,
        verbose=verbose)
    # -- Fit random forest classifier.
    tstart = _start("Training RF {}/{}".format(ii, iters))
    clf.fit(trn_data, trn_labs)
    _finish(tstart)
    # -- Make predictions from the test set.
    preds = clf.predict(tst_data)


def main(lc, path, inpath=False, outpath=False, whiten=True, append_coords=True,
    verbose=0, iters=1, excl_bbl=False):
    """"""
    days, crds, lcs, ons, offs = load_data(lc, path)
    for ii in range(1, iters + 1):
        # -- Train/test split keeping BBLs in the same set.
        trn, trn_data, trn_labs, tst, tst_data, tst_labs = bbl_split(
            lc, crds, lcs, seed=ii, excl_bbl=excl_bbl)
        # -- Whiten and append coords if chosen.
        trn_data, tst_data = preprocess(trn, trn_data, tst, tst_data,
            whiten=whiten, append_coords=append_coords)
        # -- Load classifier if it exists.
        if type(inpath) == str:
            clf = joblib.load(fpath)
        else: # -- Create and save classifier.
            clf = RandomForestClassifier(n_estimators=100, random_state=0,
                class_weight="balanced", n_jobs=multiprocessing.cpu_count() - 2,
                verbose=verbose)
            # -- Fit random forest classifier.
            tstart = _start("Training RF {}/{}".format(ii, iters))
            clf.fit(trn_data, trn_labs)
            _finish(tstart)
            if type(outpath) == str: # -- Save classifier to file.
                if iters > 1:
                    joblib.dump(clf, outpath[:-4] + "_{}.pkl".format(ii))
                else:
                    joblib.dump(clf, outpath)
        # -- Make predictions from the test set.
        preds = clf.predict(tst_data)
        # -- Calculate accuracy scores and voting scores.
        votescore(preds, tst_labs)
    return [clf, trn, trn_data, trn_labs, tst, tst_data, tst_labs]


def main_excl_bbls(lc, path, greaterN=1, popN=10, iters=1):
    """Excluding bbls for training RF classifiers.
    Args:
        lc (obj) - LightCurves object.
        path (str) - path to data.
        greaterN (int) - exclude bbls with greater than N sources.
        popN (int) - exclude top N bbls and iteratively train an RF.
        iters (int)
    """
    # -- Remove bbl.
    bblN = np.array(Counter(lc.coords_bbls.values()).items())
    bblN = bblN[bblN[:, 1].argsort()]
    # -- If a greaterN value is passed. Exclude BBLs.
    if greaterN > 1:
        main(lc, path, excl_bbl=bblN[bblN[:, 1] > greaterN][:, 0], iters=iters)
    else: # -- If not, iteratively train RFs removing the top N bbl.
        for ii in range(1, popN+1):
            main(lc, path, excl_bbl=bblN[:, 0][-ii])


if __name__ == "__main__":
    results = main(lc, os.path.join(lc.path_out, "onsoffs"), iters=10,
        outpath="tmp.pkl")

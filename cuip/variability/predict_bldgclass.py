from __future__ import print_function, absolute_import, division

import numbers
import numpy as np
import pandas as pd
from fastdtw import fastdtw
from sklearn.cluster import KMeans
from collections import defaultdict
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import silhouette_score, confusion_matrix
from scipy.ndimage.filters import gaussian_filter1d, gaussian_filter


def stack_nights():
    """"""

    # -- Sample of good nights (clean data collection).
    nday = [4, 5, 8, 15, 20, 21, 25]
    ndates = [pd.datetime(2013, 11, day).date() for day in nday]

    data = []
    # -- Stack multiple nights.
    for date in ndates:
        lc.loadnight(date)
        # -- Pull nightly lightcurves for same range of timesteps.
        lightc = lc.src_lightc[:2878, :]
        # -- Detrend (i.e., subtract the median).
        dlightc = (lightc.T - np.median(lightc, axis=1)).T
        # -- Min max each source, and append lightcurve array to data.
        data.append(MinMaxScaler().fit_transform(dlightc))
    data = np.array(data).T
    # -- Take the mean of each source over all nights and min max the result.
    mean_gaus = gaussian_filter(data.mean(axis=2), [0, 10])
    mean_gaus_mm = MinMaxScaler().fit_transform(mean_gaus.T).T

    # -- Pull source indices for higher level classifications (excl. indust).
    idxs = [zip(*filter(lambda x: x[1] == ii, lc.coords_cls.items()))[0]
        for ii in [1, 2]]

    # -- Select class timeseries, take the mean, and min max.
    ts_bldg = [mean_gaus[np.array(idx) - 1].mean(axis=0) for idx in idxs]
    ts_mm = [MinMaxScaler().fit_transform(ts.reshape(-1, 1)) for ts in ts_bldg]

    # -- Plot filter timeseries for each class and some example timeseries.
    for ts, lab in zip(ts_mm, ["Res", "Com"]):
        plt.plot(ts, label=lab, lw=2)
    for ii in idxs[0][:5]:
        plt.plot(mean_over_nights_gaus[ii], c="r", ls="dashed", lw=0.5)
    for ii in idxs[1][:5]:
        plt.plot(mean_over_nights_gaus[ii], c="b", ls="dashed", lw=0.5)
    plt.legend(loc=1)
    plt.show()

    return [mean_gaus_mm, ts_mm, mean_over_nights_gaus, idxs]


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
    from tsfresh import extract_features
    from tsfresh.feature_extraction.settings import EfficientFCParameters

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

"""
@author: Jun Mo
"""


"""
Run PCA on the full dataset for Tree classifier
"""

import os
import sys
import errno
import shutil
import tempfile

import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from user_sensors import SensorUserPopulation
from defend_gait import generate_protection_noise



def main_run():
    resample_users = True


    if resample_users:
        sensors_loc = "./../uci_har_full_dataset.csv"
        n_feat = 562

        a = SensorUserPopulation(sensors_loc, n_feat)

        a.normalize_data()
        a.split_user_data(0.3)

        X = a.get_train_sets


    u = a.labels[0]
    target_data, other_data = a.get_train_sets(u, concatenate=False)
    data = np.concatenate([target_data, other_data])
    n0 = len(target_data)
    n1 = len(other_data)
    y = np.concatenate([np.zeros(n0), np.ones(n1)])
    target_names = [0, 1]

    pca = PCA(n_components = 2)
    pca.fit(data)
    X_r = pca.transform(data)

    plt.figure()
    colors = ['red', 'navy']
    lw = 2

    for color, i, target_name in zip(colors, [0,1], target_names):
        plt.scatter(X_r[y == i, 0], X_r[y == i, 1], color=color, alpha=.8, lw=lw, label=target_name)


    plt.legend(loc='best', shadow=False, scatterpoints=1)
    plt.title('PCA for User 1 vs Rest')
    plt.show()


if __name__ == "__main__":
    main_run()

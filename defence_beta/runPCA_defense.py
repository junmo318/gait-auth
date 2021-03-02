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
from mpl_toolkits.mplot3d import Axes3D


sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from user_sensors import SensorUserPopulation
from defend_gait import generate_protection_noise, only_noise



def main_run():
    resample_users = True


    if resample_users:
        sensors_loc = "./../uci_har_full_dataset.csv"
        n_feat = 562

        a = SensorUserPopulation(sensors_loc, n_feat)

        a.normalize_data()
        a.split_user_data(0.3)


    u = 1
    target_data, other_data = a.get_train_sets(u, concatenate=False)
    target_data, other_data2, n2 = generate_protection_noise(target_data, other_data, 0.5)




    data = np.concatenate([target_data, other_data2])

    n0 = len(target_data)
    n1 = len(other_data) - n2
    print(n2)
    print(len(other_data2))


    y = np.concatenate([np.zeros(n0), np.ones(n1), np.full((n2), 2)])
    target_names = [0, 1]
    # print(y)

    pca = PCA(n_components = 3)
    pca.fit(data)
    X_r = pca.transform(data)

    # plt.figure()
    colors = ['r.', 'b.', 'g.']
    # lw = 2
    #
    # # for color, i, target_name in zip(colors, [0,1], target_names):
    # #     plt.scatter(X_r[y == i, 0], X_r[y == i, 1], color=color, alpha=.8, lw=lw, label=target_name)
    # # print(y)
    # n = 0
    # m = 0
    # for point, label in zip(X_r, y):
    #     if (label == 0):
    #         plt.plot(point[0], point[1], colors[int(label)]);
    #     elif (label == 1):
    #         plt.plot(point[0], point[1], colors[int(label)]);
    #     elif (label == 2):
    #         plt.plot(point[0], point[1], colors[int(label)]);
    #
    #
    #
    #
    # plt.legend(loc='best', shadow=False, scatterpoints=1)
    # plt.title('PCA for User 1 vs Rest')
    # plt.show()

    fig = plt.figure(1, figsize=(4, 3))
    plt.clf()
    ax = Axes3D(fig, elev=-150, azim=110)

    for point, label in zip(X_r, y):
        if (label == 0):
            plt.plot(point[0], point[1], point[2], colors[int(label)]);
        elif (label == 1):
            plt.plot(point[0], point[1], point[2], colors[int(label)]);
        elif (label == 2):
            plt.plot(point[0], point[1], point[2], colors[int(label)]);

    # plt.show()
    # print(X_r)
    # target_data, n3 = only_noise(target_data, other_data, 0.5)




if __name__ == "__main__":
    main_run()

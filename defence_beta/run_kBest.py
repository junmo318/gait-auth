import os
import sys
import errno
import shutil
import tempfile

import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from mpl_toolkits.mplot3d import Axes3D
from sklearn.feature_selection import SelectKBest, chi2


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

    u = 1
    target_data, other_data = a.get_train_sets(u, concatenate=False)
    target_data, other_data, n2 = generate_protection_noise(target_data, other_data, 0.5)


    data = np.concatenate([target_data, other_data])

    n0 = len(target_data)
    n1 = len(other_data) - n2

    y = np.concatenate([np.zeros(n0), np.ones(n1), np.full((n2), 2)])

    # y = np.concatenate([np.full((n0), 'red'), np.full((n1), 'blue'), np.full((n2), 'green')])

    colors = ['r.', 'b.', 'g.']


    X = SelectKBest(chi2, k=3).fit_transform(data, y);
    # plt.figure()

    fig = plt.figure(1, figsize=(4, 3))
    plt.clf()
    ax = Axes3D(fig, elev=-150, azim=110)


    # for point, label in zip(X, y):
    #     plt.plot(point[0], point[1], point[2], colors[int(label)]);

    # plt.scatter(X[:,0], X[:,1], X[:,2], c = y)
    for point, label in zip(X, y):
        if (label == 0):
            plt.plot(point[0], point[1], point[2], colors[int(label)], label = "Genuine");
        elif (label == 1):
            plt.plot(point[0], point[1], colors[int(label)], label = "Impostor");
        elif (label == 2):
            plt.plot(point[0], point[1], colors[int(label)], label = "Synthetic");


    red_patch = mpatches.Patch(color='red', label='Genuine')
    green_patch = mpatches.Patch(color='blue', label='Genuine')
    blue_patch = mpatches.Patch(color='green', label='Genuine')


    plt.legend(handles=[red_patch, green_patch, blue_patch], loc = 'upper left')
    plt.title("PCA for User " +  str(u) + " 2D graph")
    plt.show()














main_run();

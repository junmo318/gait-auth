#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 11 10:53:48 2019

@author: zha197
"""

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from math import ceil
import numpy as np
import pandas as pd


class SensorUserPopulation:
    # Initialize the population prameters
    def __init__(self, sensor_data, n_feat):
        # Store the population statistics within the class
        self.data_loc = sensor_data
        self.data = pd.read_csv(self.data_loc, index_col=[0])

        self.n_feat = n_feat
        self.labels = self.data['subject_id']
        self.data.drop('subject_id', axis=1, inplace=True)

        users = {}
        for l in set(self.labels):
            dat = self.data[self.labels == l]
            users[l] = UserData(l, dat, dat.shape[0], n_feat)
        self.users = users
        self.n_users = len(set(self.labels))

    def get_user(self, i=None):
        if i is None:
            return self.users
        else:
            return self.users[i]

    def get_feature_means(self, i=None):
        if i is None:
            return np.mean([self.users[h].get_feature_means() for
                            h in range(self.n_samples)])
        else:
            return self.users[i].get_feature_means()

    def get_feature_stds(self, i=None):
        if i is None:
            return self.users
        else:
            return self.users[i].get_feature_stds()

    def normalize_data(self):
        self.scaler = MinMaxScaler()
        # Read the user's data to build the scaler
        for u, u_data in self.users.items():
            self.scaler.partial_fit(u_data.get_user_data())
        # Apply the common scaler on every user's data
        for u, u_data in self.users.items():
            self.users[u].normalize_data(self.scaler)

    def split_user_data(self, test_split=0.2):
        self.test_split = test_split
        for u, u_data in self.users.items():
            u_data.split_user_data(test_split)

    def get_train_sets(self, training_user, concatenate=True):
        train_user = self.get_user(training_user)
        size = int(ceil((1.0-self.test_split)*train_user.get_user_data(
                    count=True) / self.n_users))
        pos_d = train_user.get_train_data()
        data_arr = []
        for u_n, user_data in self.users.items():
            if u_n != training_user:
                data = user_data.get_train_data()
                data_arr.append(data[np.random.choice(data.shape[0], size), :])

        neg_d = np.concatenate(data_arr)

        if concatenate:
            label = np.concatenate([np.ones(pos_d.shape[0]),
                                    np.zeros(neg_d.shape[0])])
            return np.concatenate([pos_d, neg_d]), label
        else:
            return pos_d, neg_d

    def get_test_sets(self, target_user, concatenate=True):
        test_user = self.get_user(target_user)
        size = int(ceil((self.test_split)*test_user.get_user_data(
                    count=True) / self.n_users))
        pos_d = test_user.get_test_data()
        data_arr = []
        for u_n, user_data in self.users.items():
            if u_n != target_user:
                data = user_data.get_test_data()
                data_arr.append(data[np.random.choice(data.shape[0], size), :])

        neg_d = np.concatenate(data_arr)

        if concatenate:
            label = np.concatenate([np.ones(pos_d.shape[0]),
                                    np.zeros(neg_d.shape[0])])
            return np.concatenate([pos_d, neg_d]), label
        else:
            return pos_d, neg_d


class UserData:
    # Initialize the user distributions
    def __init__(self, label, features, n_samples, n_dim):
        assert features.shape == (n_samples, n_dim)
        self.label = label
        self.features = features
        self.n_samp = n_samples
        self.n_dim = n_dim

    def __repr__(self):
        return str(self.__class__) + ": " + str(self.__dict__)

    def get_user_data(self, count=False):
        if count:
            return len(self.features)
        else:
            return self.features

    def get_feature_means(self):
        return np.mean(self.features, axis=1)

    def get_feature_stds(self):
        return np.std(self.features, axis=1)

    def normalize_data(self, scaler):
        self.unnormalize = self.features
        self.features = scaler.transform(self.features)

    def split_user_data(self, test_size):
        self.train, self.test = train_test_split(self.features,
                                                 test_size=test_size)

    def get_train_data(self):
        return self.train

    def get_test_data(self):
        return self.test


if __name__ == "__main__":
    print("Run")
    sensors_loc = "./uci_har_full_dataset.csv"

    n_feat = 562

    user_sensors = SensorUserPopulation(sensors_loc, n_feat)

    print(user_sensors)

    print([u_data.get_user_data() for u, u_data in user_sensors.users.items() if u == 1])
    user_sensors.normalize_data()
    print([u_data.get_user_data() for u, u_data in user_sensors.users.items() if u == 1])
    user_sensors.split_user_data(0.3)
#    user_faces.get_train_sets(1, concatenate=False)
    for u in user_sensors.users.keys():
        print(u)
        print(list(map(len, user_sensors.get_train_sets(u, concatenate=False))))
        print(list(map(len, user_sensors.get_test_sets(u, concatenate=False))))
        print('')

    data_len = []
    for u in user_sensors.users.keys():
        data_len.append(user_sensors.users[u].get_user_data(count=True))
    print(np.mean(data_len), np.std(data_len))

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Benjamin Zhao

Useful script if dump of normalized data is required.
"""

import pandas as pd
from user_sensors import SensorUserPopulation


if __name__ == "__main__":
    print("Run")
    sensors_loc = "./uci_har_full_dataset.csv"

    n_feat = 562

    user_sensors = SensorUserPopulation(sensors_loc, n_feat)

    print(user_sensors)

    print([u_data.get_user_data() for u, u_data in user_sensors.users.items()
           if u == user_sensors.labels[0]])
    user_sensors.normalize_data()
    print([u_data.get_user_data() for u, u_data in user_sensors.users.items()
           if u == user_sensors.labels[0]])
    user_sensors.split_user_data(0.3)

    a = pd.DataFrame(user_sensors.scaler.transform(user_sensors.data))
    a['user'] = user_sensors.labels

    a.to_csv('./sensor_data_dump.csv')

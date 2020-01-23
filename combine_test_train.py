#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
@author: Benjamin Zhao

The purpose of this script is to combine the training and testing files of the
UCI HAR Dataset, as we do not seek to preserve the chronological train/test
split of user recorded HAR activities.

Place this file in the same directory as the UCI HAR Dataset.

The UCI HAR Dataset is avaliable for download here:
https://archive.ics.uci.edu/ml/datasets/human+activity+recognition+using+smartphones
"""
import pandas as pd
from os.path import join

files = {'subject_id': "subject_{}.txt",
         'subject_features': "X_{}.txt",
         'subject_action': "y_{}.txt"}

combined_files = {}

for file_type, file_name in files.items():
    for split in ["test", "train"]:
        addr = join("./{}/".format(split), file_name.format(split))
        print(addr)
        data = pd.read_csv(addr, delim_whitespace=True, header=None)

        data.columns = range(len(data.columns))

        if file_type not in combined_files:
            combined_files[file_type] = data
        else:
            combined_files[file_type] = pd.concat([combined_files[file_type],
                                                  data], ignore_index=True)

# for saving into combined files of the original format
for file_type, file_data in combined_files.items():
    file_data.to_csv('./' + file_type + '.csv', header=False, index=False)

# read the feature names from the given file.
feature_names = pd.read_csv("./features.txt", delim_whitespace=True,
                            header=None)[1].values

single_df = combined_files['subject_features']
single_df.columns = feature_names
# combine subject id and subject actiontype into the dataframe
single_df['subject_id'] = combined_files['subject_id'][0].values
single_df['subject_action'] = combined_files['subject_action'][0].values
# reorder_columns
single_df = single_df[['subject_id', 'subject_action'] + list(feature_names)]

single_df.to_csv("uci_har_full_dataset.csv")

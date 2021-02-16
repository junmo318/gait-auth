import os
import re

directory = './smartwatchdata'

n_feat = 125
users = {}

filenames = {}
for filename in os.listdir(directory):
    res = re.sub("\D", "", filename)
    print(res)
    filenames[filename] = res

# with open('./mydata.txt', 'w') as outfile:
#     for fname in filenames:
#         with open(fname) as infile:
#             for line in infile:
#                 outfile.write(line)

combined_files = {}

# for file_type, file_name in filenames.items():

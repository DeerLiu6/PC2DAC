import json
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from matplotlib import rcParams

config = {
    "font.family":'serif',
    # "font.size": 20,
    "mathtext.fontset":'stix',
    "font.serif": ['SimSun'],
}
rcParams.update(config)

file_path1 = 'logs/peak_age_threshold/peak_age_threshold_15.json'
file_path2 = 'logs/peak_age_threshold/peak_age_threshold_20.json'
file_path3 = 'logs/peak_age_threshold/peak_age_threshold_25.json'
file_path4 = 'logs/peak_age_threshold/peak_age_threshold_30.json'
file_path5 = 'logs/peak_age_threshold/peak_age_threshold_40.json'


with open(file_path1, 'r') as file:
    data1 = json.load(file)
with open(file_path2, 'r') as file:
    data2 = json.load(file)
with open(file_path3, 'r') as file:
    data3 = json.load(file)
with open(file_path4, 'r') as file:
    data4 = json.load(file)
with open(file_path5, 'r') as file:
    data5 = json.load(file)


time1 = [entry[0] for entry in data1]
step_nums1 = [entry[1] for entry in data1]
vals1 = [entry[2] for entry in data1]

time2 = [entry[0] for entry in data2]
step_nums2 = [entry[1] for entry in data2]
vals2 = [entry[2] for entry in data2]

time3 = [entry[0] for entry in data3]
step_nums3 = [entry[1] for entry in data3]
vals3 = [entry[2] for entry in data3]

time4 = [entry[0] for entry in data4]
step_nums4 = [entry[1] for entry in data4]
vals4 = [entry[2] for entry in data4]

time5 = [entry[0] for entry in data5]
step_nums5 = [entry[1] for entry in data5]
vals5 = [entry[2] for entry in data5]


marker_interval = 50
plt.plot(step_nums1[::marker_interval], vals1[::marker_interval], linestyle='-', marker='o')
plt.plot(step_nums2[::marker_interval], vals2[::marker_interval], linestyle='--', marker='x')
plt.plot(step_nums3[::marker_interval], vals3[::marker_interval], linestyle='-.', marker='s')
plt.plot(step_nums4[::marker_interval], vals4[::marker_interval], linestyle=':', marker='d')
plt.plot(step_nums5[::marker_interval], vals5[::marker_interval], linestyle='-', marker='^')

plt.ylim(0, 100)
plt.xlabel('Step Number')
plt.ylabel('Peak Ages')
plt.legend(['15', '20', '25', '30', '40'],loc='upper left')

plt.show()

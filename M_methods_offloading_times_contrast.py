import json
import numpy as np
import matplotlib.pyplot as plt


file_path = 'logs/offloading_times/offloading_times_pc2dac.json'

with open(file_path, 'r') as file:
    data1 = json.load(file)

vals1 = [entry[2] for entry in data1]

x = ['0-999', '1000-1999', '2000-2999', '3000-3999', '4000-4999',
     '5000-5999', '6000-6999', '7000-7999', '8000-8999', '9000-9999']
off_self = []
off_d2d = []
off_UAV = []
off_server = []
for i in range(10):
    off_self.append(vals1[i][0])
    off_d2d.append(vals1[i][1])
    off_UAV.append(vals1[i][2])
    off_server.append(vals1[i][3])

print(off_self)
print(off_d2d)
print(off_UAV)
print(off_server)

bar_width = 0.5
x_positions = np.arange(len(x))

plt.bar(x_positions, off_self, bar_width, label='off_self', hatch='++', edgecolor='black', color='#FF9999')
plt.bar(x_positions, off_d2d, bar_width, bottom=off_self, label='off_d2d', hatch='..', edgecolor='black', color='#FFCC99')
plt.bar(x_positions, off_UAV, bar_width, bottom=np.array(off_self)+np.array(off_d2d), label='off_UAV', hatch='xx', edgecolor='black', color='#FFFF99')
plt.bar(x_positions, off_server, bar_width, bottom=np.array(off_self)+np.array(off_d2d)+np.array(off_UAV), label='off_server', hatch='\\\\', edgecolor='black', color='#CCFF99')

plt.ylabel('Total Number of Offloading')
plt.xlabel('Step Number')
plt.xticks(x_positions, x, size=6)
plt.legend(loc ='upper left', ncol=1, prop={'size': 8})

plt.ylim(0, 30000)
plt.yticks(np.arange(0, 30001, 5000),size=6)
plt.grid(axis='y', linestyle='--')      
plt.title('Offloading Times of PC2DAC Method')
plt.tight_layout()


plt.show()
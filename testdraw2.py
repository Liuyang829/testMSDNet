import matplotlib.pyplot as plt
import numpy as np

f = np.loadtxt("dynamic_cifar100.txt")
acc = f[:, 0]
flops = f[:, 1]
# print(flops)
for i in range(len(flops)):
    flops[i]=flops[i]/100000000
print(flops)
print(acc)
plt.plot(flops, acc, color='black', linestyle='-',label='MSDNet with dynamic evaluation')
font1 = {'family': 'Consolas',
         'weight': 'light',
         'size': 14,
         }
plt.rcParams['font.sans-serif'] = ['Consolas']
plt.title('Budgeted batch classification on CIFAR-100', font1)
plt.xlabel('average budget(in MUL-ADD)', font1)
plt.ylabel('accuracy(%)', font1)
plt.text(0.54, 55, r'x 10$^{8}$')
plt.ylim(ymin=56)
plt.xlim(xmin=0,xmax=0.6)
plt.tick_params(axis='x', direction='in')  # 将x周的刻度线方向设置向内
plt.tick_params(axis='y', direction='in')  # 将x周的刻度线方向设置向内
plt.grid()
plt.legend(loc=4, prop=font1)
plt.show()

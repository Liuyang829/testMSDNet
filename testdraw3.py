import matplotlib.pyplot as plt
import numpy as np

f1 = np.loadtxt("dynamic_imagenet_step=4.txt")
f2 = np.loadtxt("dynamic_imagenet_step=7.txt")
acc1 = f1[:, 0]
flops1 = f1[:, 1]
acc2=f2[:,0]
flops2=f2[:,1]
# print(flops)
for i in range(len(flops1)):
    flops1[i] = flops1[i] / 1000000000
    flops2[i] = flops2[i] / 1000000000
# print(flops1)
# print(acc1)
plt.plot(flops1, acc1, color='blue', linestyle='-', label='MSDNet with dynamic evaluation(step=4)')
plt.plot(flops2, acc2, color='red', linestyle='-', label='MSDNet with dynamic evaluation(step=7)')
font1 = {'family': 'Consolas',
         'weight': 'light',
         'size': 14,
         }
plt.rcParams['font.sans-serif'] = ['Consolas']
plt.title('Budgeted batch classification on ImageNet', font1)
plt.xlabel('average budget(in MUL-ADD)', font1)
plt.ylabel('accuracy(%)', font1)
plt.text(2.6, 55, r'x 10$^{9}$')
plt.ylim(ymin=56)
plt.xlim(xmin=0)
plt.tick_params(axis='x', direction='in')  # 将x周的刻度线方向设置向内
plt.tick_params(axis='y', direction='in')  # 将x周的刻度线方向设置向内
plt.grid()
plt.legend(loc=4, prop=font1)
plt.show()

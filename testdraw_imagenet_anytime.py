import matplotlib.pyplot as plt


def draw(flops, acc, color, linestyle, legend):
    for i in range(len(flops)):
        vertical_x, vertical_y, horizontal_x, horizontal_y = [], [], [], []
        vertical_x.append(flops[i])
        vertical_x.append(flops[i])
        if i == 0:
            vertical_y.append(0)
            vertical_y.append(acc[i])
            # plt.scatter(vertical_x[-1],vertical_y[-1],color=color)
            plt.plot(vertical_x, vertical_y, color=color, linestyle=linestyle, label=legend)
        else:
            vertical_y.append(acc[i-1])
            vertical_y.append(acc[i])
            horizontal_x.append(flops[i-1])
            horizontal_x.append(flops[i])
            horizontal_y.append(acc[i-1])
            horizontal_y.append(acc[i-1])
            plt.plot(vertical_x,vertical_y,color=color,linestyle=linestyle)
            plt.plot(horizontal_x,horizontal_y,color=color,linestyle=linestyle)
        plt.scatter(flops,acc,color=color)

if __name__=='__main__':
    anytime_cifar100_acc=[63.05,65.41,67.81,69.2,69.87,70.63,71.53]
    anytime_cifar100_flops=[1514052,2550592,37095980,46331024,57335540,66960728,75937724]
    anytime_imagenet_acc_step4=[56.632,65.136,68.42,69.77,71.336]
    anytime_imagenet_flops_step4=[339902824,685457720,1008156120,1254472688,1360529624]
    anytime_imagenet_acc_step7=[62.896,70.316,73,73.594,74.592]
    anytime_imagenet_flops_step7=[615692968,1436392016,2283210200,2967421456,3253794488]
    for i in range(len(anytime_cifar100_flops)):
        anytime_cifar100_flops[i]=anytime_cifar100_flops[i]/100000000
    for i in range(len(anytime_imagenet_flops_step4)):
        anytime_imagenet_flops_step4[i]=anytime_imagenet_flops_step4[i]/10000000000
        anytime_imagenet_flops_step7[i]=anytime_imagenet_flops_step7[i]/10000000000
    # draw(anytime_cifar100_flops,anytime_cifar100_acc,'black','-','MSDNet')
    draw(anytime_imagenet_flops_step4,anytime_imagenet_acc_step4,'blue',':','MSDNet(step=4)')
    draw(anytime_imagenet_flops_step7,anytime_imagenet_acc_step7,'red','-.','MSDNet(step=7)')
    # draw(aa,bb,'red','-.','line2')

    font1 = {'family': 'Consolas',
             'weight': 'light',
             'size': 14,
             }
    plt.rcParams['font.sans-serif'] = ['Consolas']
    plt.title('Anytime prediction on ImageNet',font1)
    plt.xlabel('budget(in MUL-ADD)',font1)
    plt.ylabel('accuracy(%)',font1)
    plt.text(0.53,48, r'x 10$^{10}$')
    plt.ylim(ymin=50)
    plt.xlim(xmax=0.6)
    plt.tick_params(axis='x',direction='in') # 将x周的刻度线方向设置向内
    plt.tick_params(axis='y', direction='in')  # 将x周的刻度线方向设置向内
    plt.grid()
    plt.legend(loc=4,prop=font1)
    plt.show()

# for i in range(len(a)):
#     aa, bb, cc, dd = [], [], [], []
#     aa.append(a[i])
#     aa.append(a[i])
#     if i == 0:
#         bb.append(0)
#         bb.append(b[i])
#     else:
#         cc.append(a[i - 1])
#         cc.append(a[i])
#         dd.append(b[i - 1])
#         dd.append(b[i - 1])
#         bb.append(b[i] - b[i - 1])
#         bb.append(b[i])
#         plt.plot(cc, dd, color='black')
#     plt.plot(aa, bb, color='black', label='test')
#     plt.scatter(aa, bb, color='black')
# plt.legend()
# plt.show()

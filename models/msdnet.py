import torch.nn as nn
import torch
import math
import pdb


class ConvBasic(nn.Module):
    def __init__(self, nIn, nOut, kernel=3, stride=1,
                 padding=1):
        # 调用父辈属性
        super(ConvBasic, self).__init__()
        # 一个容器，构造一个模块
        self.net = nn.Sequential(
            nn.Conv2d(nIn, nOut, kernel_size=kernel, stride=stride,
                      padding=padding, bias=False),
            nn.BatchNorm2d(nOut),
            nn.ReLU(True)
        )
    # x为输入
    def forward(self, x):
        return self.net(x)


class ConvBN(nn.Module):
    def __init__(self, nIn, nOut, type: str, bottleneck,
                 bnWidth):
        """
        a basic conv in MSDNet, two type
        :param nIn:
        :param nOut:
        :param type: normal or down
        :param bottleneck: use bottlenet or not
        :param bnWidth: bottleneck factor
        """
        super(ConvBN, self).__init__()
        layer = []
        nInner = nIn
        if bottleneck is True:
            nInner = min(nInner, bnWidth * nOut)
            layer.append(nn.Conv2d(
                nIn, nInner, kernel_size=1, stride=1, padding=0, bias=False))
            layer.append(nn.BatchNorm2d(nInner))
            layer.append(nn.ReLU(True))
            print('\t\tinChannels {} outChannels {} bottleneck\t\t'.format(nIn, nInner))

        if type == 'normal':
            layer.append(nn.Conv2d(nInner, nOut, kernel_size=3,
                                   stride=1, padding=1, bias=False))
            print('\t\tinChannels {} outChannels {} normal\t\t'.format(nInner, nOut))
        elif type == 'down':
            layer.append(nn.Conv2d(nInner, nOut, kernel_size=3,
                                   stride=2, padding=1, bias=False))
            print('\t\tinChannels {} outChannels {} down\t\t'.format(nInner, nOut))
        else:
            raise ValueError

        layer.append(nn.BatchNorm2d(nOut))
        layer.append(nn.ReLU(True))

        self.net = nn.Sequential(*layer)

    def forward(self, x):

        return self.net(x)


class ConvDownNormal(nn.Module):
    def __init__(self, nIn1, nIn2, nOut, bottleneck, bnWidth1, bnWidth2):
        super(ConvDownNormal, self).__init__()
        # 两个卷积 down来自左上角 normal来自左侧
        self.conv_down = ConvBN(nIn1, nOut // 2, 'down',
                                bottleneck, bnWidth1)
        self.conv_normal = ConvBN(nIn2, nOut // 2, 'normal',
                                  bottleneck, bnWidth2)
    # 传过来的数据为相当于来自两个tensor分别处理左上角和左侧的卷积所以要分x[0] x[1]来处理
    # x[1]来自原本左侧的feature map conv_normal代表这一层的常规卷积 conv_down就是左上角传下来的特殊卷积
    # 将前一层传过来的和本层生成的进行concate  符合densenet操作
    # 再用concat将这个list中的
    def forward(self, x):
        res = [x[1],
               self.conv_down(x[0]),
               self.conv_normal(x[1])]
        return torch.cat(res, dim=1)


class ConvNormal(nn.Module):
    def __init__(self, nIn, nOut, bottleneck, bnWidth):
        super(ConvNormal, self).__init__()
        self.conv_normal = ConvBN(nIn, nOut, 'normal',
                                  bottleneck, bnWidth)

    def forward(self, x):
        # 如果x不是list，则变成list，其实x中只有一个tensor 放进list中cat
        if not isinstance(x, list):
            x = [x]
        res = [x[0],
               self.conv_normal(x[0])]

        return torch.cat(res, dim=1)


class MSDNFirstLayer(nn.Module):
    def __init__(self, nIn, nOut, args):
        # nIn=3 nOut=nChannels
        super(MSDNFirstLayer, self).__init__()
        self.layers = nn.ModuleList()
        if args.data.startswith('cifar'):
            # 第一个scale的第一层输出的channel就是 传过来的nOut*第一个scale的growthrate*1
            self.layers.append(ConvBasic(nIn, nOut * args.grFactor[0],
                                         kernel=3, stride=1, padding=1))
        elif args.data == 'ImageNet':
            conv = nn.Sequential(
                nn.Conv2d(nIn, nOut * args.grFactor[0], 7, 2, 3),
                nn.BatchNorm2d(nOut * args.grFactor[0]),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(3, 2, 1))
            self.layers.append(conv)

        # 接下来一个scale的第一层相当于前一层的输出要输入进来，所以要更新输入的channel数
        nIn = nOut * args.grFactor[0]

        for i in range(1, args.nScales):
            self.layers.append(ConvBasic(nIn, nOut * args.grFactor[i],
                                         kernel=3, stride=2, padding=1))
            nIn = nOut * args.grFactor[i]

        print("****************************First Layer:*********************************")
        print(self.layers)

    def forward(self, x):
        # 将输入x分别经过每一层的处理之后所生成的张量放进list中每一个的张量
        res = []
        for i in range(len(self.layers)):
            x = self.layers[i](x)
            res.append(x)
        return res


class MSDNLayer(nn.Module):
    def __init__(self, nIn, nOut, args, inScales=None, outScales=None):
        super(MSDNLayer, self).__init__()
        self.nIn = nIn
        self.nOut = nOut
        self.inScales = inScales if inScales is not None else args.nScales
        self.outScales = outScales if outScales is not None else args.nScales

        self.nScales = args.nScales
        # 这个表示scale有减少
        self.discard = self.inScales - self.outScales
        # args.nScales = len(args.grFactor) 所以对于cifar为1-2-4 为3
        self.offset = self.nScales - self.outScales
        self.layers = nn.ModuleList()
        # print("普通层",self.discard,self.offset)
        if self.discard > 0:
            nIn1 = nIn * args.grFactor[self.offset - 1]
            nIn2 = nIn * args.grFactor[self.offset]
            _nOut = nOut * args.grFactor[self.offset]
            self.layers.append(ConvDownNormal(nIn1, nIn2, _nOut, args.bottleneck,
                                              args.bnFactor[self.offset - 1],
                                              args.bnFactor[self.offset]))

        else:
            # growthrate=6 为'number of output channels for each layer (the first scale)'
            # 第一个普通层的时候 nIn=16 nOut=6 第一个scale内的growthrate比例为1 因为offset=0 代表这是第一个scale
            self.layers.append(ConvNormal(nIn * args.grFactor[self.offset],
                                          nOut * args.grFactor[self.offset],
                                          args.bottleneck,
                                          args.bnFactor[self.offset]))
            # print('\t\tinChannels {} outChannels {}   first scale\t\t'.format(nIn * args.grFactor[self.offset], nOut * args.grFactor[self.offset]))
            # print("xxxxxxxxxxxxxxxxxxxxxxxxxxxx")
            # print(self.layers)
            # print("xxxxxxxxxxxxxxxxxxxxxxxxxxxx")
        # 相同层的后面几个scale 第一层为例 offset=0 就是从1，2
        for i in range(self.offset + 1, self.nScales):
            nIn1 = nIn * args.grFactor[i - 1]
            nIn2 = nIn * args.grFactor[i]
            _nOut = nOut * args.grFactor[i]
            # offset = 0 i=1时 nIn1=16 nIn2=16*2 _nOut=6*2
            self.layers.append(ConvDownNormal(nIn1, nIn2, _nOut, args.bottleneck,
                                              args.bnFactor[i - 1],
                                              args.bnFactor[i]))

        # print("===========================")
        # print(self.layers)

    def forward(self, x):
        if self.discard > 0:
            inp = []
            for i in range(1, self.outScales + 1):
                inp.append([x[i - 1], x[i]])
        else:
            # 假设scale=3并且左右不变，inp=[[x[0]], [x[0],x[1]], [x[1],x[2]]]
            inp = [[x[0]]]
            for i in range(1, self.outScales):
                inp.append([x[i - 1], x[i]])
        # print(".............................")
        # print(inp)
        # print(".............................")
        # 给每个层分别的输入
        res = []
        for i in range(self.outScales):
            res.append(self.layers[i](inp[i]))

        return res


class ParallelModule(nn.Module):
    """
    This module is similar to luatorch's Parallel Table
    input: N tensor
    network: N module
    output: N tensor
    """

    def __init__(self, parallel_modules):
        super(ParallelModule, self).__init__()
        self.m = nn.ModuleList(parallel_modules)

    def forward(self, x):
        res = []
        for i in range(len(x)):
            res.append(self.m[i](x[i]))

        return res


class ClassifierModule(nn.Module):
    def __init__(self, m, channel, num_classes):
        super(ClassifierModule, self).__init__()
        # m代表放在分类器前的几层
        self.m = m
        self.linear = nn.Linear(channel, num_classes)
        print("======================classifier====================")
        print(self.m)
        print(self.linear)
        print("=====================================================")

    def forward(self, x):
        # -1是代表最下面一个scale的x
        res = self.m(x[-1])
        # 在第0个维度上展开为一维向量
        res = res.view(res.size(0), -1)
        return self.linear(res)


class MSDNet(nn.Module):
    def __init__(self, args):
        super(MSDNet, self).__init__()
        self.blocks = nn.ModuleList()
        self.classifier = nn.ModuleList()
        self.nBlocks = args.nBlocks
        self.steps = [args.base]
        self.args = args
        # step表示分类器的间隔，也表示一共有多少层
        n_layers_all, n_layer_curr = args.base, 0
        # 只有cifar budgeted为lin_grow(分类器缩放的位置为什么什么的和) 其余都是even
        # 对于cifar100 anytime step=2 base=4 block=7 nchannels=16，7个分类器16层
        for i in range(1, self.nBlocks):
            self.steps.append(args.step if args.stepmode == 'even'
                              else args.step * i + 1)
            n_layers_all += self.steps[-1]

        print("building network of steps: ")
        print(self.steps, n_layers_all)
        # cifar nchannels=16
        nIn = args.nChannels
        for i in range(self.nBlocks):
            print(' ********************** Block {} '
                  ' **********************'.format(i + 1))
            m, nIn = \
                self._build_block(nIn, args, self.steps[i],
                                  n_layers_all, n_layer_curr)
            self.blocks.append(m)
            n_layer_curr += self.steps[i]

            if args.data.startswith('cifar100'):
                self.classifier.append(
                    self._build_classifier_cifar(nIn * args.grFactor[-1], 100))
            elif args.data.startswith('cifar10'):
                self.classifier.append(
                    self._build_classifier_cifar(nIn * args.grFactor[-1], 10))
            elif args.data == 'ImageNet':
                self.classifier.append(
                    self._build_classifier_imagenet(nIn * args.grFactor[-1], 1000))
            else:
                raise NotImplementedError

        for m in self.blocks:
            # 检查m是否有'__iter__'
            if hasattr(m, '__iter__'):
                for _m in m:
                    self._init_weights(_m)
            else:
                self._init_weights(m)

        for m in self.classifier:
            if hasattr(m, '__iter__'):
                for _m in m:
                    self._init_weights(_m)
            else:
                self._init_weights(m)

    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            m.weight.data.normal_(0, math.sqrt(2. / n))
        elif isinstance(m, nn.BatchNorm2d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()
        elif isinstance(m, nn.Linear):
            m.bias.data.zero_()

    def _build_block(self, nIn, args, step, n_layer_all, n_layer_curr):

        # 如果当前层为0则为第一层 3是因为输入进去的RGB channel肯定是3啊
        layers = [MSDNFirstLayer(3, nIn, args)] \
            if n_layer_curr == 0 else []
        # 对于这个block中的每一层
        for i in range(step):
            n_layer_curr += 1
            inScales = args.nScales
            outScales = args.nScales
            # prune方式都是max 根据层数判断scale是否减少
            if args.prune == 'min':
                inScales = min(args.nScales, n_layer_all - n_layer_curr + 2)
                outScales = min(args.nScales, n_layer_all - n_layer_curr + 1)
            elif args.prune == 'max':
                interval = math.ceil(1.0 * n_layer_all / args.nScales)
                inScales = args.nScales - math.floor(1.0 * (max(0, n_layer_curr - 2)) / interval)
                outScales = args.nScales - math.floor(1.0 * (n_layer_curr - 1) / interval)
            else:
                raise ValueError
            # 加入普通层 cifar的growthrate默认为6 nIn=16
            layers.append(MSDNLayer(nIn, args.growthRate, args, inScales, outScales))
            print('|\t\tinScales {} outScales {} inChannels {} outChannels {}\t\t|'.format(inScales, outScales, nIn,
                                                                                           args.growthRate))

            nIn += args.growthRate
            if args.prune == 'max' and inScales > outScales and \
                    args.reduction > 0:
                # offset可能为1或者2
                offset = args.nScales - outScales
                layers.append(
                    self._build_transition(nIn, math.floor(1.0 * args.reduction * nIn),
                                           outScales, offset, args))
                _t = nIn
                nIn = math.floor(1.0 * args.reduction * nIn)
                print('|\t\tTransition layer inserted! (max), inChannels {}, outChannels {}\t|'.format(_t, math.floor(
                    1.0 * args.reduction * _t)))
            elif args.prune == 'min' and args.reduction > 0 and \
                    ((n_layer_curr == math.floor(1.0 * n_layer_all / 3)) or
                     n_layer_curr == math.floor(2.0 * n_layer_all / 3)):
                offset = args.nScales - outScales
                layers.append(self._build_transition(nIn, math.floor(1.0 * args.reduction * nIn),
                                                     outScales, offset, args))

                nIn = math.floor(1.0 * args.reduction * nIn)
                print('|\t\tTransition layer inserted! (min)\t|')
            print("")

        return nn.Sequential(*layers), nIn

    def _build_transition(self, nIn, nOut, outScales, offset, args):
        net = []
        for i in range(outScales):
            net.append(ConvBasic(nIn * args.grFactor[offset + i],
                                 nOut * args.grFactor[offset + i],
                                 kernel=1, stride=1, padding=0))
        print("+++++++++++++++++++++++++++trasition layer++++++++++++++++++++")
        print(net)
        print("+++++++++++++++++++++++++++trasition end++++++++++++++++++++")
        return ParallelModule(net)

    def _build_classifier_cifar(self, nIn, num_classes):
        interChannels1, interChannels2 = 128, 128
        conv = nn.Sequential(
            ConvBasic(nIn, interChannels1, kernel=3, stride=2, padding=1),
            ConvBasic(interChannels1, interChannels2, kernel=3, stride=2, padding=1),
            nn.AvgPool2d(2),
        )
        return ClassifierModule(conv, interChannels2, num_classes)

    def _build_classifier_imagenet(self, nIn, num_classes):
        conv = nn.Sequential(
            ConvBasic(nIn, nIn, kernel=3, stride=2, padding=1),
            ConvBasic(nIn, nIn, kernel=3, stride=2, padding=1),
            nn.AvgPool2d(2)
        )
        return ClassifierModule(conv, nIn, num_classes)

    def forward(self, x):
        res = []
        for i in range(self.nBlocks):
            x = self.blocks[i](x)
            res.append(self.classifier[i](x))
        return res

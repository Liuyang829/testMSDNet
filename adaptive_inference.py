from __future__ import absolute_import
from __future__ import unicode_literals
from __future__ import print_function
from __future__ import division

import torch
import torch.nn as nn
import os
import math

def dynamic_evaluate(model, test_loader, val_loader, args):
    tester = Tester(model, args)
    if os.path.exists(os.path.join(args.save, 'logits_single.pth')): 
        val_pred, val_target, test_pred, test_target = \
            torch.load(os.path.join(args.save, 'logits_single.pth')) 
    else:
        # 这里已经计算了每个分类器对于每一个样本的一个预测结果置信度
        val_pred, val_target = tester.calc_logit(val_loader) 
        test_pred, test_target = tester.calc_logit(test_loader) 
        torch.save((val_pred, val_target, test_pred, test_target), 
                    os.path.join(args.save, 'logits_single.pth'))

    flops = torch.load(os.path.join(args.save, 'flops.pth'))

    with open(os.path.join(args.save, 'dynamic.txt'), 'w') as fout:
        for p in range(1, 40):
            print("*********************")
            #生成一个0.05-1.95
            _p = torch.FloatTensor(1).fill_(p * 1.0 / 20)
            # 通过一个对数生成一个blocks维的tensor
            probs = torch.exp(torch.log(_p) * torch.range(1, args.nBlocks))
            probs /= probs.sum()
            # 利用验证集去找阈值
            acc_val, _, T = tester.dynamic_eval_find_threshold(
                val_pred, val_target, probs, flops)
            # 利用验证集上的结果去给分类器置信度
            acc_test, exp_flops = tester.dynamic_eval_with_threshold(
                test_pred, test_target, flops, T)
            print('valid acc: {:.3f}, test acc: {:.3f}, test flops: {:.2f}M'.format(acc_val, acc_test, exp_flops / 1e6))
            fout.write('{}\t{}\n'.format(acc_test, exp_flops.item()))


class Tester(object):
    def __init__(self, model, args=None):
        self.args = args
        self.model = model
        # softmax操作之后在dim这个维度相加等于1
        self.softmax = nn.Softmax(dim=1).cuda()

    def calc_logit(self, dataloader):
        self.model.eval()
        n_stage = self.args.nBlocks
        logits = [[] for _ in range(n_stage)]
        targets = []
        for i, (input, target) in enumerate(dataloader):
            targets.append(target)
            with torch.no_grad():
                input_var = torch.autograd.Variable(input)
                # 模型生成每个分类器的预测结果
                output = self.model(input_var)
                if not isinstance(output, list):
                    output = [output]
                # softmax相当于将值映射到0-1直接并且和为1
                for b in range(n_stage):
                    _t = self.softmax(output[b])

                    logits[b].append(_t) 

            if i % self.args.print_freq == 0: 
                print('Generate Logit: [{0}/{1}]'.format(i, len(dataloader)))

        for b in range(n_stage):
            logits[b] = torch.cat(logits[b], dim=0)
        # logits相当于每个block输出的结果，首先是nBlocks维，
        # 因为有多个分类器输出，logits[0]~logits[nBlocks-1]
        # 对于每个输出结果，肯定是输入的数量num*classes类别置信度向量 然后根据size变成张量
        size = (n_stage, logits[0].size(0), logits[0].size(1))
        ts_logits = torch.Tensor().resize_(size).zero_()
        for b in range(n_stage):
            ts_logits[b].copy_(logits[b])
        # 将targets也变成张量
        targets = torch.cat(targets, dim=0)
        ts_targets = torch.Tensor().resize_(size[1]).copy_(targets)

        return ts_logits, ts_targets

    def dynamic_eval_find_threshold(self, logits, targets, p, flops):
        """
            logits: m * n * c
            m: Stages-nblocks
            n: Samples
            c: Classes
        """
        n_stage, n_sample, c = logits.size()
        print(logits.size())
        # dim=2返回的max_preds是每一行的最大值,这个最大值也就是预测出来最大置信度的那个置信度
        # argmax_greds就是这个最大预测值的下标，代表是第几个
        # 所以max_preds的维度是nblocks行，samples列，代表每个分类器出来的每个样本的最大置信度预测结果
        # 例如7行10列 就表示7个分类器有10个样本进去输出的预测结果置信度，arg代表的就是原来的下标代表第几个
        max_preds, argmax_preds = logits.max(dim=2, keepdim=False)

        # 这里对max_preds在行上进行排序，_为排序后的结果，sort_id就是对应原来矩阵中的下标
        _, sorted_idx = max_preds.sort(dim=1, descending=True)
        # 样本个数个
        filtered = torch.zeros(n_sample)
        # 用来装阈值的
        T = torch.Tensor(n_stage).fill_(1e8)
        # p是每个分类器的权重
        # 对一个中间分类器而言，已经设定好了从这个分类其中分出去的数量，
        # 那就把这个分类器出来的所有结果排序后前n个当作这个分类器可以退出的，
        # 那么第n个分出去的那个预测结果的置信度就是阈值
        for k in range(n_stage - 1):
            acc, count = 0.0, 0
            # 计划每个分类器按照权重分出去的个数
            out_n = math.floor(n_sample * p[k])
            for i in range(n_sample):
                # ori_idx表示
                ori_idx = sorted_idx[k][i]
                # filter记录着每个样本是否已经退出 只有还没有退出的才能作为计算 起标记作用
                if filtered[ori_idx] == 0:
                    count += 1
                    # 到了预计的退出数量就记下来那个阈值
                    if count == out_n:
                        T[k] = max_preds[k][ori_idx]
                        break
            # ge判断张量内每一个数值大小的函数 type_as为了该表数据类型后才能用ge进行比较 add_为加的操作
            # ge的比较结果在于在本层分类器中有多少个样本已经退出，得出来的理想结果应该是[1,1,1,1,1...0,0,0,...]
            # filter本来为一个sample维的0向量,加上比较的结果后就说明标记好了已经退出去的样本。

            filtered.add_(max_preds[k].ge(T[k]).type_as(filtered))

        T[n_stage -1] = -1e8 # accept all of the samples at the last stage

        # 计算正确率
        acc_rec, exp = torch.zeros(n_stage), torch.zeros(n_stage)
        acc, expected_flops = 0, 0
        for i in range(n_sample):
            gold_label = targets[i]
            for k in range(n_stage):
                if max_preds[k][i].item() >= T[k]: # force the sample to exit at k
                    if int(gold_label.item()) == int(argmax_preds[k][i].item()):
                        acc += 1
                        acc_rec[k] += 1
                    exp[k] += 1
                    break
        acc_all = 0
        # 根据比例计算flops
        for k in range(n_stage):
            _t = 1.0 * exp[k] / n_sample
            expected_flops += _t * flops[k]
            acc_all += acc_rec[k]

        return acc * 100.0 / n_sample, expected_flops, T

    def dynamic_eval_with_threshold(self, logits, targets, flops, T):
        # 和上面类似 接下来是一个比较的过程
        n_stage, n_sample, _ = logits.size()
        max_preds, argmax_preds = logits.max(dim=2, keepdim=False) # take the max logits as confidence
        # acc为总的正确个数 acc_rec为每一个分类器的正确个数
        acc_rec, exp = torch.zeros(n_stage), torch.zeros(n_stage)
        acc, expected_flops = 0, 0
        for i in range(n_sample):
            gold_label = targets[i]
            for k in range(n_stage):
                if max_preds[k][i].item() >= T[k]: # force to exit at k
                    _g = int(gold_label.item())
                    _pred = int(argmax_preds[k][i].item())
                    if _g == _pred:
                        acc += 1
                        acc_rec[k] += 1
                    exp[k] += 1
                    break
        # 根据每个分类器退出数量计算出flops，flops计算的时候本身就是一个分类器一个flops
        acc_all, sample_all = 0, 0
        for k in range(n_stage):
            _t = exp[k] * 1.0 / n_sample
            sample_all += exp[k]
            expected_flops += _t * flops[k]
            acc_all += acc_rec[k]

        return acc * 100.0 / n_sample, expected_flops

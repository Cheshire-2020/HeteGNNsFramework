import numpy as np
import torch
from .logreg import LogReg
import torch.nn as nn
from sklearn.metrics import f1_score
from torch.nn.functional import softmax
from sklearn.metrics import roc_auc_score

##################################################
# This section of code adapted from pcy1302/DMGI #
##################################################
'''
evaluate.py 文件的主要功能是对模型生成的嵌入进行分类任务的评估，
使用逻辑回归（LogReg）作为分类器，并计算多种评估指标（如准确率、F1 分数和 AUC）。

输入：
    embeds：节点嵌入矩阵，形状为 N×D，其中 N 是节点数，D 是嵌入维度。
    ratio：训练、验证和测试集的划分比例。
    idx_train、idx_val、idx_test：训练、验证和测试集的节点索引。
    label：节点的标签，形状为 N×C，其中 C 是类别数。
    nb_classes：类别数。
    device：设备（CPU 或 GPU）。
    dataset：数据集名称。
    lr：逻辑回归分类器的学习率。
    wd：逻辑回归分类器的权重衰减。
    isTest：是否打印测试结果。

输出：
    如果 isTest=True，打印分类任务的评估结果（Macro-F1、Micro-F1 和 AUC）。
    如果 isTest=False，返回验证集和测试集的 Macro-F1 均值。

评估指标
Macro-F1 和 Micro-F1
Macro-F1：
    计算每个类别的 F1 分数，然后取平均值。
    对所有类别一视同仁，适合类别分布不均衡的情况。
Micro-F1：
    计算全局的精确率和召回率，然后计算 F1 分数。
    更关注整体的分类性能。
AUC
AUC（Area Under Curve）：
    使用 roc_auc_score 计算多分类任务的 AUC 分数。
    multi_class='ovr' 表示使用一对多的方式计算 AUC。
'''


def evaluate(embeds, ratio, idx_train, idx_val, idx_test, label, nb_classes, device, dataset, lr, wd
             , isTest=True):
    '''
    数据准备：
    嵌入划分：
        根据训练、验证和测试集的索引，将嵌入划分为 train_embs、val_embs 和 test_embs。
    标签处理：
        将独热编码的标签（label）转换为类别索引（通过 torch.argmax）。
    '''
    hid_units = embeds.shape[1]
    xent = nn.CrossEntropyLoss()

    train_embs = embeds[idx_train]
    val_embs = embeds[idx_val]
    test_embs = embeds[idx_test]

    train_lbls = torch.argmax(label[idx_train], dim=-1)
    val_lbls = torch.argmax(label[idx_val], dim=-1)
    test_lbls = torch.argmax(label[idx_test], dim=-1)
    accs = []
    micro_f1s = []
    macro_f1s = []
    macro_f1s_val = []
    auc_score_list = []

    '''
    分类器初始化:
    逻辑回归分类器：
        使用 LogReg（定义在 logreg.py 中）作为分类器。
        输入维度为嵌入维度（hid_units），输出维度为类别数（nb_classes）。
    优化器：
        使用 Adam 优化器，学习率为 lr，权重衰减为 wd。
    '''
    for _ in range(50):
        log = LogReg(hid_units, nb_classes)
        opt = torch.optim.Adam(log.parameters(), lr=lr, weight_decay=wd)
        log.to(device)

        val_accs = []
        test_accs = []
        val_micro_f1s = []
        test_micro_f1s = []
        val_macro_f1s = []
        test_macro_f1s = []

        logits_list = []

        for iter_ in range(200):
            '''
            分类器训练与验证:
            训练过程：
                每次迭代中，使用训练集嵌入（train_embs）和标签（train_lbls）训练逻辑回归分类器。
                损失函数为交叉熵损失（CrossEntropyLoss）。
                通过反向传播更新分类器参数。
            '''
            # train
            log.train()
            opt.zero_grad()

            logits = log(train_embs)
            loss = xent(logits, train_lbls)

            loss.backward()
            opt.step()

            '''
            验证与测试
            验证集评估：
                使用验证集嵌入（val_embs）计算分类器的预测结果。
                计算验证集的准确率（val_acc）、Macro-F1 和 Micro-F1。
            测试集评估：
                使用测试集嵌入（test_embs）计算分类器的预测结果。
                计算测试集的准确率（test_acc）、Macro-F1 和 Micro-F1。
            '''
            # val
            logits = log(val_embs)
            preds = torch.argmax(logits, dim=1)

            val_acc = torch.sum(preds == val_lbls).float() / val_lbls.shape[0]
            val_f1_macro = f1_score(val_lbls.cpu(), preds.cpu(), average='macro')
            val_f1_micro = f1_score(val_lbls.cpu(), preds.cpu(), average='micro')

            val_accs.append(val_acc.item())
            val_macro_f1s.append(val_f1_macro)
            val_micro_f1s.append(val_f1_micro)

            # test
            logits = log(test_embs)
            preds = torch.argmax(logits, dim=1)

            test_acc = torch.sum(preds == test_lbls).float() / test_lbls.shape[0]
            test_f1_macro = f1_score(test_lbls.cpu(), preds.cpu(), average='macro')
            test_f1_micro = f1_score(test_lbls.cpu(), preds.cpu(), average='micro')

            test_accs.append(test_acc.item())
            test_macro_f1s.append(test_f1_macro)
            test_micro_f1s.append(test_f1_micro)
            logits_list.append(logits)

        max_iter = val_accs.index(max(val_accs))
        accs.append(test_accs[max_iter])
        max_iter = val_macro_f1s.index(max(val_macro_f1s))
        macro_f1s.append(test_macro_f1s[max_iter])
        macro_f1s_val.append(val_macro_f1s[max_iter])

        max_iter = val_micro_f1s.index(max(val_micro_f1s))
        micro_f1s.append(test_micro_f1s[max_iter])

        '''
        AUC 计算：
            选择验证集表现最好的分类器（max_iter 对应的分类器）。
            使用测试集的预测概率（best_proba）计算 AUC 分数。
        '''
        # auc
        best_logits = logits_list[max_iter]
        best_proba = softmax(best_logits, dim=1)
        auc_score_list.append(roc_auc_score(y_true=test_lbls.detach().cpu().numpy(),
                                            y_score=best_proba.detach().cpu().numpy(),
                                            multi_class='ovr'
                                            ))

    '''
    结果记录与输出
    结果输出：
        如果 isTest=True，打印测试集的 Macro-F1、Micro-F1 和 AUC 的均值和方差。
        如果 isTest=False，返回验证集和测试集的 Macro-F1 均值。
    结果保存：
        将结果保存到文件中，文件名为 result_<dataset><ratio>.txt。
    '''
    if isTest:
        print("\t[Classification] Macro-F1_mean: {:.4f} var: {:.4f}  Micro-F1_mean: {:.4f} var: {:.4f} auc {:.4f}"
              .format(np.mean(macro_f1s),
                      np.std(macro_f1s),
                      np.mean(micro_f1s),
                      np.std(micro_f1s),
                      np.mean(auc_score_list),
                      np.std(auc_score_list)
                      )
              )
    else:
        return np.mean(macro_f1s_val), np.mean(macro_f1s)

    f = open("result_" + dataset + str(ratio) + ".txt", "a")
    f.write(str(np.mean(macro_f1s)) + "\t" + str(np.mean(micro_f1s)) + "\t" + str(np.mean(auc_score_list)) + "\n")
    f.close()

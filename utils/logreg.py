import torch
import torch.nn as nn

# logreg.py 文件定义了一个简单的逻辑回归分类器（LogReg），用于对节点嵌入进行分类任务。

class LogReg(nn.Module):
    '''
    功能：实现一个单层的逻辑回归分类器。
    输入参数：
        ft_in：输入特征的维度（即嵌入的维度）。
        nb_classes：输出类别数。
    核心组件：
        self.fc：一个全连接层（nn.Linear），将输入特征映射到类别空间，形状从ft_in→nb_classes。
    '''
    def __init__(self, ft_in, nb_classes):
        super(LogReg, self).__init__()
        self.fc = nn.Linear(ft_in, nb_classes)

        for m in self.modules():
            self.weights_init(m)

    '''
    功能：对模型的参数进行初始化。
    逻辑：
        如果模块是一个线性层（nn.Linear），则：
            使用 Xavier 均匀分布初始化权重（torch.nn.init.xavier_uniform_）。
            将偏置（bias）初始化为 0。
    Xavier 初始化的作用是让权重的初始值保持在一个合理的范围内，避免梯度消失或爆炸问题。
    '''
    def weights_init(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    '''
    功能：定义前向传播的逻辑。
    输入：seq：输入特征，形状为 N×ft_in，其中 N 是样本数
    输出：分类器的输出，形状为 N×nb_classes，表示每个样本属于各类别的分数。
    '''
    def forward(self, seq):
        ret = self.fc(seq)
        return ret

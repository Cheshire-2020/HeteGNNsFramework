import torch
import torch.nn.functional as F


class Trainer:
    """
    通用 Trainer，用于训练和测试不同模型
    """

    def __init__(self, model, data, device, task_type='node_classification', **kwargs):
        """
        初始化 Trainer

        :param model: 模型实例
        :param data: 数据集对象
        :param device: 设备 ('cpu', 'cuda', etc.)
        :param task_type: 任务类型 ('node_classification' 或 'embedding_learning')
        :param kwargs: 其他训练相关参数
        """
        self.model = model.to(device)
        self.data = data.to(device)
        self.device = device
        self.task_type = task_type
        self.optimizer = torch.optim.Adam(
            model.parameters(), lr=kwargs.get('lr', 0.005), weight_decay=kwargs.get('weight_decay', 0.001)
        )
        self.loss_fn = F.cross_entropy  # 默认使用 CrossEntropyLoss

    def train_epoch(self):
        """
        单轮训练逻辑

        :return: 本轮训练的损失值
        """
        self.model.train()
        self.optimizer.zero_grad()

        if self.task_type == 'node_classification':
            # 前向传播
            out = self.model(self.data.x_dict, self.data.edge_index_dict)
            mask = self.data['movie'].train_mask  # 使用训练集 Mask
            loss = self.loss_fn(out[mask], self.data['movie'].y[mask])  # 计算损失

        elif self.task_type == 'embedding_learning':
            # MetaPath2Vec 的损失计算
            pos_rw, neg_rw = next(self.data.loader)  # 从 loader 中取出随机游走路径
            loss = self.model.loss(pos_rw.to(self.device), neg_rw.to(self.device))

        else:
            raise ValueError(f"Unsupported task type: {self.task_type}")

        loss.backward()  # 反向传播
        self.optimizer.step()  # 优化器更新
        return loss.item()

    @torch.no_grad()
    def evaluate(self, split='val'):
        """
        模型评估逻辑

        :param split: 数据集划分 ('train', 'val', 'test')
        :return: 当前数据集上的分类准确率
        """
        self.model.eval()

        if self.task_type == 'node_classification':
            # 节点分类任务评估
            out = self.model(self.data.x_dict, self.data.edge_index_dict)
            mask = self.data['movie'][f'{split}_mask']  # 使用不同的 Mask (train/val/test)
            pred = out.argmax(dim=-1)  # 获取预测类别
            acc = (pred[mask] == self.data['movie'].y[mask]).sum() / mask.sum()  # 准确率计算
            return acc.item()

        elif self.task_type == 'embedding_learning':
            # Embedding 任务评估
            z = self.model('author', batch=self.data['author'].y_index.to(self.device))
            y = self.data['author'].y

            # 划分训练集和测试集
            train_size = int(len(z) * 0.1)
            perm = torch.randperm(len(z))
            train_perm = perm[:train_size]
            test_perm = perm[train_size:]

            return self.model.test(z[train_perm], y[train_perm], z[test_perm], y[test_perm])

        else:
            raise ValueError(f"Unsupported task type: {self.task_type}")

    def train(self, num_epochs, log_interval=10):
        """
        模型完整训练过程

        :param num_epochs: 训练的总轮数
        :param log_interval: 日志打印间隔
        :return: 最好的验证准确率和对应的测试准确率
        """
        best_val_acc = 0
        best_test_acc = 0

        for epoch in range(1, num_epochs + 1):
            loss = self.train_epoch()  # 单轮训练

            if epoch % log_interval == 0 or epoch == 1:
                train_acc = self.evaluate(split='train')  # 训练集准确率
                val_acc = self.evaluate(split='val')  # 验证集准确率
                test_acc = self.evaluate(split='test')  # 测试集准确率
                print(f"Epoch: {epoch:03d}, Loss: {loss:.4f}, Train: {train_acc:.4f}, "
                      f"Val: {val_acc:.4f}, Test: {test_acc:.4f}")

                # 保存最佳验证结果对应的测试准确率
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    best_test_acc = test_acc

        return best_val_acc, best_test_acc

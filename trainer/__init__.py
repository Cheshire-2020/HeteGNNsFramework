from .trainer import Trainer


def get_trainer(model, data, device, task_type='node_classification', **kwargs):
    """
    初始化 Trainer 工具

    :param model: 模型实例
    :param data: 数据集实例
    :param device: 设备类型 ('cpu', 'cuda', etc.)
    :param task_type: 任务类型 ('node_classification', 'embedding_learning')
    :param kwargs: 其他训练超参数（如 lr, weight_decay 等）
    :return: Trainer 实例
    """
    return Trainer(model, data, device, task_type, **kwargs)

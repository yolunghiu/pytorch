import numpy as np
import torch
import torch.nn as nn

"""
Notes: 这个损失函数就是一个Sigmoid层+BCELoss
"""

# ----------------------------------- BCE with Logis Loss -----------------------------------
# flag = 0
flag = 1
if flag:
    inputs = torch.tensor([[1, 2], [2, 2], [3, 4], [4, 5]], dtype=torch.float)
    target = torch.tensor([[1, 0], [1, 0], [0, 1], [0, 1]], dtype=torch.float)

    target_bce = target

    # inputs = torch.sigmoid(inputs)

    weights = torch.tensor([1, 1], dtype=torch.float)

    loss_f_none_w = nn.BCEWithLogitsLoss(weight=weights, reduction='none')
    loss_f_sum = nn.BCEWithLogitsLoss(weight=weights, reduction='sum')
    loss_f_mean = nn.BCEWithLogitsLoss(weight=weights, reduction='mean')

    # forward
    loss_none_w = loss_f_none_w(inputs, target_bce)
    loss_sum = loss_f_sum(inputs, target_bce)
    loss_mean = loss_f_mean(inputs, target_bce)

    # view
    print("\nweights: ", weights)
    print(loss_none_w, loss_sum, loss_mean)

    # 先对input进行sigmoid激活，再算BCELoss，结果与BCEWithLogitsLoss一致
    loss_f_bce = nn.BCELoss(reduction='sum')
    inputs_sigmoid = nn.Sigmoid()(inputs)
    loss_bce = loss_f_bce(inputs_sigmoid, target)
    print(loss_bce)

# --------------------------------- pos weight

flag = 0
# flag = 1
if flag:
    inputs = torch.tensor([[1, 2], [2, 2], [3, 4], [4, 5]], dtype=torch.float)
    target = torch.tensor([[1, 0], [1, 0], [0, 1], [0, 1]], dtype=torch.float)

    target_bce = target

    # itarget
    # inputs = torch.sigmoid(inputs)

    weights = torch.tensor([1], dtype=torch.float)
    pos_w = torch.tensor([3], dtype=torch.float)  # 3

    loss_f_none_w = nn.BCEWithLogitsLoss(weight=weights, reduction='none', pos_weight=pos_w)
    loss_f_sum = nn.BCEWithLogitsLoss(weight=weights, reduction='sum', pos_weight=pos_w)
    loss_f_mean = nn.BCEWithLogitsLoss(weight=weights, reduction='mean', pos_weight=pos_w)

    # forward
    loss_none_w = loss_f_none_w(inputs, target_bce)
    loss_sum = loss_f_sum(inputs, target_bce)
    loss_mean = loss_f_mean(inputs, target_bce)

    # view
    print("\npos_weights: ", pos_w)
    print(loss_none_w, loss_sum, loss_mean)

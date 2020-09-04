import matplotlib.pyplot as plt
import torch
import torch.optim as optim
import math

LR = 0.1
iteration = 10
max_epoch = 200

# %% fake data and optimizer

weights = torch.randn((1), requires_grad=True)
target = torch.zeros((1))

optimizer = optim.SGD([weights], lr=LR, momentum=0.9)

# %% Cosine Annealing LR
lf = lambda x: (((1 + math.cos(x * math.pi / max_epoch)) / 2) ** 1.0) * 0.95 + 0.05  # cosine
scheduler_lr = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)

lr_list, epoch_list = [], []
for epoch in range(max_epoch):
    lr_list.append(scheduler_lr.get_lr())
    epoch_list.append(epoch)

    for i in range(iteration):
        loss = torch.pow((weights - target), 2)
        loss.backward()

        optimizer.step()
        optimizer.zero_grad()

    scheduler_lr.step()

plt.plot(epoch_list, lr_list, label="Cosine Learning Rate Decay")
plt.xlabel("Epoch")
plt.ylabel("Learning rate")
plt.legend()
plt.show()

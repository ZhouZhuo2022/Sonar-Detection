import torch
from typing import Iterable

from utils.misc import Accumulator


def train_one_epoch(model: torch.nn.Module, Loss: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, max_norm: float = 0,
                    animator=None):

    metric = Accumulator(2)

    model.train()
    for iteration, batch in enumerate(data_loader):
        images, targets = batch[0], batch[1]
        images = images.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        optimizer.zero_grad()

        outputs = model(images)

        loss_value_all = 0
        num_pos_all = 0

        for l in range(len(outputs)):
            loss_item, num_pos = Loss(l, outputs[l], targets)
            loss_value_all += loss_item
            num_pos_all += num_pos

        loss_value = loss_value_all / num_pos_all
        loss_value.backward()
        optimizer.step()

        metric.add(loss_value_all, num_pos_all)
    animator.add(iteration+1, metric[0]/metric[1])

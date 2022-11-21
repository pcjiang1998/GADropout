from typing import List

import torch
from tqdm import tqdm


def accuracy(output, target, top_k=(1,)):
    if top_k == 1 or (len(top_k) == 1 and top_k[0] == 1):
        with torch.no_grad():
            pred = output.argmax(dim=1)
            return [torch.eq(pred, target).sum().float().item()]
    else:
        """Computes the accuracy over the k top predictions for the specified values of k"""
        with torch.no_grad():
            max_k = max(top_k)
            _, pred = output.topk(max_k, 1, True, True)
            pred = pred.t()
            correct = pred.eq(target.view(1, -1).expand_as(pred)).contiguous()
            res = []
            for k in top_k:
                correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
                res.append(correct_k.item())
            return res


def test_top_1(model, device, dataloader, loss_fun, verbose: bool = False):
    model = model.to(device)
    running_loss = 0.0
    running_corrects = 0.0
    total = 0
    model.eval()
    with torch.no_grad():
        pbar = enumerate(dataloader)
        if verbose:
            pbar = tqdm(pbar)
        for batch_idx, (inputs, labels) in pbar:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = loss_fun(outputs, labels)
            running_loss += loss.item() * inputs.size(0)
            total += labels.size(0)
            running_corrects += accuracy(outputs, labels, top_k=[1])[0]
            if verbose:
                pbar.set_description(
                    f"{(batch_idx + 1) * 100 / len(dataloader):.2f}% "
                    f"| Loss: {running_loss / (batch_idx + 1):.5e} "
                    f"| Acc: {100. * running_corrects / total:.3f}% ({running_corrects}/{total})"
                )
    epoch_loss = running_loss / (batch_idx + 1)
    epoch_acc = running_corrects / total
    return epoch_loss, epoch_acc


def test_top_k(model, device, dataloader, loss_fun, top_k: List, verbose: bool = False):
    model = model.to(device)
    running_loss = 0
    running_corrects = [0 for _ in top_k]
    total = 0
    model.eval()
    with torch.no_grad():
        pbar = enumerate(dataloader)
        if verbose:
            pbar = tqdm(pbar)
        for batch_idx, (inputs, labels) in pbar:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = loss_fun(outputs, labels)
            running_loss += loss.item() * inputs.size(0)
            total += labels.size(0)
            res = accuracy(outputs, labels, top_k=top_k)
            for i in range(len(running_corrects)):
                running_corrects[i] += res[i]
            if verbose:
                append = [
                    f"| Acc-{k}: {100. * running_corrects[k] / total:.3f}% ({running_corrects[k]}/{total})"
                    for k in range(len(running_corrects))
                ]
                s = f"{(batch_idx + 1) * 100 / len(dataloader):.2f}% | Loss: {running_loss / (batch_idx + 1):.5e} "
                for i in append:
                    s += i
                pbar.set_description(s)
    return running_loss / (batch_idx + 1), [100.0 * i / total for i in running_corrects]

from typing import List

import numpy as np
import torch
import torch.nn as nn


def dropout(X, drop_prob):
    X = X.float()
    assert 0 <= drop_prob <= 1
    keep_prob = 1 - drop_prob
    if keep_prob == 0:
        return torch.zeros_like(X)
    mask = (torch.rand(X.shape) < keep_prob).float()
    return mask * X / keep_prob


class GADropout(nn.Module):
    def __init__(self, num_features):
        super(GADropout, self).__init__()
        self.gene: List or None = None
        self.num_features: int = num_features

    def forward(self, x):
        if self.training:
            assert self.gene is not None
            self.gene = self.gene.to(x.device)
            num_features = x.size(1)
            mask = torch.rand_like(x) < self.gene
            scale = torch.sum(mask, dim=1).unsqueeze_(1)
            scale = num_features / scale
            scale = torch.where(torch.isinf(scale), torch.full_like(scale, -1), scale)
            return x * mask * scale
        else:
            return x

    def set_gene(self, gene: np.ndarray or torch.Tensor):
        assert len(gene) == self.num_features
        if isinstance(gene, np.ndarray):
            self.gene = torch.from_numpy(gene)
        else:  # torch.Tensor or torch.cuda.Tensor
            self.gene = gene

    def extra_repr(self) -> str:
        return "drop_features={}".format(self.num_features)

    @classmethod
    def convert_to_normal(cls, module, ModuleType=nn.Dropout):
        module_output = module
        if type(module) == GADropout:
            module_output = ModuleType()
        else:
            for name, child in module.named_children():
                module_output.add_module(name, cls.convert_to_normal(child, ModuleType))
        del module
        return module_output


def main():
    num_features = 10
    model = GADropout(num_features)
    model.train()
    model.set_gene(torch.Tensor([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]))
    x = torch.rand([5, num_features]).cuda()
    y = model(x)
    print(y)


if __name__ == "__main__":
    main()

import numpy as np
import torch
import torch.nn as nn

from .GADropout import GADropout


class MLP(nn.Module):
    def __init__(self, layer_list):
        super(MLP, self).__init__()
        self.layers = []
        self.gadropout_list = []
        for i in layer_list:
            if isinstance(i, list):
                self.layers.append(nn.Linear(i[0], i[1]))
            elif i.lower() == "relu":
                self.layers.append(nn.ReLU(inplace=True))
            elif i.lower() == "dropout":
                self.layers.append(nn.Dropout(0.5))
            elif i.lower().startswith("gadropout_"):
                t = GADropout(int(i[10:]))
                self.layers.append(t)
                self.gadropout_list.append(t)
        self.classifier = nn.Sequential(*self.layers)

    def set_gene(self, gene: np.ndarray or torch.Tensor):
        assert len(gene) == self.gene_len()
        offset = 0
        for i in range(len(self.gadropout_list)):
            self.gadropout_list[i].set_gene(
                torch.Tensor(
                    gene[offset : offset + self.gadropout_list[i].num_features]
                )
            )
            offset += self.gadropout_list[i].num_features

    def gene_len(self):
        return sum(i.num_features for i in self.gadropout_list)

    def forward(self, x):
        return self.classifier(x)


class MLPv2(nn.Module):
    def __init__(
        self, node_list, drop_first=False, activate_last=False, DropoutType=GADropout
    ):
        super(MLPv2, self).__init__()
        self.layers = []
        self.DropoutType = DropoutType
        if self.DropoutType == GADropout:
            self.gadropout_list = []
            for in_f, out_f in zip(node_list[:-1], node_list[1:]):
                d = GADropout(in_f)
                l = nn.Linear(in_f, out_f)
                r = nn.Sigmoid()
                self.layers += [d, l, r]
                self.gadropout_list += [d]
            if not drop_first:
                del self.gadropout_list[0]
                del self.layers[0]
            if not activate_last:
                del self.layers[-1]
        elif self.DropoutType == nn.Dropout:
            for in_f, out_f in zip(node_list[:-1], node_list[1:]):
                d = nn.Dropout(0.5)
                l = nn.Linear(in_f, out_f)
                r = nn.Sigmoid()
                self.layers += [d, l, r]
            if not drop_first:
                del self.layers[0]
            if not activate_last:
                del self.layers[-1]
        else:
            raise Exception()
        self.classifier = nn.Sequential(*self.layers)

    def set_gene(self, gene: np.ndarray or torch.Tensor):
        assert len(gene) == self.gene_len() and self.DropoutType == GADropout
        offset = 0
        for i in range(len(self.gadropout_list)):
            self.gadropout_list[i].set_gene(
                torch.Tensor(
                    gene[offset : offset + self.gadropout_list[i].num_features]
                )
            )
            offset += self.gadropout_list[i].num_features

    def gene_len(self):
        assert self.DropoutType == GADropout
        return sum(i.num_features for i in self.gadropout_list)

    def forward(self, x):
        return self.classifier(x)


def main():
    batch_size = 5
    num_features = 28 * 28
    num_classes = 10
    x = torch.ones([batch_size, num_features])
    model = MLPv2([784, 1600, 10], True, True)
    print(model)
    print(model.gene_len())
    gene = torch.rand([model.gene_len()])
    model.set_gene(gene)
    y = model(x)
    print(y.size())


if __name__ == "__main__":
    main()

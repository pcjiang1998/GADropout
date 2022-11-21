import logging

import numpy as np
import torch.nn
from math import ceil
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.core.problem import ElementwiseProblem
from pymoo.factory import get_sampling, get_crossover, get_mutation, get_termination
from pymoo.optimize import minimize

import data_provider.pkl as pkl
from models.MLP import MLPv2
from test import test_top_k
from utils import RecurrentDataloader
from utils import init_dataloader

logging.basicConfig(level=logging.INFO)

pkl._folder = "/path/to/pkldataset"
device = torch.device("cuda")

dataset_name = "dataset_name or path"
batch_size = 32
train_loader, valid_loader, _ = init_dataloader(
    dataset_name,
    train_batch_size=batch_size,
    val_batch_size=batch_size,
    test_batch_size=batch_size,
)
_, _, test_loader = init_dataloader(dataset_name, normal=True)
_ = (
    train_loader.dataloader
    if isinstance(train_loader, RecurrentDataloader)
    else train_loader
).dataset
print(len(test_loader.dataset))
data, label = _[0]

########################################
num_samples = len(_)  # N
epochs = 200  # E
batch_size = batch_size  # B
pop_size = ceil(num_samples / batch_size)  # P
n_gen = epochs  # G
########################################

evaluate_times = epochs * ceil(num_samples / batch_size)
logging.info(f"Number of evaluations: {evaluate_times / 10000}W")
logging.info(f"Size of population: {pop_size}")
logging.info(f"Generations: {n_gen}")

del _
in_features = data.shape[0]

classes = 3
model = MLPv2([in_features, 1024, 1024, classes]).to(device)
logging.info("###############################")
logging.info(f"Info: {dataset_name}")
logging.info(f"{in_features=}")
logging.info(f"{classes=}")
logging.info(f"gene len={model.gene_len()}")
logging.info("###############################")
logging.info(f'{"gen":3s},{"lr":10s},{"loss":10s},{"top-1":5s}')

optimizer = torch.optim.SGD(
    model.parameters(), lr=1e-3, weight_decay=5e-5, momentum=0.9
)
gene_len = model.gene_len()
loss_fun = torch.nn.CrossEntropyLoss().to(device)


def do_every_generations(algorithm):
    global max_test_top1
    gen = algorithm.n_gen
    pop_var = algorithm.pop.get("X")
    pop_obj = algorithm.pop.get("F")
    result = test_top_k(model, device, test_loader, loss_fun, [1])
    lr = optimizer.param_groups[0]["lr"]
    logging.info(f"{gen:3d},{lr:10.4e},{result[0]:10.4e},{result[1][0]:5.2f}")
    max_test_top1 = max(max_test_top1, result[1][0])


def validate_one_batch():
    model.eval()
    x, y = valid_loader.get_data()
    x, y = x.float().to(device), y.to(device)
    predict = model(x)
    loss = loss_fun(predict, y).item()
    predicts = predict.argmax(dim=1)
    acc = torch.eq(predicts, y).sum().float().item()
    return loss, acc


def train_one_batch(gene):
    model.set_gene(gene)
    model.train()
    x, y = train_loader.get_data()
    x, y = x.float().to(device), y.to(device)
    predict = model(x)
    loss = loss_fun(predict, y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    loss = loss.item()
    predicts = predict.argmax(dim=1)
    acc = torch.eq(predicts, y).sum().float().item()
    return loss, acc


class DropoutProblem(ElementwiseProblem):
    def __init__(self, n_var, n_obj=2, xl=0.2, xu=1):
        super().__init__(n_var=n_var, n_obj=n_obj, xl=xl, xu=xu)

    def _evaluate(self, gene, out, *args, **kwargs):
        pre_loss, pre_acc = validate_one_batch()
        loss, acc = train_one_batch(gene)
        after_loss, after_acc = validate_one_batch()
        out["F"] = np.array([pre_acc - after_acc, sum(abs(gene - 0.5))])


max_test_top1 = 0
problem = DropoutProblem(n_var=gene_len, n_obj=2)
algorithm = NSGA2(
    pop_size=pop_size,
    n_offsprings=pop_size,
    sampling=get_sampling("real_random"),
    crossover=get_crossover("real_ux"),
    mutation=get_mutation("real_pm", eta=20),
    eliminate_duplicates=True,
)
termination = get_termination("n_gen", n_gen)
res = minimize(
    problem,
    algorithm,
    termination,
    seed=1,
    callback=do_every_generations,
    save_history=True,
    verbose=False,
)
logging.info(f"{max_test_top1=}")

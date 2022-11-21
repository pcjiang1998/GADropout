import torch.nn

import data_provider.pkl as pkl
from models.GADropout import GADropout
from models.MLP import MLPv2
from test import test_top_k
from utils import init_dataloader

pkl._folder = "/path/to/pkldataset"
device = torch.device("cuda")

dataset_name = "dataset_name or path"
batch_size = 32
classes = 3
epochs = 200
train_loader, valid_loader, test_loader = init_dataloader(
    dataset_name,
    train_batch_size=batch_size,
    val_batch_size=batch_size,
    test_batch_size=batch_size,
    normal=True,
)

data, label = train_loader.dataset[0]
in_features = data.shape[0]
model = MLPv2([in_features, 1024, 1024, classes])
model.classifier = GADropout.convert_to_normal(model.classifier, torch.nn.Identity)
for i in range(len(model.classifier)):
    if isinstance(model.classifier[i], torch.nn.Dropout):
        model.classifier[i] = torch.nn.Dropout(0.9)
model = model.to(device)
optimizer = torch.optim.SGD(
    model.parameters(), lr=1e-3, weight_decay=5e-5, momentum=0.9
)
loss_fun = torch.nn.CrossEntropyLoss().to(device)

print(model)
max_test_top1 = 0
for e in range(epochs):
    model.train()
    train_loss = 0
    correct = 0
    total = 0
    for x, y in train_loader:
        x, y = x.float().to(device), y.to(device)
        optimizer.zero_grad()
        predict = model(x)
        loss = loss_fun(predict, y)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        _, predicted = predict.max(1)
        total += y.size(0)
        correct += predicted.eq(y).sum().item()
    result = test_top_k(model, device, test_loader, loss_fun, [1])
    print(
        "{:3d}, {:10.4e}, {:10.4e}, {:5.2f}, {:10.4e}, {:5.2f}".format(
            e,
            optimizer.param_groups[0]["lr"],
            train_loss / len(train_loader),
            100.0 * correct / total,
            result[0],
            result[1][0],
        )
    )
    max_test_top1 = max(max_test_top1, result[1][0])
print(dataset_name)
print(max_test_top1)

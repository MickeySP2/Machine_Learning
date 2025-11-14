from pathlib import Path
import torchvision
from torch.utils.data import DataLoader
import torch
from network import neural_network
import torch.nn as nn
import os


cwd = Path.cwd()
src = Path.joinpath(cwd, "src")
data = Path.joinpath(cwd, "data")


def get_config_settings():
    #Open config.txt from specified path.
    with open(Path.joinpath(src, "config.txt")) as config:
        #Create list of settings from config file.
        settings = config.readlines()
        #Make lowercase, remove newlines and comma. "settings" is now a list of key-value pairs.
        settings = [item.lower().replace("\n", "").split(",") for item in settings]
        #Convert list of KvP to dict
        settings = dict(settings)
        #Should check that values are not None before returning

    return settings


def write_stats(stats:list[list]):
    count = 0
    fname = Path.joinpath(data, f"ml_stats_{count}.csv")
    #Check if file exists. If it exist increment count and try again
    while Path(fname).is_file():
        count += 1
        fname = Path.joinpath(data, f"ml_stats_{count}.csv")
        print(f"File exists, checking next: {fname}")
    
    #Open file for writing. Write the avg stats for each epoch.
    with open(fname, "w") as ml_stats:
        ml_stats.write("accuracy,avg_loss,epoch\n") 
        for acc, loss, t in stats:
            ml_stats.write(f"{t+1},{acc},{loss}\n")


def train_loop(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    # Set the model to training mode - important for batch normalization and dropout layers
    # Unnecessary in this situation but added for best practices
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        # Compute prediction and loss
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * int(settings.get("batch_size")) + len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


def test_loop(dataloader, model, loss_fn):
    # Set the model to evaluation mode - important for batch normalization and dropout layers
    # Unnecessary in this situation but added for best practices
    model.eval()
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0
    # Evaluating the model with torch.no_grad() ensures that no gradients are computed during test mode
    # also serves to reduce unnecessary gradient computations and memory usage for tensors with requires_grad=True
    with torch.no_grad():
        for X, y in dataloader:
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
    return f"{(100*correct):>0.1f}", f"{test_loss:>8f}"


settings = get_config_settings()


training_data = torchvision.datasets.FashionMNIST(root=data, train=True, transform=torchvision.transforms.ToTensor(), download=True)
test_data = torchvision.datasets.FashionMNIST(root=data, train=False, transform=torchvision.transforms.ToTensor(), download=True)


train_dataloader = DataLoader(training_data, batch_size=int(settings.get("batch_size")), shuffle=True)
test_dataloader = DataLoader(test_data, batch_size=int(settings.get("batch_size")), shuffle=True)

device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
model = neural_network().to(device)
optimizer = torch.optim.SGD(params = model.parameters(), lr = float(settings.get("learning_rate")))
loss_fn = nn.CrossEntropyLoss()

def test_train():
    #Test and train data. Write stats to csv file.
    stats = []
    for t in range(int(settings.get("epochs"))): 
        print(f"Epoch {t+1}\n-----------------") 
        train_loop(train_dataloader, model=model, loss_fn=loss_fn, optimizer=optimizer) 
        acc, loss = test_loop(test_dataloader, model=model, loss_fn=loss_fn)
        stats.append([acc, loss, t])
    write_stats(stats)


test_train()


model_file = "model.pt" 
model = neural_network() 
if os.path.exists(model_file): 
    model.load_state_dict(torch.load(model_file, weights_only=True))
    model.eval()
torch.save(model.state_dict(), model_file)
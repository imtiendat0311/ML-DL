import sys  # nopep8
sys.path.append('../')  # nopep8
import os
import torch.nn as nn
import torch
from torch.utils.tensorboard import SummaryWriter
import time
from model import VAE
from torch.optim import Adam
from torchvision.datasets import MNIST
import torchvision.transforms as transforms
from torch.utils.data import DataLoader


# default device
device = torch.device("cpu")
# for nvidia gpu ( cuda backend )
if (torch.cuda.is_available()):
    device = torch.device("cuda")
  # mps for apple M GPU ( metal backend )
elif (torch.backends.mps.is_available()):
    device = torch.device("mps")
print("Using device:", device)

# hyperparameters

lr = 1e-3
batch_size = 256
epochs = 100
beta = 0.5


# tensorboard writer

date = time.strftime("%d-%m-%Y--%H-%M-%S")
writer = SummaryWriter(
    f"runs/VAE-lr: {lr} - batch_size: {batch_size} - epochs: {epochs} - beta: {beta} - date: {date}")


# load dataset

dataset_path = '../data'

mnist_transform = transforms.Compose([
    transforms.ToTensor(),
])

kwargs = {'num_workers': 0, 'pin_memory': True}

train_dataset = MNIST(
    dataset_path, transform=mnist_transform, train=True, download=True)
test_dataset = MNIST(dataset_path, transform=mnist_transform,
                     train=False, download=True)

train_loader = DataLoader(dataset=train_dataset,
                          batch_size=batch_size, shuffle=True, **kwargs)
test_loader = DataLoader(dataset=test_dataset,
                         batch_size=batch_size, shuffle=False, **kwargs)

# init model and optimizer
vae = VAE(
    encoder=nn.Sequential(
        nn.Linear(784, 512),
        nn.ReLU(),
        nn.Linear(512, 256),
        nn.ReLU(),
    ),
    decoder=nn.Sequential(
        nn.Linear(128, 256),
        nn.ReLU(),
        nn.Linear(256, 512),
        nn.ReLU(),
        nn.Linear(512, 784),
        nn.Sigmoid()
    ),
    mean=nn.Linear(256, 128),
    var=nn.Linear(256, 128)
).to(device)

optimizer = Adam(vae.parameters(), lr=lr)


MSE = nn.MSELoss(reduction="sum").to(device)

# define loss function


def loss_function(x, x_hat, mean, log_var):
    reproduction_loss = MSE(x_hat, x)
    KLD = - beta * torch.sum(1 + log_var - mean.pow(2) - log_var.exp())
    return reproduction_loss, KLD


print("Start training VAE...")
vae.train()

for epoch in range(epochs):
    overall_loss = 0
    overall_rec = 0
    overall_kl = 0
    for batch_idx, (x, _) in enumerate(train_loader):
        batch_size = x.size(0)
        x = x.view(batch_size, 784)
        x = x.to(device)
        optimizer.zero_grad()
        _, mean, log_var, _, x_hat = vae(x)
        if batch_idx == 0:
            writer.add_images('input', x[0].view(
                1, 1, 28, 28), epoch+1)
            writer.add_images('reconstruction', x_hat[0].view(
                1, 1, 28, 28), epoch+1)
        loss_rec, loss_kld = loss_function(x, x_hat, mean, log_var)
        loss = loss_rec + loss_kld
        overall_rec += loss_rec.item()
        overall_kl += loss_kld.item()
        overall_loss += loss.item()
        loss.backward()
        optimizer.step()

    writer.add_scalar('Loss/total', overall_loss /
                      ((batch_idx)*batch_size), epoch+1)
    writer.add_scalar('Loss/rec', overall_rec /
                      ((batch_idx)*batch_size), epoch+1)
    writer.add_scalar('Loss/kl', overall_kl /
                      ((batch_idx)*batch_size), epoch+1)
    print("\tEpoch", epoch + 1, "complete!", "\tAverage Loss: ",
          overall_loss / (batch_idx*batch_size))

print("Finish!!")


vae.eval()

with torch.no_grad():
    for batch_idx, (x, _) in enumerate(test_loader):
        batch_size = x.size(0)
        x = x.view(batch_size, 784)
        x = x.to(device)
        _, _, _, _, x_hat = vae(x)
        writer.add_images('input_eval', x.view(
            batch_size, 1, 28, 28), epoch+1)
        writer.add_images('reconstruction_eval', x_hat.view(
            batch_size, 1, 28, 28), epoch+1)
        break
current = os.getcwd()
os.makedirs(f"./trained_model/vae_{date}", exist_ok=True)
torch.save(vae, f"./trained_model/vae_{date}/vae_{date}.pt")
torch.save(vae.state_dict(),
           f"./trained_model/vae_{date}/vae_{date}_state_dict.pt")
torch.save(optimizer.state_dict(),
           f"./trained_model/vae_{date}/optimizer_{date}_state_dict.pt")

writer.close()

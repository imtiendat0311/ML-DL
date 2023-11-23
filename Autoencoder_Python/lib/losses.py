from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision.datasets import MNIST
from torch.optim import Adam
from torchvision.utils import save_image, make_grid
from tqdm import tqdm
import numpy as np
import torch.nn as nn
import torch
from ray.tune.schedulers import ASHAScheduler
from ray import train, tune

EPOCH_SIZE = 512
TEST_SIZE = 256

dataset_path = '~/datasets'
batch_size = 100


device = torch.device("cude:0" if torch.cuda.is_available() else "cpu")


class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, activation_list, latent_dim):
        super(Encoder, self).__init__()

        self.activation = {
            "relu": nn.ReLU(),
            "sigmoid": nn.Sigmoid(),
            "tanh": nn.Tanh(),
            "identity": nn.Identity(),
            "softmax": nn.Softmax(dim=1),
            "leaky_relu": nn.LeakyReLU(),
        }
        self.activation_list = activation_list
        self.linear = []
        self.linear.append(nn.Linear(input_dim, hidden_dim[0]))
        for i in range(len(hidden_dim)-1):
            self.linear.append(nn.Linear(hidden_dim[i], hidden_dim[i+1]))
        self.mean = nn.Linear(hidden_dim[-1], latent_dim)
        self.log_var = nn.Linear(hidden_dim[-1], latent_dim)
        self.training = True

    def forward(self, x):
        for i in range(len(self.linear)):
            x = self.linear[i](x)
            x = self.activation[self.activation_list[i]](x)
        mean = self.mean(x)
        log_var = self.log_var(x)
        return mean, log_var


class Decoder(nn.Module):
    def __init__(self, latent_dim, hidden_dim, output_dim, activation_list):
        super(Decoder, self).__init__()

        self.activation = {
            "relu": nn.ReLU(),
            "sigmoid": nn.Sigmoid(),
            "tanh": nn.Tanh(),
            "identity": nn.Identity(),
            "softmax": nn.Softmax(dim=1),
            "leaky_relu": nn.LeakyReLU(),
        }

        self.activation_list = activation_list
        self.linear = []
        self.linear.append(nn.Linear(latent_dim, hidden_dim[0]))
        for i in range(len(hidden_dim)-1):
            self.linear.append(nn.Linear(hidden_dim[i], hidden_dim[i+1]))
        self.linear.append(nn.Linear(hidden_dim[-1], output_dim))
        self.training = True

    def forward(self, x):
        for i in range(len(self.linear)):
            x = self.linear[i](x)
            x = self.activation[self.activation_list[i]](x)
        return x


class Model(nn.Module):
    def __init__(self, encoder, decoder):
        super(Model, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def reparameterize(self, mean, log_var):
        epsilon = torch.randn_like(log_var).to(device)
        z = mean + log_var*epsilon
        return z

    def forward(self, x):
        mean, log_var = self.encoder(x)
        z = self.reparameterize(mean, torch.exp(0.5 * log_var))
        x_hat = self.decoder(z)
        return x_hat, mean, log_var


def loss_function(x, x_hat, mean, log_var):
    reconstruction_loss = nn.functional.binary_cross_entropy(
        x_hat, x, reduction="sum")
    kl_divergence = -0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp())
    return reconstruction_loss + kl_divergence


def train_func(model, optimizer, train_loader):
    model.train()
    for epoch in range(50):
        overall_loss = 0
        for batch_idx, (x, _) in enumerate(train_loader):
            if batch_idx * len(x) > EPOCH_SIZE:
                break
            x = x.view(batch_size, 784)
            x = x.to(device)
            optimizer.zero_grad()
            x_hat, mean, log_var = model(x)
            loss = loss_function(x, x_hat, mean, log_var)
            overall_loss += loss.item()
            loss.backward()
            optimizer.step()


def test_func(model, test_loader):
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for batch_idx, (x, _) in enumerate(test_loader):
            if batch_idx * len(x) > TEST_SIZE:
                break
            x = x.view(batch_size, 784)
            x = x.to(device)
            x_hat, mean, log_var = model(x)
            test_loss += loss_function(x, x_hat, mean, log_var).item()
    test_loss /= len(test_loader.dataset)
    return test_loss


def train_mnist(config):
    mnist_transforms = transforms.Compose([
        transforms.ToTensor(),
    ])

    train_loader = DataLoader(
        datasets.MNIST("~/data", train=True, download=True,
                       transform=mnist_transforms),
        batch_size=batch_size,
        shuffle=True)

    test_loader = DataLoader(
        datasets.MNIST("~/data", train=False, transform=mnist_transforms),
        batch_size=batch_size,

        shuffle=True)

    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = Model(
        Encoder(784, [512, 256], ["relu", "relu"], 200),
        Decoder(200, [256, 512], 784, ["relu", "relu", "sigmoid"])
    ).to(device)
    optimizer = Adam(model.parameters(), lr=config["lr"])
    for i in range(10):
        train_func(model, optimizer, train_loader)
        test_loss = test_func(model, test_loader)
        train.report({"mean_accuracy": test_loss})
        if i % 5 == 0:
            # This saves the model to the trial directory
            torch.save(model.state_dict(), "./model.pth")


search_space = {
    "lr": tune.sample_from(lambda spec: 10 ** (-10 * np.random.rand())),
}
datasets.MNIST("~/data", train=True, download=True)

tuner = tune.Tuner(
    train_mnist,
    param_space=search_space,
)

results = tuner.fit()

dfs = {result.path: result.metrics_dataframe for result in results}
[d.mean_accuracy.plot() for d in dfs.values()]

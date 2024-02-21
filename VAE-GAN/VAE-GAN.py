import sys  # nopep8
sys.path.append('../')  # nopep8
import os
import torch.nn as nn
import torch
from torch.utils.tensorboard import SummaryWriter
import time
from model import VAE, Discriminator
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


lr_g = 1e-3
lr_d = 1e-3
batch_size = 256
epochs = 100
alpha = 0.8
beta = 0.5
threshold = 0.5

date = time.strftime("%d-%m-%Y--%H-%M-%S")
writer = SummaryWriter(
    f"runs/VAE-lr-g: {lr_g} - lr-d: {lr_d} - batch_size: {batch_size} - epochs: {epochs} - alpha: {alpha} - beta: {beta} - threshold: {threshold} - date: {date}")


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

vae = VAE(encoder=nn.Sequential(
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
    var=nn.Linear(256, 128)).to(device)
dis = Discriminator(
    nn.Sequential(
        nn.Linear(784, 512),
        nn.ReLU(),
        nn.Linear(512, 256),
        nn.ReLU(),
        nn.Linear(256, 1),
        nn.Sigmoid()
    )
).to(device)

vae_opt = torch.optim.Adam(vae.parameters(), lr=lr_g)
dis_opt = torch.optim.Adam(dis.parameters(), lr=lr_d)

print("Start training VAE GAN...")
vae.train()
dis.train()
BCE = nn.BCELoss(reduction='sum').to(device)
MSE = nn.MSELoss(reduction='sum').to(device)
for epoch in range(epochs):
    overall_gen_loss = 0
    overall_dis_loss = 0
    overall_rec_loss = 0
    overall_kl_loss = 0
    overall_dis_loss_fake = 0
    overall_dis_loss_real = 0
    for batch_idx, (x, _) in enumerate(train_loader):
        batch_size = x.size(0)
        x = x.view(batch_size, 784)
        x = x.to(device)
        real_label = torch.ones(batch_size, 1).to(device)
        fake_label = torch.zeros(batch_size, 1).to(device)

        # Train Discriminator
        _, mean, log_var, _, x_hat = vae(x)
        if batch_idx == 0:
            writer.add_images('input', x[0].view(
                1, 1, 28, 28), epoch+1)
            writer.add_images('reconstruction', x_hat[0].view(
                1, 1, 28, 28), epoch+1)
        dis_real = dis(x)
        dis_fake = dis(x_hat.detach())
        dis_loss_real = BCE(dis_real, real_label)
        dis_loss_fake = BCE(dis_fake, fake_label)
        dis_loss = dis_loss_real + dis_loss_fake
        overall_dis_loss += dis_loss.item()
        overall_dis_loss_real += dis_loss_real.item()
        overall_dis_loss_fake += dis_loss_fake.item()
        dis_opt.zero_grad()
        dis_loss.backward()
        dis_opt.step()

        if (dis_loss / (batch_idx)*batch_size*2) < threshold:
            # Train Generator
            dis_fake = dis(x_hat)
            dis_loss_fake = BCE(dis_fake, real_label)
            rec_loss = MSE(x_hat, x)
            overall_rec_loss += rec_loss.item()
            kl = -beta * torch.sum(1 + log_var - mean.pow(2) - log_var.exp())
            overall_kl_loss += kl.item()
            vae_loss = rec_loss + kl + alpha*dis_loss_fake
            overall_gen_loss += vae_loss.item()
            vae_opt.zero_grad()
            vae_loss.backward()
            vae_opt.step()

    writer.add_scalar('Generator Loss', overall_gen_loss /
                      ((batch_idx)*batch_size), epoch+1)
    writer.add_scalar('Discriminator Loss', overall_dis_loss /
                      ((batch_idx)*batch_size*2), epoch+1)
    writer.add_scalar('Reconstruction Loss', overall_rec_loss /
                      ((batch_idx)*batch_size), epoch+1)
    writer.add_scalar('KL Loss', overall_kl_loss /
                      ((batch_idx)*batch_size), epoch+1)
    writer.add_scalar('Discriminator Loss Real', overall_dis_loss_real /
                      ((batch_idx)*batch_size), epoch+1)
    writer.add_scalar('Discriminator Loss Fake', overall_dis_loss_fake /
                      ((batch_idx)*batch_size), epoch+1)
    print("Epoch: {}/{} Generator Loss: {} Discriminator Loss:{}".format(epoch, epochs,
          overall_gen_loss / (batch_idx*batch_size), overall_dis_loss/(batch_idx*batch_size*2)))

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
os.makedirs(f"./trained_model/vae_gan_{date}", exist_ok=True)
torch.save(vae, f"./trained_model/vae_gan_{date}/vae_{date}.pt")
torch.save(dis, f"./trained_model/vae_gan_{date}/dis_{date}.pt")
torch.save(vae.state_dict(),
           f"./trained_model/vae_gan_{date}/vae_{date}_state_dict.pt")
torch.save(dis.state_dict(),
           f"./trained_model/vae_gan_{date}/dis_{date}_state_dict.pt")
torch.save(vae_opt.state_dict(),
           f"./trained_model/vae_gan_{date}/vae_optimizer_{date}_state_dict.pt")
torch.save(dis_opt.state_dict(),
           f"./trained_model/vae_gan_{date}/dis_optimizer_{date}_state_dict.pt")
writer.close()

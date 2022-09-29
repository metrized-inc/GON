import torch
import torch.nn as nn
import torchvision
import numpy as np
import os
from torch.utils.data import Dataset
import torchvision.transforms as T
from PIL import Image

plot_dir = 'imgs'
os.makedirs(plot_dir, exist_ok=True)

# image data
dataset_name = 'custom' # ['mnist', 'fashion', 'custom]
# dataset_name = 'fashion'
if dataset_name == 'custom':
    data_path = r'E:\Metrized-Data\Sikorsky\material_classifier\sikorsky_only\combined\only_carbon\split_temp'
    img_size = 32
    n_channels = 3
    nc = n_channels
    img_coords = 2
else:
    img_size = 32
    n_channels = 1
    nc = n_channels
    img_coords = 2

# training info
lr = 1e-4
batch_size = 8
nz = 32
ngf = 16
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# create GON network
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            nn.ConvTranspose2d(nz, ngf * 4, 4, 1, 0, bias=True),
            nn.BatchNorm2d(ngf * 4),
            nn.ELU(),
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=True),
            nn.BatchNorm2d(ngf * 2),
            nn.ELU(),
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=True),
            nn.BatchNorm2d(ngf),
            nn.ELU(),
            nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=True),
            nn.Sigmoid()
        )

    def forward(self, input):
        return self.main(input)

def cycle(iterable):
    while True:
        for x in iterable:
            yield x

def load_custom_dataset(data_path):
    train_path = os.path.join(data_path, 'train')
    # val_path = os.path.join(data_path, 'val')

    train_names = os.listdir(train_path)
    # val_names = os.listdir(val_path)

    train_img_paths = [os.path.join(train_path, name) for name in train_names]
    # val_img_paths = [os.path.join(val_path, name) for name in val_names]
    return train_img_paths#, val_img_paths

class CustomDataset(Dataset):
    def __init__(self, data_paths, transform):
        self.data_paths = data_paths
        self.transform = transform

    def __getitem__(self, index):
        image = Image.open(self.data_paths[index])
        image = self.transform(image)
        return image, 0

    def __len__(self):
        return len(self.data_paths)

def _data_transforms_custom():
    if n_channels == 1:
        transform = T.Compose([
            T.Resize(size=(img_size, img_size), interpolation=Image.Resampling.BICUBIC),
            T.Grayscale(),
            T.ToTensor()
        ])
    else:
        transform = T.Compose([
            T.Resize(size=(img_size, img_size), interpolation=Image.Resampling.BICUBIC),
            T.ToTensor()
        ])

    return transform


# load datasets
if dataset_name == 'mnist':
    dataset = torchvision.datasets.MNIST('data', train=True, download=True, transform=torchvision.transforms.Compose([
        torchvision.transforms.Resize(img_size), torchvision.transforms.ToTensor()
    ]))
if dataset_name == 'fashion':
    dataset = torchvision.datasets.FashionMNIST('data', train=True, download=True, transform=torchvision.transforms.Compose([
        torchvision.transforms.Resize(img_size), torchvision.transforms.ToTensor()
    ]))
if dataset_name == 'custom':
    transforms = _data_transforms_custom()
    # train_img_paths, val_img_paths = load_custom_dataset(data_path)
    train_img_paths = load_custom_dataset(data_path)
    train_data = CustomDataset(train_img_paths, transforms)
    # valid_data = CustomDataset(val_img_paths, transforms)

if dataset_name == 'custom':
    train_loader = torch.utils.data.DataLoader(train_data, sampler=None, shuffle=True, batch_size=batch_size, drop_last=True)
else:
    train_loader = torch.utils.data.DataLoader(dataset, sampler=None, shuffle=True, batch_size=batch_size, drop_last=True)

train_iterator = iter(cycle(train_loader))

F = Generator().to(device)

optim = torch.optim.Adam(lr=lr, params=F.parameters())
print(f'> Number of parameters {len(torch.nn.utils.parameters_to_vector(F.parameters()))}')

for step in range(1001):
    # sample a batch of data
    x, t = next(train_iterator)
    x, t = x.to(device), t.to(device)

    # compute the gradients of the inner loss with respect to zeros (gradient origin)
    z = torch.zeros(batch_size, nz, 1, 1).to(device).requires_grad_()
    g = F(z)
    inner_loss = ((g - x)**2).sum(1).mean()
    grad = torch.autograd.grad(inner_loss, [z], create_graph=True, retain_graph=True)[0]
    z = (-grad)

    # now with z as our new latent points, optimise the data fitting loss
    g = F(z)
    outer_loss = ((g - x)**2).sum(1).mean()
    optim.zero_grad()
    outer_loss.backward()
    optim.step()

    if step % 100 == 0 and step > 0:    
        print(f"Step: {step}  Loss: {outer_loss.item():.3f}")

        # plot reconstructions
        torchvision.utils.save_image(torch.clamp(g, 0, 1), f'imgs/recon_{step}.png', 
            nrow=int(np.sqrt(batch_size)), padding=0)

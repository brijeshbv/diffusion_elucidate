from k_diffusion import models
import torch
from copy import deepcopy
import json
import math
from pathlib import Path
#import accelerate
import torch
from torch import optim
from torch import multiprocessing as mp
from torch.utils import data
from torchvision import transforms, utils as tv_utils
from tqdm import trange
import datasets # HF version
from matplotlib import pyplot as plt
from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset
from k_diffusion import evaluation, gns, layers, models, sampling, utils
import tqdm
import wandb
wandb.login()

class CelebADataset(Dataset):
    def __init__(self, img_size=128):
        self.dataset = load_dataset('huggan/CelebA-faces', split='train')
        self.preprocess = transforms.Compose([transforms.ToTensor(),transforms.Resize(img_size), transforms.CenterCrop(img_size)])
    def __len__(self):
        return len(self.dataset)
    def __getitem__(self, idx):
        x = self.dataset[idx]
        return self.preprocess(x['image']) * 2 - 1 # Images scaled to (-1, 1)

img_size = 64
batch_size=16
dataset = CelebADataset(img_size)
dl = DataLoader(dataset, batch_size=batch_size)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

inner_model_edm = models.ImageDenoiserModelV1(
    3, # input channels
    256, # mapping out
    [2, 2, 4], # depths
    [64, 128, 256], # channels
    [False, True, True] # self attention
).to(device)


inner_model_vpsong = models.SongUNet(
    64,
    3,
    3
).to(device)

batch = next(iter(dl))

utils.to_pil_image(batch[3]) # View the first image

# Duplicate the first image 12 times for demo purposes:
input_images = batch[3].unsqueeze(0).repeat(12, 1, 1, 1)

# Add noise (linearly)
reals = input_images.to(device)
noise = torch.randn_like(reals)
sigma = torch.linspace(0, 3, 12).to(device) # Gradually increasing noise
noised_input = reals + noise * utils.append_dims(sigma, reals.ndim)

# View the result
utils.to_pil_image(tv_utils.make_grid(noised_input))

sigma_mean, sigma_std = -1.2, 1.2
reals = input_images.to(device)
noise = torch.randn_like(reals)
sigma = torch.distributions.LogNormal(sigma_mean, sigma_std).sample([reals.shape[0]]).to(device)
noised_input = reals + noise * utils.append_dims(sigma, reals.ndim)

utils.to_pil_image(tv_utils.make_grid(noised_input))


inner_model_edm_output = inner_model_edm(noised_input, sigma)
print(inner_model_edm_output.shape)

inner_model_vpsong_output = inner_model_vpsong(noised_input, sigma)
print(inner_model_vpsong_output.shape)

outer_model_edm = layers.Denoiser(inner_model_edm, sigma_data=0.5).to(device)
edm_model_output = outer_model_edm(noised_input, sigma)

edm_loss = outer_model_edm.loss(noised_input, noise, sigma)


outer_model_vpsong = layers.DenoiserVPScore(inner_model_vpsong, sigma_data=0.5).to(device)
vpsong_model_output = outer_model_vpsong(noised_input, sigma)

vpsong_loss = outer_model_vpsong.loss(noised_input, noise, sigma)


# Setup the model, optimizer and sceduler
inner_model_edm = models.SongUNet(
    64, # Image resolution at input/output.
    3,  # Number of color channels at input.
    3  # Number of color channels at input.
).to(device)
opt = optim.AdamW(inner_model_edm.parameters(), lr=2e-4, betas=(0.95, 0.999), eps=1e-6, weight_decay=1e-3)
sched = utils.InverseLR(opt, inv_gamma=50000, power=1/2, warmup=0.99)
model = layers.DenoiserVPScore(inner_model_edm, sigma_data=0.5).to(device)


epoch = 0
step = 0
size = [64, 64]

wandb.init(project='vp-diffusion-demo')
sigma_max = 80
sigma_min = 1e-2
@torch.no_grad()
@utils.eval_mode(model)
def demo():
    #tqdm.write('Sampling...')
    filename = f'demo_{step:08}.png'
    n_per_proc = 16
    x = torch.randn([n_per_proc, 3, size[0], size[1]], device=device) * sigma_max
    sigmas = sampling.get_sigmas_karras(50, sigma_min, sigma_max, rho=7., device=device)
    x_0 = sampling.sample_lms(model, x, sigmas)
    grid = tv_utils.make_grid(x_0, nrow=4, padding=0)
    utils.to_pil_image(grid).save(filename)
    wandb.log({'vp_demo_grid': wandb.Image(filename)}, step=step)

for epoch in range(1):
  for batch in dl:
    opt.zero_grad()
    reals = batch.to(device)
    noise = torch.randn_like(reals)
    sigma = torch.distributions.LogNormal(sigma_mean, sigma_std).sample([reals.shape[0]]).to(device)
    loss = model.loss(reals, noise, sigma).mean()
    loss.backward()
    opt.step()
    sched.step()

    wandb.log({'epoch': epoch,'loss': loss.item(),'lr': sched.get_last_lr()[0]}, step=step)

    # if step % 50 == 0:
    #   tqdm.write(f'Epoch: {epoch}, step: {step}, loss: {loss.item():g}')
    
    if step % 250 == 0:
        demo()
    
    step += 1
    if step % 1000 == 0:
        torch.save(model.state_dict(), 'vp_model_faces64_1e.pt')


torch.save(model.state_dict(), 'vp_model_faces64_1e.pt')
wandb.finish()
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


from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
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
batch_size=8
dataset = CelebADataset(img_size)
dl = DataLoader(dataset, batch_size=batch_size)
batch = next(iter(dl))
print('Batch shape:', batch.shape)
utils.to_pil_image(batch[3]) # View the first image
# Duplicate the first image 8 times for demo purposes:
input_images = batch[3].unsqueeze(0).repeat(8, 1, 1, 1)

# Add noise (linearly)
reals = input_images.to(device)
noise = torch.randn_like(reals)
sigma = torch.linspace(0, 3, 8).to(device) # Gradually increasing noise
noised_input = reals + noise * utils.append_dims(sigma, reals.ndim)

# View the result
utils.to_pil_image(tv_utils.make_grid(noised_input))
# Plot the lognormal distribution:
sigma_mean, sigma_std = -1.2, 1.2
fig, axs = plt.subplots(1, 2, figsize=(8, 4))
samples = torch.distributions.LogNormal(sigma_mean, sigma_std).sample([10000])
print('Min:', samples.min(), 'Max:', samples.max())
# axs[0].hist([s for s in samples if s < 5], bins=40)
# axs[0].set_title('Distribution (where sigma < 5)')
# axs[1].hist([s for s in samples if s < 1], bins=40)
# axs[1].set_title('Distribution (where sigma < 1)')
# plt.show()

# Add noise, drawing from the specified noise distribution:
sigma_mean, sigma_std = -1.2, 1.2
reals = input_images.to(device)
noise = torch.randn_like(reals)
sigma = torch.distributions.LogNormal(sigma_mean, sigma_std).sample([reals.shape[0]]).to(device)
noised_input = reals + noise * utils.append_dims(sigma, reals.ndim)
utils.to_pil_image(tv_utils.make_grid(noised_input))

inner_model_vpsong = models.SongUNet(
    64,
    3,
    3
).to(device)

inner_model_vpsong_output = inner_model_vpsong(noised_input, sigma)
print(inner_model_vpsong_output.shape)

size = [64, 64]
inner_model_vpsong = models.SongUNet(
    64,
    3,
    3
).to(device)
## Load the VP model.
vpsong_model = layers.DenoiserVPScore(inner_model_vpsong, sigma_data=0.5).to(device)
vpsong_model.load_state_dict(torch.load('./pre_trained/vp_model_faces64_1e.pt',map_location=device))

inner_model_edm = models.ImageDenoiserModelV1(
    3, # input channels
    256, # mapping out
    [2, 2, 2, 4], # depths
    [64, 128, 128, 256], # channels
    [False, False, True, True] # self attention
).to(device)

## Load the EDM model
edm_model = layers.Denoiser(inner_model_edm, sigma_data=0.5).to(device)
edm_model.load_state_dict(torch.load('./pre_trained/model_faces64_1e.pt',map_location=device))


# Plot the noise schedule used when sampling:
sigma_min, sigma_max = 1e-2, 80


# # VP Sample from the LMS sampler

x = torch.randn([16, 3, size[0], size[1]], device=device) * sigma_max
sigmas = sampling.get_sigmas_karras(10, sigma_min, sigma_max, rho=7., device=device)
x_0 = sampling.sample_lms(vpsong_model, x, sigmas)
grid = tv_utils.make_grid(x_0, nrow=4, padding=0)
utils.to_pil_image(grid).save("./demos/VP_lms_sampler.png")

# # VP Sample from the DDPM sampler

x = torch.randn([16, 3, size[0], size[1]], device=device) * sigma_max
sigmas = sampling.get_sigmas_karras(10, sigma_min, sigma_max, rho=7., device=device)
x_0 = sampling.sample_dpmpp_2m(vpsong_model, x, sigmas)
grid = tv_utils.make_grid(x_0, nrow=4, padding=0)
utils.to_pil_image(grid).save("./demos/VP_DDMP_sampler.png")

# # VP Sample  from the Heun sampler

x = torch.randn([16, 3, size[0], size[1]], device=device) * sigma_max
sigmas = sampling.get_sigmas_karras(10, sigma_min, sigma_max, rho=7., device=device)
x_0 = sampling.sample_heun(vpsong_model, x, sigmas)
grid = tv_utils.make_grid(x_0, nrow=4, padding=0)
utils.to_pil_image(grid).save("./demos/VP_heun_sampler.png")

# # EDM Sample from the LMS sampler

x = torch.randn([16, 3, size[0], size[1]], device=device) * sigma_max
sigmas = sampling.get_sigmas_karras(10, sigma_min, sigma_max, rho=7., device=device)
x_0 = sampling.sample_lms(edm_model, x, sigmas)
grid = tv_utils.make_grid(x_0, nrow=4, padding=0)
utils.to_pil_image(grid).save("./demos/edm_lms_sampler.png")

# # EDM Sample from the DDPM sampler

x = torch.randn([16, 3, size[0], size[1]], device=device) * sigma_max
sigmas = sampling.get_sigmas_karras(10, sigma_min, sigma_max, rho=7., device=device)
x_0 = sampling.sample_dpmpp_2m(edm_model, x, sigmas)
grid = tv_utils.make_grid(x_0, nrow=4, padding=0)
utils.to_pil_image(grid).save("./demos/edm_DDMP_sampler.png")

# EDM Sample from the Heun sampler

x = torch.randn([16, 3, size[0], size[1]], device=device) * sigma_max
sigmas = sampling.get_sigmas_karras(10, sigma_min, sigma_max, rho=7., device=device)
x_0 = sampling.sample_heun(edm_model, x, sigmas)
grid = tv_utils.make_grid(x_0, nrow=4, padding=0)
utils.to_pil_image(grid).save("./demos/edm_heun_sampler.png")


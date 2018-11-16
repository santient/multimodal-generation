import os
import glob
import sys
import torch
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F
import torchvision
from torchvision import transforms
from torchvision.utils import save_image
from torchvision import datasets
from torch.utils.data import Dataset, DataLoader
from torch import optim
from torch import nn
from torch.backends import cudnn
import skimage.io
from comet_ml import Experiment

# experiment = Experiment(api_key="E3oWJUSFulpXpCUQfc5oGz0zY", project_name="obama_avb")

cudnn.benchmark = True
device = torch.device("cuda:2")
torch.cuda.set_device(2)

img_size = 256
channels = 3
seq_len = 20
img_latent_dim = 64
seq_latent_dim = 64

class VideoDataset(Dataset):
    def __init__(self, root_dir, seq_len, transform=None):
        self.root_dir = root_dir
        self.seq_len = seq_len
        self.transform = transform
        self.filenames = sorted(glob.glob(os.path.join(root_dir, "*.png")))
    def __len__(self):
        return len(self.filenames) - (self.seq_len - 1)
    def __getitem__(self, idx):
        images = [skimage.io.imread(self.filenames[idx+i]) for i in range(self.seq_len)]
        if self.transform:
            images = list(map(self.transform, images))
        else:
            images = list(map(transforms.ToTensor(), images))
        return torch.stack(images)

dataset = VideoDataset("/home/santiago/Downloads/obama/images/", 20, transform=transforms.Compose([
    transforms.ToPILImage(),
    transforms.CenterCrop(1024),
    transforms.Resize(img_size),
    transforms.ToTensor()
]))

batch_size = 4
workers = 4
dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=workers)

class ImageAVB(nn.Module):
    def __init__(self, img_size, channels, latent_dim):
        super(ImageAVB, self).__init__()
        
        self.img_size = img_size
        self.channels = channels
        self.latent_dim = latent_dim
        self.ds_size = img_size // 2**5
        
        self.gen_proj = nn.Linear(self.latent_dim, 256*self.ds_size**2)
        self.gen_blocks = nn.Sequential(
            nn.BatchNorm2d(256),
            nn.ConvTranspose2d(256, 256, 3, stride=2, padding=1, output_padding=1),
            nn.ConvTranspose2d(256, 128, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            
            nn.BatchNorm2d(128, 0.8),
            nn.ConvTranspose2d(128, 128, 3, stride=2, padding=1, output_padding=1),
            nn.ConvTranspose2d(128, 64, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            
            nn.BatchNorm2d(64, 0.8),
            nn.ConvTranspose2d(64, 64, 3, stride=2, padding=1, output_padding=1),
            nn.ConvTranspose2d(64, 32, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            
            nn.BatchNorm2d(32, 0.8),
            nn.ConvTranspose2d(32, 32, 3, stride=2, padding=1, output_padding=1),
            nn.ConvTranspose2d(32, 16, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            
            nn.BatchNorm2d(16, 0.8),
            nn.ConvTranspose2d(16, 16, 3, stride=2, padding=1, output_padding=1),
            nn.ConvTranspose2d(16, self.channels, 3, stride=1, padding=1),
            nn.Sigmoid()
        )
        
        self.enc_proj = nn.Linear(self.latent_dim, img_size**2)
        self.enc_blocks = nn.Sequential(
            nn.Conv2d(self.channels+1, 16, 3, 1, 1),
            nn.Conv2d(16, 16, 3, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(0.25),
            nn.BatchNorm2d(16, 0.8),
            
            nn.Conv2d(16, 32, 3, 1, 1),
            nn.Conv2d(32, 32, 3, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(0.25),
            nn.BatchNorm2d(32, 0.8),
            
            nn.Conv2d(32, 64, 3, 1, 1),
            nn.Conv2d(64, 64, 3, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(0.25),
            nn.BatchNorm2d(64, 0.8),
            
            nn.Conv2d(64, 128, 3, 1, 1),
            nn.Conv2d(128, 128, 3, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(0.25),
            nn.BatchNorm2d(128, 0.8),
            
            nn.Conv2d(128, 256, 3, 1, 1),
            nn.Conv2d(256, 256, 3, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(0.25),
            nn.BatchNorm2d(256, 0.8)
        )
        self.enc_layer = nn.Linear(256*self.ds_size**2, self.latent_dim)
        
        self.dis_proj = nn.Linear(self.latent_dim, self.img_size**2)
        self.dis_blocks = nn.Sequential(
            nn.Conv2d(self.channels+1, 16, 3, 1, 1),
            nn.Conv2d(16, 16, 3, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(0.25),
            nn.BatchNorm2d(16, 0.8),
            
            nn.Conv2d(16, 32, 3, 1, 1),
            nn.Conv2d(32, 32, 3, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(0.25),
            nn.BatchNorm2d(32, 0.8),
            
            nn.Conv2d(32, 64, 3, 1, 1),
            nn.Conv2d(64, 64, 3, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(0.25),
            nn.BatchNorm2d(64, 0.8),
            
            nn.Conv2d(64, 128, 3, 1, 1),
            nn.Conv2d(128, 128, 3, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(0.25),
            nn.BatchNorm2d(128, 0.8),
            
            nn.Conv2d(128, 256, 3, 1, 1),
            nn.Conv2d(256, 256, 3, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(0.25),
            nn.BatchNorm2d(256, 0.8)
        )
        self.dis_layer = nn.Sequential(
            nn.Linear(256*self.ds_size**2, 1),
            nn.Sigmoid()
        )
        
    def sample_prior(self, s):
        if self.training:
            m = torch.zeros((s.data.shape[0], self.latent_dim))
            std = torch.ones((s.data.shape[0], self.latent_dim))
            d = Variable(torch.normal(m,std))
        else:
            d = Variable(torch.zeros((s.data.shape[0], self.latent_dim)))
        return d.cuda()
    
    def discriminator(self, x, z):
        z_proj = self.dis_proj(z)
        z_proj = z_proj.view(z_proj.shape[0], 1, self.img_size, self.img_size)
        i = torch.cat((x, z_proj), dim=1)
        h = self.dis_blocks(i)
        h = h.view(h.shape[0], 256*self.ds_size**2)
        out = self.dis_layer(h)
        return out
    
    def sample_posterior(self, x):
        prior_proj = self.enc_proj(self.sample_prior(x))
        prior_proj = prior_proj.view(prior_proj.shape[0], 1, self.img_size, self.img_size)
        i = torch.cat((x, prior_proj), dim=1)
        h = self.enc_blocks(i)
        h = h.view(h.shape[0], 256*self.ds_size**2)
        out = self.enc_layer(h)
        return out
    
    def decoder(self, z):
        z_proj = self.gen_proj(z)
        z_proj = z_proj.view(z_proj.shape[0], 256, self.ds_size, self.ds_size)
        out = self.gen_blocks(z_proj)
        return out
    
    def forward(self, x):
        z_p = self.sample_prior(x)
        z_q = self.sample_posterior(x)
        
        log_d_prior = self.discriminator(x, z_p)
        log_d_posterior = self.discriminator(x, z_q)
        
        dis_loss = torch.mean(
            torch.nn.functional.binary_cross_entropy_with_logits(
                log_d_posterior, torch.ones_like(log_d_posterior))
            + torch.nn.functional.binary_cross_entropy_with_logits(
                log_d_prior, torch.zeros_like(log_d_prior)))
        
        x_recon = self.decoder(z_q)
        recon_likelihood = -torch.nn.functional.binary_cross_entropy(
                                                x_recon, x)*x.data.shape[0]
        
        gen_loss = torch.mean(log_d_posterior)-torch.mean(recon_likelihood)
        
        return z_p, z_q, dis_loss, gen_loss

class SequenceAVB(nn.Module):
    def __init__(self, vector_dim, seq_len, latent_dim):
        super(SequenceAVB, self).__init__()
        
        self.vector_dim = vector_dim
        self.seq_len = seq_len
        self.latent_dim = latent_dim
        
        self.gen_lstm = nn.LSTM(self.vector_dim, self.latent_dim, num_layers=2, dropout=0.25)
        
        self.enc_lstm = nn.LSTM(self.vector_dim, self.latent_dim, num_layers=2, dropout=0.25)
        
        self.dis_lstm = nn.LSTM(self.vector_dim, self.latent_dim, num_layers=2, dropout=0.25)
        self.dis_layer = nn.Sequential(
            nn.Linear(4*self.latent_dim, 1),
            nn.Sigmoid()
        )
        
        self.sigmoid = nn.Sigmoid()
        
    def sample_prior(self, s):
        if self.training:
            m = torch.zeros((2, s.data.shape[1], self.latent_dim))
            std = torch.ones((2, s.data.shape[1], self.latent_dim))
            d = (Variable(torch.normal(m,std)).cuda(), Variable(torch.normal(m,std)).cuda())
        else:
            d = (Variable(torch.zeros((2, s.data.shape[1], self.latent_dim))).cuda(), Variable(torch.zeros((2, s.data.shape[1], self.latent_dim))).cuda())
        return d
    
    def discriminator(self, x, z):
        _, states = self.dis_lstm(x, z)
        h = torch.cat((states[0].view(states[0].shape[1], 2*self.latent_dim), states[1].view(states[1].shape[1], 2*self.latent_dim)), dim=1)
        out = self.dis_layer(h)
        return out
    
    def sample_posterior(self, x):
        _, states = self.enc_lstm(x, self.sample_prior(x))
        return (torch.stack([states[0][1], states[0][0]]), torch.stack([states[1][1], states[1][0]]))
    
    def decoder(self, z, num_steps):
        step = Variable(torch.zeros(1, z[0].shape[1], self.latent_dim)).cuda()
        decoded = []
        for i in range(num_steps):
            step, z = self.gen_lstm(step, z)
            decoded.append(step[0])
        return torch.stack(list(reversed(decoded)))
    
    def forward(self, x):
        z_p = self.sample_prior(x)
        z_q = self.sample_posterior(x)
        
        log_d_prior = self.discriminator(x, z_p)
        log_d_posterior = self.discriminator(x, z_q)
        
        dis_loss = torch.mean(
            torch.nn.functional.binary_cross_entropy_with_logits(
                log_d_posterior, torch.ones_like(log_d_posterior))
            + torch.nn.functional.binary_cross_entropy_with_logits(
                log_d_prior, torch.zeros_like(log_d_prior)))
        
        x_recon = self.decoder(z_q, self.seq_len)
        recon_likelihood = -torch.nn.functional.binary_cross_entropy(
                                                self.sigmoid(x_recon), self.sigmoid(x))*x.data.shape[1]
        
        gen_loss = torch.mean(log_d_posterior)-torch.mean(recon_likelihood)
        
        return z_p, z_q, dis_loss, gen_loss

class VideoAVB(nn.Module):
    def __init__(self, img_size, channels, img_latent_dim, seq_len, seq_latent_dim):
        super(VideoAVB, self).__init__()
        
        self.img_size = img_size
        self.channels = channels
        self.img_latent_dim = img_latent_dim
        self.seq_len = seq_len
        self.seq_latent_dim = seq_latent_dim
        self.ds_size = img_size // 2**5
        
        self.img_model = ImageAVB(self.img_size, self.channels, self.img_latent_dim)
        self.seq_model = SequenceAVB(self.img_latent_dim, self.seq_len, self.seq_latent_dim)
        
        self.dis_proj = nn.Linear(self.img_latent_dim, self.img_size**2)
        self.dis_blocks = nn.Sequential(
            nn.Conv2d(self.channels+1, 16, 3, 1, 1),
            nn.Conv2d(16, 16, 3, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(0.25),
            nn.BatchNorm2d(16, 0.8),
            
            nn.Conv2d(16, 32, 3, 1, 1),
            nn.Conv2d(32, 32, 3, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(0.25),
            nn.BatchNorm2d(32, 0.8),
            
            nn.Conv2d(32, 64, 3, 1, 1),
            nn.Conv2d(64, 64, 3, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(0.25),
            nn.BatchNorm2d(64, 0.8),
            
            nn.Conv2d(64, 128, 3, 1, 1),
            nn.Conv2d(128, 128, 3, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(0.25),
            nn.BatchNorm2d(128, 0.8),
            
            nn.Conv2d(128, 256, 3, 1, 1),
            nn.Conv2d(256, 256, 3, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(0.25),
            nn.BatchNorm2d(256, 0.8)
        )
        self.dis_rep = nn.Linear(256*self.ds_size**2, self.img_latent_dim)
        self.dis_lstm = nn.LSTM(self.img_latent_dim, self.seq_latent_dim, num_layers=2, dropout=0.25)
        self.dis_layer = nn.Sequential(
            nn.Linear(4*self.seq_latent_dim, 1),
            nn.Sigmoid()
        )
        
    def sample_prior(self, s):
        frame_p = []
        for frame in s.split(1):
            frame_p.append(self.img_model.sample_prior(frame[0]))
        return self.seq_model.sample_prior(torch.stack(frame_p))
    
    def discriminator(self, x, z_img, z_seq):
        reps = []
        for frame, z in zip(x.split(1), z_img.split(1)):
            z_proj = self.dis_proj(z[0])
            z_proj = z_proj.view(z_proj.shape[0], 1, self.img_size, self.img_size)
            i = torch.cat((frame[0], z_proj), dim=1)
            h = self.dis_blocks(i)
            h = h.view(h.shape[0], 256*self.ds_size**2)
            rep = self.dis_rep(h)
            reps.append(rep)
        _, states = self.dis_lstm(torch.stack(reps), z_seq)
        h = torch.cat((states[0].view(states[0].shape[1], 2*self.seq_latent_dim), states[1].view(states[1].shape[1], 2*self.seq_latent_dim)), dim=1)
        out = self.dis_layer(h)
        return out
    
    def sample_posterior(self, x):
        frame_q = []
        for frame in s.split(1):
            frame_q.append(self.img_model.sample_posterior(frame[0]))
        return self.seq_model.sample_posterior(torch.stack(frame_q))
    
    def decoder(self, z, num_steps):
        reps = self.seq_model.decoder(z, num_steps)
        frames = []
        for rep in reps.split(1):
            frame = self.img_model.decoder(rep[0])
            frames.append(frame)
        return torch.stack(frames)
    
    def forward(self, x):
        z_p_img = []
        z_q_img = []
        img_dis_loss =  []
        img_gen_loss = []
        for frame in x.split(1):
            z_p, z_q, dis_loss, gen_loss = self.img_model.forward(frame[0])
            z_p_img.append(z_p)
            z_q_img.append(z_q)
            img_dis_loss.append(dis_loss)
            img_gen_loss.append(gen_loss)
        z_p_img = torch.stack(z_p_img)
        z_q_img = torch.stack(z_q_img)
        img_dis_loss = sum(img_dis_loss)
        img_gen_loss = sum(img_gen_loss)
        
        z_p_seq, z_q_seq, seq_dis_loss, seq_gen_loss = self.seq_model.forward(z_q_img.detach())
        
        log_d_prior = self.discriminator(x, z_p_img, z_p_seq)
        log_d_posterior = self.discriminator(x, z_q_img, z_q_seq)
        
        dis_loss = torch.mean(
            torch.nn.functional.binary_cross_entropy_with_logits(
                log_d_posterior, torch.ones_like(log_d_posterior))
            + torch.nn.functional.binary_cross_entropy_with_logits(
                log_d_prior, torch.zeros_like(log_d_prior)))
        
        x_recon = self.decoder(z_q_seq, self.seq_len)
        recon_likelihood = -torch.nn.functional.binary_cross_entropy(
                                                x_recon, x)*x.data.shape[1]
        
        gen_loss = torch.mean(log_d_posterior)-torch.mean(recon_likelihood)
        
        return img_dis_loss, img_gen_loss, seq_dis_loss, seq_gen_loss, dis_loss, gen_loss

model = VideoAVB(256, 3, 64, 20, 64).cuda()
print(model)

img_dis_params = []
img_gen_params = []
seq_dis_params = []
seq_gen_params = []
dis_params = []
gen_params = []
for name, param in model.named_parameters():
    if 'dis' not in name:
        gen_params.append(param)
    if 'img_model.dis' in name:
        img_dis_params.append(param)
    elif 'seq_model.dis' in name:
        seq_dis_params.append(param)
    elif 'dis' in name:
        dis_params.append(param)
    elif 'img_model' in name:
        img_gen_params.append(param)
    elif 'seq_model' in name:
        seq_gen_params.append(param)
    else:
        assert False  # all params should be covered

img_dis_optimizer = torch.optim.Adam(img_dis_params, lr=1e-3)
img_gen_optimizer = torch.optim.Adam(img_gen_params, lr=1e-3)
seq_dis_optimizer = torch.optim.Adam(seq_dis_params, lr=1e-3)
seq_gen_optimizer = torch.optim.Adam(seq_gen_params, lr=1e-3)
dis_optimizer = torch.optim.Adam(dis_params, lr=1e-3)
gen_optimizer = torch.optim.Adam(gen_params, lr=1e-3)

epochs = 1
global_step = 0
checkpoint_interval = 100

for epoch in range(epochs):
    for i, data in enumerate(dataloader):
        inputs = Variable(data.permute(1, 0, 2, 3, 4), requires_grad=False).cuda()  # (t, b, c, h, w)
        img_dis_loss, img_gen_loss, seq_dis_loss, seq_gen_loss, dis_loss, gen_loss = model.forward(inputs)
        
        img_dis_optimizer.zero_grad()
        img_dis_loss.backward(retain_graph=True)
        img_dis_optimizer.step()
        
        img_gen_optimizer.zero_grad()
        img_gen_loss.backward(retain_graph=True)
        img_gen_optimizer.step()
        
        seq_dis_optimizer.zero_grad()
        seq_dis_loss.backward(retain_graph=True)
        seq_dis_optimizer.step()
        
        seq_gen_optimizer.zero_grad()
        seq_gen_loss.backward(retain_graph=True)
        seq_gen_optimizer.step()
        
        dis_optimizer.zero_grad()
        dis_loss.backward(retain_graph=True)
        dis_optimizer.step()
        
        gen_optimizer.zero_grad()
        gen_loss.backward(retain_graph=True)
        gen_optimizer.step()
        
#         experiment.log_metric("loss", loss.item(), step=global_step)
        print("(Epoch {}) (Global Step {}) (Img Dis Loss {}) (Img Gen Loss {}) (Seq Dis Loss {}) (Seq Gen Loss {}) (Dis Loss {})".format(
            epoch, global_step, img_dis_loss.item(), img_gen_loss.item(), seq_dis_loss.item(), seq_gen_loss.item(), dis_loss.item()), end='\r', flush=True)
#         if i % checkpoint_interval == 0:
#             torch.save(model.state_dict(), "../experiments/obama_avb/checkpoints/model_{}.pth".format(global_step))
#             torch.save(optimizer.state_dict(), "../experiments/obama_avb/checkpoints/optimizer_{}.pth".format(global_step))
        global_step += 1
    print("Epoch {} done!".format(epoch))


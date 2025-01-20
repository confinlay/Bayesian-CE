from __future__ import division
from __future__ import print_function
import torch
from torch.autograd import Variable
import sys
import os
import torch.nn as nn
import matplotlib.pyplot as plt

def mkdir(paths):
    if not isinstance(paths, (list, tuple)):
        paths = [paths]
    for path in paths:
        if not os.path.isdir(path):
            os.makedirs(path, mode=0o777)

import torch.nn as nn

suffixes = ['B', 'KB', 'MB', 'GB', 'TB', 'PB']


def humansize(nbytes):
    i = 0
    while nbytes >= 1024 and i < len(suffixes) - 1:
        nbytes /= 1024.
        i += 1
    f = ('%.2f' % nbytes)
    return '%s%s' % (f, suffixes[i])


def get_num_batches(nb_samples, batch_size, roundup=True):
    if roundup:
        return ((nb_samples + (-nb_samples % batch_size)) / batch_size)  # roundup division
    else:
        return nb_samples / batch_size


def generate_ind_batch(nb_samples, batch_size, random=True, roundup=True):
    if random:
        ind = np.random.permutation(nb_samples)
    else:
        ind = range(int(nb_samples))
    for i in range(int(get_num_batches(nb_samples, batch_size, roundup))):
        yield ind[i * batch_size: (i + 1) * batch_size]


def to_variable(var=(), cuda=True, volatile=False):
    out = []
    for v in var:
        if isinstance(v, np.ndarray):
            v = torch.from_numpy(v).type(torch.FloatTensor)
        if not v.is_cuda and cuda:
            v = v.cuda()
        if not isinstance(v, Variable):
            v = Variable(v, volatile=volatile)
        out.append(v)
    return out


def cprint(color, text, **kwargs):
    if color[0] == '*':
        pre_code = '1;'
        color = color[1:]
    else:
        pre_code = ''
    code = {
        'a': '30',
        'r': '31',
        'g': '32',
        'y': '33',
        'b': '34',
        'p': '35',
        'c': '36',
        'w': '37'
    }
    print("\x1b[%s%sm%s\x1b[0m" % (pre_code, code[color], text), **kwargs)
    sys.stdout.flush()


def shuffle_in_unison_scary(a, b):
    rng_state = np.random.get_state()
    np.random.shuffle(a)
    np.random.set_state(rng_state)
    np.random.shuffle(b)


import torch.utils.data as data
import numpy as np


class Datafeed(data.Dataset):

    def __init__(self, x_train, y_train=None, transform=None):
        self.data = x_train
        self.targets = y_train
        self.transform = transform

    def __getitem__(self, index):
        img = self.data[index]
        if self.transform is not None:
            img = self.transform(img)
        if self.targets is not None:
            return img, self.targets[index]
        else:
            return img

    def __len__(self):
        return len(self.data)


# ----------------------------------------------------------------------------------------------------------------------
class BaseNet(object):
    def __init__(self):
        cprint('c', '\nNet:')

    def get_nb_parameters(self):
        return np.sum(p.numel() for p in self.model.parameters())

    def set_mode_train(self, train=True):
        if train:
            self.model.train()
        else:
            self.model.eval()

    def update_lr(self, epoch, gamma=0.99):
        self.epoch += 1
        if self.schedule is not None:
            if len(self.schedule) == 0 or epoch in self.schedule:
                self.lr *= gamma
                print('learning rate: %f  (%d)\n' % (self.lr, epoch))
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = self.lr

    def save(self, filename):
        cprint('c', 'Writting %s\n' % filename)
        torch.save({
            'epoch': self.epoch,
            'lr': self.lr,
            'model': self.model,
            'optimizer': self.optimizer}, filename)

    def load(self, filename):
        cprint('c', f'Reading {filename}\n')
        
        # Determine best available device
        if torch.cuda.is_available():
            device = torch.device('cuda')
        elif torch.backends.mps.is_available():
            device = torch.device('mps')
        else:
            device = torch.device('cpu')
        print(f"Loading model to device: {device}")
        
        state_dict = torch.load(filename, map_location=device)
        self.epoch = state_dict['epoch']
        self.lr = state_dict['lr']
        self.model = state_dict['model']
        self.optimizer = state_dict['optimizer']
        print(f'  restoring epoch: {self.epoch}, lr: {self.lr}')
        return self.epoch
    
    def new_save(self, filename):
        cprint('c', f'Writing {filename}\n')
        torch.save({
            'epoch': self.epoch,
            'lr': self.lr,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict()
        }, filename)

    def new_load(self, filename, device=None):
        cprint('c', f'Reading {filename}\n')

        # Determine best available device
        if device is None:
            if torch.cuda.is_available():
                device = torch.device('cuda')
            elif torch.backends.mps.is_available():
                device = torch.device('mps')
            else:
                device = torch.device('cpu')
        print(f"Loading model to device: {device}")

        state_dict = torch.load(filename, map_location=device)
        
        self.epoch = state_dict['epoch']
        self.lr = state_dict['lr']
        self.model.load_state_dict(state_dict['model_state_dict'])
        self.optimizer.load_state_dict(state_dict['optimizer_state_dict'])
        
        print(f'  Restored epoch: {self.epoch}, lr: {self.lr}')
        return self.epoch


def fgsm_attack(image, epsilon, data_grad):
    # Collect the element-wise sign of the data gradient
    sign_data_grad = data_grad.sign()
    # Create the perturbed image by adjusting each pixel of the input image
    perturbed_image = image + epsilon*sign_data_grad
    # Adding clipping to maintain [0,1] range
#     perturbed_image = torch.clamp(perturbed_image, 0, 1)
    # Return the perturbed image
    return perturbed_image


def torch_onehot(y, Nclass):
    if y.is_cuda:
        y = y.type(torch.cuda.LongTensor)
    else:
        y = y.type(torch.LongTensor)
    y_onehot = torch.zeros((y.shape[0], Nclass)).type(y.type())
    # In your for loop
    y_onehot.scatter_(1, y.unsqueeze(1), 1)
    return y_onehot


def save_object(obj, filename):
    # Use torch.save to save the object in .pt format
    torch.save(obj, filename)


class StrToBytes:
    def __init__(self, fileobj):
        self.fileobj = fileobj
    def read(self, size):
        return self.fileobj.read(size).encode()
    def readline(self, size=-1):
        return self.fileobj.readline(size).encode()

def load_object(filename, device=None):
    try:
        # Determine the device if not specified
        if device is None:
            if torch.cuda.is_available():
                device = torch.device('cuda')
            elif torch.backends.mps.is_available():
                device = torch.device('mps')
            else:
                device = torch.device('cpu')

        # Load the model directly to the specified device
        model = torch.load(filename, map_location=device)
        return model
    except Exception as e:
        print(f"Error loading file {filename}: {e}")
        raise
            

def array_to_bin_np(array, ncats):
    array = np.array(array)
    bin_vec = np.zeros(ncats)
    bin_vec[array] = 1
    return bin_vec.astype(bool)


def MNIST_mean_std_norm(x):
    mean = 0.1307
    std = 0.3081
    x = x - mean
    x = x / std
    return x

def complete_logit_norm_vec(vec):
    last_term = 1 - vec.sum(dim=1, keepdim=True)
    cvec = torch.cat((vec, last_term), dim=1)
    return cvec


class Ln_distance(nn.Module):
    """If dims is None Compute across all dimensions except first"""
    def __init__(self, n, dim=None):
        super(Ln_distance, self).__init__()
        self.n = n
        self.dim = dim

    def forward(self, x, y):
        d = x - y
        if self.dim is None:
            self.dim = list(range(1, len(d.shape)))
        return torch.abs(d).pow(self.n).sum(dim=self.dim).pow(1./float(self.n))


def smooth_median(X, dim=0):
    """Just gets numpy behaviour instead of torch default
    dim is dimension to be reduced, across which median is taken"""
    yt = X.clone()
    ymax = yt.max(dim=dim, keepdim=True)[0] # maybe this is wrong  and dont need keepdim
    yt_exp = torch.cat((yt, ymax), dim=dim)
    smooth_median = (yt_exp.median(dim=dim)[0] + yt.median(dim=dim)[0]) / 2.
    return smooth_median


class l1_MAD(nn.Module):
    """Intuition behind this metric -> allows variability only where the dataset has variability
    Otherwise it penalises discrepancy heavily. Might not make much sense if data is already normalised to
    unit std. Might also not make sense if we want to detect outlier values in specific features."""
    def __init__(self, trainset_data, median_dim=0, dim=None):
        """Median dim are those across whcih to normalise (not features)
        dim is dimension to sum (features)"""
        super(l1_MAD, self).__init__()
        self.dim = dim
        feature_median = smooth_median(trainset_data, dim=median_dim).unsqueeze(dim=median_dim)
        self.MAD = smooth_median((trainset_data - feature_median).abs(), dim=median_dim).unsqueeze(dim=median_dim)
        self.MAD = self.MAD.clamp(min=1e-4)

    def forward(self, x, y):
        d = x - y
        if self.dim is None:
            self.dim = list(range(1, len(d.shape)))
        return (torch.abs(d) / self.MAD).sum(dim=self.dim)


def evaluate_vae(vae, test_loader, device='mps', num_samples=10):
    """
    Evaluate VAE on multiple metrics:
    1. Reconstruction quality (visual + MSE)
    2. Random samples from prior
    3. VLB on test set
    """
    vae.model = vae.model.to(device)
    vae.model.eval()
    
    # 1. Reconstruction Quality
    x_test, _ = next(iter(test_loader))
    x_test = x_test[:num_samples].to(device)
    
    with torch.no_grad():
        # Get reconstructions
        approx_post = vae.model.encode(x_test)
        z_sample = approx_post.rsample()
        x_rec = torch.sigmoid(vae.model.decode(z_sample))
        
        # Compute MSE
        mse = torch.mean((x_test - x_rec) ** 2).item()
        
        # Plot reconstructions
        plt.figure(figsize=(15, 3))
        for i in range(num_samples):
            # Original
            plt.subplot(2, num_samples, i + 1)
            plt.imshow(x_test[i].cpu().squeeze(), cmap='gray')
            plt.axis('off')
            # Reconstruction
            plt.subplot(2, num_samples, i + num_samples + 1)
            plt.imshow(x_rec[i].cpu().squeeze(), cmap='gray')
            plt.axis('off')
        plt.suptitle(f'Top: Original, Bottom: Reconstruction (MSE: {mse:.4f})')
        plt.show()
        
        # 2. Random samples
        z_rand = torch.randn(num_samples, vae.latent_dim, device=device)
        x_gen = torch.sigmoid(vae.model.decode(z_rand))
        
        plt.figure(figsize=(15, 3))
        for i in range(num_samples):
            plt.subplot(1, num_samples, i + 1)
            plt.imshow(x_gen[i].cpu().squeeze(), cmap='gray')
            plt.axis('off')
        plt.suptitle('Random samples from prior')
        plt.show()
        
        # 3. Compute average VLB on test set
        total_vlb = 0
        n_batches = 0
        for x, _ in test_loader:
            x = x.to(device)
            approx_post = vae.model.encode(x)
            z_sample = approx_post.rsample()
            rec_params = vae.model.decode(z_sample)
            vlb = vae.model.vlb(vae.prior, approx_post, x, rec_params)
            total_vlb += vlb.mean().item()
            n_batches += 1
            
        avg_vlb = total_vlb / n_batches
        print(f'Test set VLB: {avg_vlb:.4f}')
        
        return {'mse': mse, 'vlb': avg_vlb}
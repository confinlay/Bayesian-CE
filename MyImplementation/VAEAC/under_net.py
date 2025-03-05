from __future__ import division
from src.utils import BaseNet, to_variable, cprint
import torch
from torch.distributions.normal import Normal
from VAEAC.fc_gauss import VAE_gauss
import torch.backends.cudnn as cudnn
from src.probability import normal_parse_params
import os

def get_best_device(use_gpu=True):
    """
    Determines best available device with priority:
    1. CUDA (if available and on Linux/Windows)
    2. MPS (if available and on macOS)
    3. CPU (fallback)
    """
    if not use_gpu:
        return torch.device('cpu')
        
    # Check CUDA first (preferred for non-macOS)
    if torch.cuda.is_available():
        return torch.device('cuda')
        
    # Then check MPS (Apple Silicon)
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return torch.device('mps')
        
    # Fallback to CPU
    return torch.device('cpu')

class under_VAEAC(BaseNet):
    def __init__(self, base_VAE, width, depth, latent_dim, lr=1e-3, use_gpu=True):
        super(under_VAEAC, self).__init__()
        cprint('y', 'VAE_gauss_net')

        self.base_VAEAC = base_VAE
        self.pred_sig = False

        # Get the best available device
        self.device = get_best_device(use_gpu)
        self.use_gpu = self.device.type in ['cuda', 'mps']
        
        # Print device info
        print(f"Using device: {self.device}")

        self.input_dim = self.base_VAEAC.latent_dim
        self.width = width
        self.depth = depth
        self.latent_dim = latent_dim
        self.lr = lr

        self.create_net()
        self.create_opt()
        self.epoch = 0
        self.schedule = None

        # Create prior distribution on the appropriate device
        zeros = torch.zeros(latent_dim).to(self.device)
        ones = torch.ones(latent_dim).to(self.device)
        self.prior = Normal(loc=zeros, scale=ones)
        
        self.vlb_scale = 1 / self.input_dim  # scale for dimensions of input so we can use same LR always

        # Add a new attribute for caching latent vectors
        self.z_cache = None

    def create_net(self):
        torch.manual_seed(42)
        if self.device.type == 'cuda':
            torch.cuda.manual_seed(42)
            
        self.model = VAE_gauss(self.input_dim, self.width, self.depth, self.latent_dim, self.pred_sig)
        
        if self.use_gpu:
            self.model = self.model.to(self.device)
            if self.device.type == 'cuda':
                cudnn.benchmark = True
                
        print('    Total params: %.2fM' % (self.get_nb_parameters() / 1000000.0))

    def create_opt(self):
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)

    def fit(self, x):
        """
        Train on batch of data
        """
        self.set_mode_train(True)
        self.optimizer.zero_grad()
        
        # If z_cache is provided, use it directly instead of encoding
        if self.z_cache is not None:
            z_sample = self.z_cache.to(self.device)
            # Clear the cache after using it
            self.z_cache = None
        else:
            # Use the original encoding method if no cache is provided
            z_sample = self.base_VAEAC.recognition_encode(x).sample()
        
        approx_post = self.model.encode(z_sample)
        u_sample = approx_post.rsample()
        
        reconstruction = self.model.decode(u_sample)
        
        log_p_x_given_z = reconstruction.log_prob(z_sample).sum(-1)
        kl = torch.distributions.kl_divergence(approx_post, self.model.prior).sum(-1)
        
        loss = -(log_p_x_given_z - kl).mean()
        
        loss.backward()
        self.optimizer.step()
        
        return -loss.item(), reconstruction.mean

    def eval(self, x):
        """
        Evaluate on batch of data
        """
        self.set_mode_train(False)
        
        # If z_cache is provided, use it directly instead of encoding
        if self.z_cache is not None:
            z_sample = self.z_cache.to(self.device)
            # Clear the cache after using it
            self.z_cache = None
        else:
            # Use the original encoding method if no cache is provided
            z_sample = self.base_VAEAC.recognition_encode(x).sample()
        
        approx_post = self.model.encode(z_sample)
        u_sample = approx_post.sample()
        
        reconstruction = self.model.decode(u_sample)
        
        log_p_x_given_z = reconstruction.log_prob(z_sample).sum(-1)
        kl = torch.distributions.kl_divergence(approx_post, self.model.prior).sum(-1)
        
        loss = -(log_p_x_given_z - kl).mean()
        
        return -loss.item(), reconstruction.mean

    def eval_iw(self, x, k=50):
        self.set_mode_train(train=False)
        
        x = x.to(self.device)
        z_sample = self.base_VAEAC.recognition_encode(x).sample()
        approx_post = self.model.encode(z_sample)

        iw_lb = self.model.iwlb(self.prior, approx_post, z_sample, k)
        return iw_lb.mean().item()

    def recongnition(self, x, grad=False):
        self.set_mode_train(train=False)
        
        x = x.to(self.device)
        if grad and not x.requires_grad:
            x.requires_grad = True
            
        approx_post = self.model.encode(x)
        return approx_post

    def regenerate(self, z, grad=False):
        self.set_mode_train(train=False)
        
        z = z.to(self.device)
        if grad and not z.requires_grad:
            z.requires_grad = True
            
        out = self.model.decode(z)
        return out.data

    def u_recongnition(self, x, grad=False):
        self.set_mode_train(train=False)
        
        x = x.to(self.device)
        if grad and not x.requires_grad:
            x.requires_grad = True

        z = self.base_VAEAC.recognition_encode(x).loc
        approx_post = self.model.encode(z)
        return approx_post

    def u_mask_recongnition(self, x, mask, grad=False):
        self.set_mode_train(train=False)
        
        x = x.to(self.device)
        mask = mask.to(self.device)
        
        if grad and not x.requires_grad:
            x.requires_grad = True

        z = self.base_VAEAC.prior_encode(x, mask).loc
        approx_post = self.model.encode(z)
        return approx_post

    def u_regenerate(self, u, grad=False):
        self.set_mode_train(train=False)
        
        u = u.to(self.device)
        if grad and not u.requires_grad:
            u.requires_grad = True

        z = self.model.decode(u)
        out = self.base_VAEAC.decode(z)
        if self.base_VAEAC.pred_sig:
            return normal_parse_params(out, 1e-2)
        else:
            return out.data
            
    def update_lr(self, epoch):
        """Update learning rate if a schedule is specified"""
        if self.schedule is not None:
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = self.schedule[epoch]
                
    def save(self, filename):
        """Save method with proper device handling and error checking"""
        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(filename), exist_ok=True)
            
            # Move model to CPU for saving
            model_state = {k: v.cpu() for k, v in self.model.state_dict().items()}
            optimizer_state = {k: v.cpu() if isinstance(v, torch.Tensor) else v 
                              for k, v in self.optimizer.state_dict().items()}
            
            state_dict = {
                'model_state': model_state,
                'optimizer_state': optimizer_state,
                'epoch': self.epoch,
                'device_type': self.device.type,
                'width': self.width,
                'depth': self.depth,
                'latent_dim': self.latent_dim,
                'lr': self.lr
            }
            
            torch.save(state_dict, filename)
            print(f"Under_VAEAC model saved successfully to {filename}")
            
        except Exception as e:
            print(f"Error saving under_VAEAC model to {filename}: {str(e)}")
            raise

    def load(self, filename):
        """Load method with proper device handling and error checking"""
        try:
            # Load state dict to CPU first
            state_dict = torch.load(filename, map_location='cpu')
            
            # Load model parameters
            self.model.load_state_dict(state_dict['model_state'])
            self.optimizer.load_state_dict(state_dict['optimizer_state'])
            
            # Load other attributes
            self.epoch = state_dict['epoch']
            self.width = state_dict.get('width', self.width)
            self.depth = state_dict.get('depth', self.depth)
            self.latent_dim = state_dict.get('latent_dim', self.latent_dim)
            self.lr = state_dict.get('lr', self.lr)
            
            # Move model to correct device
            self.model = self.model.to(self.device)
            
            # Move optimizer states to correct device
            for state in self.optimizer.state.values():
                for k, v in state.items():
                    if isinstance(v, torch.Tensor):
                        state[k] = v.to(self.device)
                        
            print(f"Under_VAEAC model loaded successfully from {filename}")
            
        except Exception as e:
            print(f"Error loading under_VAEAC model from {filename}: {str(e)}")
            raise
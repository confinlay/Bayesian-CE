import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
from .. import H_SA_SGHMC

class BayesianLastLayerCat:
    """
    A simplified Bayesian last-layer model for categorical outputs.
    We freeze any pretrained backbone and only sample from the final linear layer's
    posterior via SGHMC.
    """

    def __init__(self, input_dim, output_dim, N_train,
                 lr=1e-2, base_C=0.05, gauss_sig=0.1,
                 device=None, seed=42):
        """
        Args:
            input_dim    : Dimension of the penultimate layer (frozen features).
            output_dim   : Number of classes (categorical outputs).
            N_train      : Size of the training set (for scaling the loss).
            lr, base_C, gauss_sig : Hyperparameters passed to H_SA_SGHMC.
            device       : 'cpu', 'cuda', or 'mps' (if available).
            seed         : Random seed for reproducibility.
        """
        # Device selection
        if device is None:
            if torch.cuda.is_available():
                self.device = torch.device('cuda')
            elif torch.backends.mps.is_available():
                self.device = torch.device('mps')
            else:
                self.device = torch.device('cpu')
        else:
            self.device = torch.device(device)

        torch.manual_seed(seed)
        if self.device.type == 'cuda':
            torch.cuda.manual_seed(seed)

        self.N_train = N_train

        # Define a simple final linear layer
        self.model = nn.Linear(input_dim, output_dim)
        self.model.to(self.device)

        # Create SGHMC optimizer
        self.optimizer = H_SA_SGHMC(self.model.parameters(),
                                    lr=lr, base_C=base_C,
                                    gauss_sig=gauss_sig)

        # We store separate copies of the model to represent posterior samples
        self.ensemble_models = []

        # For gradient clipping (optional)
        self.grad_history = []
        self.max_grad = 1e20
        self.grad_std_mul = 30

    def fit(self, x, y,
            burn_in=False,
            resample_momentum=False,
            resample_prior=False):
        """
        One SGHMC update step on batch (x, y).

        Args:
            x, y               : Tensors on CPU/GPU. y are class labels [0..output_dim-1].
            burn_in            : If True, adapt the preconditioner statistics.
            resample_momentum  : If True, draw new momenta for this SGHMC step.
            resample_prior     : If True, resample the prior precision from Gamma.
        Returns:
            (loss, err) : training loss and classification error.
        """
        self.model.train()
        x = x.to(self.device)
        y = y.long().to(self.device)

        self.optimizer.zero_grad()
        out = self.model(x)
        loss = F.cross_entropy(out, y, reduction='mean')

        # Scale by the total dataset size (like your original code)
        # so the gradient is an unbiased estimator for the entire dataset.
        loss = loss * self.N_train
        loss.backward()

        # Optional: gradient clipping to avoid exploded gradients
        grad_norm = nn.utils.clip_grad_norm_(self.model.parameters(),
                                             max_norm=self.max_grad, norm_type=2)
        self.grad_history.append(float(grad_norm))

        # Update max_grad adaptively if you like
        if len(self.grad_history) > 1000:
            grad_tensor = torch.tensor(self.grad_history)
            mean_g = grad_tensor.mean()
            std_g = grad_tensor.std()
            self.max_grad = float(mean_g + self.grad_std_mul * std_g)
            self.grad_history.pop(0)

        self.optimizer.step(burn_in=burn_in,
                            resample_momentum=resample_momentum,
                            resample_prior=resample_prior)

        # Compute error
        pred = out.argmax(dim=1)
        err = pred.ne(y).sum()

        return loss.item() / self.N_train, err.item()

    def eval(self, x, y):
        """
        Evaluate on a batch (x, y) using the *current* single model parameters.
        Not an ensemble or average. Just the current sample's parameters.
        """
        self.model.eval()
        x = x.to(self.device)
        y = y.long().to(self.device)

        with torch.no_grad():
            out = self.model(x)
            loss = F.cross_entropy(out, y, reduction='sum')
            preds = out.argmax(dim=1)
            err = preds.ne(y).sum()
            probs = F.softmax(out, dim=1).cpu()

        return loss.item(), err.item(), probs

    def predict(self, x):
        """
        Single forward pass with the *current* model parameters.
        Returns probabilities.
        """
        self.model.eval()
        x = x.to(self.device)
        with torch.no_grad():
            logits = self.model(x)
            probs = F.softmax(logits, dim=1)
        return probs.cpu()

    def save_sampled_net(self, max_samples=None):
        """
        Create a deep copy of the model and store it in self.ensemble_models.
        If max_samples is specified, limit the ensemble size by popping old samples.
        """
        if max_samples is not None and len(self.ensemble_models) >= max_samples:
            self.ensemble_models.pop(0)

        # Append a new copy of the current model parameters
        net_copy = copy.deepcopy(self.model)
        net_copy.eval()
        self.ensemble_models.append(net_copy)

        print(f" [save_sampled_net] Ensemble size = {len(self.ensemble_models)}")

    def sample_predict(self, x):
        """
        Return predictions from each model in the ensemble and stack them.
        shape of output: (ensemble_size, batch_size, output_dim).
        """
        x = x.to(self.device)
        outputs = []
        for net in self.ensemble_models:
            net.eval()
            with torch.no_grad():
                out = net(x)
                outputs.append(out)  # [batch_size, output_dim]

        # Stack along dimension 0 -> [ensemble_size, batch_size, output_dim]
        out_stack = torch.stack(outputs, dim=0)

        # Convert logits to probabilities
        prob_stack = F.softmax(out_stack, dim=2).cpu()
        return prob_stack

    def build_ensemble_from_weights(self, weight_list):
        """
        If you have a list of raw parameter dicts (e.g. state_dicts),
        you can build separate model copies from them.
        (Not strictly needed if using save_sampled_net.)
        """
        self.ensemble_models = []
        for sd in weight_list:
            net_copy = copy.deepcopy(self.model)
            net_copy.load_state_dict(sd)
            net_copy.eval()
            self.ensemble_models.append(net_copy)

    def ensemble_size(self):
        return len(self.ensemble_models)
    
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

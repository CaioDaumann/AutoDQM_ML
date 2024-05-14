import numpy as np
import torch
from torch import nn
from torchvision.utils import make_grid 
import torch.autograd as autograd
import random

#from models.ae import AE
#from models.modules import DummyDistribution
#from models.sampling import SampleBufferV2, sample_langevin_v2

def clip_vector_norm(x, max_norm):
    norm = x.norm(dim=-1, keepdim=True)
    x = x * ((norm < max_norm).to(torch.float) + (norm > max_norm).to(torch.float) * max_norm/norm + 1e-6)
    return x

class SampleBufferV2:
    def __init__(self, max_samples=10000, replay_ratio=0.95):
        self.max_samples = max_samples
        self.buffer = []
        self.replay_ratio = replay_ratio

    def __len__(self):
        return len(self.buffer)

    def push(self, samples):
        samples = samples.detach().to('cpu')

        for sample in samples:
            self.buffer.append(sample)

            if len(self.buffer) > self.max_samples:
                self.buffer.pop(0)

    def get(self, n_samples):
        samples = random.choices(self.buffer, k=n_samples)
        samples = torch.stack(samples, 0)
        return samples

def sample_langevin_v2(x, model, stepsize, n_steps, noise_scale=None, intermediate_samples=False,
                    clip_x=None, clip_grad=None, reject_boundary=False, noise_anneal=None, noise_anneal_full=None,
                    spherical=False, mh=False, temperature=None, norm=False):
    """Langevin Monte Carlo
    x: torch.Tensor, initial points
    model: An energy-based model. returns energy
    stepsize: float
    n_steps: integer
    noise_scale: Optional. float. If None, set to np.sqrt(stepsize * 2)
    clip_x : tuple (start, end) or None boundary of square domain
    reject_boundary: Reject out-of-domain samples if True. otherwise clip.
    """
    assert not ((stepsize is None) and (noise_scale is None)), 'stepsize and noise_scale cannot be None at the same time'
    if noise_scale is None:
        noise_scale = np.sqrt(stepsize * 2)
    if stepsize is None:
        stepsize = (noise_scale ** 2) / 2
    noise_scale_ = noise_scale
    stepsize_ = stepsize
    if temperature is None:
        temperature = 1.

    # initial data
    x.requires_grad = True
    E_x = model(x)
    grad_E_x = autograd.grad(E_x.sum(), x, only_inputs=True)[0]
    if clip_grad is not None:
        grad_E_x = clip_vector_norm(grad_E_x, max_norm=clip_grad)
    E_y = E_x; grad_E_y = grad_E_x;

    l_samples = [x.detach().to('cpu')]
    l_dynamics = []; l_drift = []; l_diffusion = []; l_accept = []
    for i_step in range(n_steps):
        noise = torch.randn_like(x) * noise_scale_
        dynamics = - stepsize_ * grad_E_x / temperature + noise
        y = x + dynamics
        reject = torch.zeros(len(y), dtype=torch.bool)

        if clip_x is not None:
            if reject_boundary:
                accept = ((y >= clip_x[0]) & (y <= clip_x[1])).view(len(x), -1).all(dim=1)
                reject = ~ accept
                y[reject] = x[reject]
            else:
                y = torch.clamp(y, clip_x[0], clip_x[1])
        
        if norm:
            y = y/y.sum(dim=(2,3)).view(-1,1,1,1)
            
        if spherical:
            y = y / y.norm(dim=1, p=2, keepdim=True)

        # y_accept = y[~reject]
        # E_y[~reject] = model(y_accept)
        # grad_E_y[~reject] = autograd.grad(E_y.sum(), y_accept, only_inputs=True)[0]
        E_y = model(y)
        grad_E_y = autograd.grad(E_y.sum(), y, only_inputs=True)[0]
 
        if clip_grad is not None:
            grad_E_y = clip_vector_norm(grad_E_y, max_norm=clip_grad)

        if mh:
            y_to_x = ((grad_E_x + grad_E_y) * stepsize_ - noise).view(len(x), -1).norm(p=2, dim=1, keepdim=True) ** 2
            x_to_y = (noise).view(len(x), -1).norm(dim=1, keepdim=True, p=2) ** 2
            transition = - (y_to_x - x_to_y) / 4 / stepsize_  # B x 1
            prob = -E_y + E_x
            accept_prob = torch.exp((transition + prob) / temperature)[:,0]  # B
            reject = (torch.rand_like(accept_prob) > accept_prob) # | reject
            y[reject] = x[reject]
            E_y[reject] = E_x[reject]
            grad_E_y[reject] = grad_E_x[reject]
            x = y; E_x = E_y; grad_E_x = grad_E_y
            l_accept.append(~reject)

        x = y; E_x = E_y; grad_E_x = grad_E_y

        if noise_anneal is not None:
            noise_scale_ = noise_scale / (1 + i_step)

        elif noise_anneal_full is not None:
            noise_scale_ = noise_scale / np.sqrt(1 + i_step)
            stepsize_ = stepsize / (1 + i_step)

        l_dynamics.append(dynamics.detach().cpu())
        l_drift.append((- stepsize * grad_E_x).detach().cpu())
        l_diffusion.append(noise.detach().cpu())
        l_samples.append(x.detach().cpu())
    
    return {'sample': x.detach(), 'l_samples': l_samples, 'l_dynamics': l_dynamics,
            'l_drift': l_drift, 'l_diffusion': l_diffusion, 'l_accept': l_accept}


class DummyDistribution(nn.Module):
    """ Function-less class introduced for backward-compatibility of model checkpoint files. """
    def __init__(self, net):
        super().__init__()
        self.net = net
        self.register_buffer('sigma', torch.tensor(0., dtype=torch.float))

    def forward(self, x):
        return self.net(x)

class AE(nn.Module):
    """autoencoder"""
    def __init__(self, encoder, decoder):
        """
        encoder, decoder : neural networks
        """
        super(AE, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.own_optimizer = False

    def forward(self, x):
        z = self.encode(x)
        recon = self.decoder(z)
        return recon

    def encode(self, x):
        z = self.encoder(x)
        return z

    def predict(self, x):
        """one-class anomaly prediction"""
        recon = self(x)
        if hasattr(self.decoder, 'error'):
            predict = self.decoder.error(x, recon)
        else:
            predict = ((recon - x) ** 2).view(len(x), -1).mean(dim=1)
        return predict

    def predict_and_reconstruct(self, x):
        recon = self(x)
        if hasattr(self.decoder, 'error'):
            recon_err = self.decoder.error(x, recon)
        else:
            recon_err = ((recon - x) ** 2).view(len(x), -1).mean(dim=1)
        return recon_err, recon

    def validation_step(self, x, **kwargs):
        recon = self(x)
        if hasattr(self.decoder, 'error'):
            predict = self.decoder.error(x, recon)
        else:
            predict = ((recon - x) ** 2).view(len(x), -1).mean(dim=1)
        loss = predict.mean()

        if kwargs.get('show_image', True):
            x_img = make_grid(x.detach().cpu(), nrow=10, range=(0, 1))
            recon_img = make_grid(recon.detach().cpu(), nrow=10, range=(0, 1))
        else:
            x_img, recon_img = None, None
        return {'loss': loss.item(), 'predict': predict, 'reconstruction': recon,
                'input@': x_img, 'recon@': recon_img}

    def train_step(self, x, optimizer, clip_grad=None, **kwargs):
        optimizer.zero_grad()
        recon_error = self.predict(x)
        loss = recon_error.mean()
        loss.backward()
        if clip_grad is not None:
            torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=clip_grad)
        optimizer.step()
        return {'loss': loss.item()}

    def reconstruct(self, x):
        return self(x)

    def sample(self, N, z_shape=None, device='cpu'):
        if z_shape is None:
            z_shape = self.encoder.out_shape

        rand_z = torch.rand(N, *z_shape).to(device) * 2 - 1
        sample_x = self.decoder(rand_z)
        return sample_x



def clip_vector_norm(x, max_norm):
    norm = x.norm(dim=-1, keepdim=True)
    x = x * ((norm < max_norm).to(torch.float) + (norm > max_norm).to(torch.float) * max_norm/norm + 1e-6)
    return x

class FFEBM(nn.Module):
    """feed-forward energy-based model"""
    def __init__(self, net, x_step=None, x_stepsize=None, x_noise_std=None, x_noise_anneal=None,
                 x_noise_anneal_full=None, x_bound=None, x_clip_langevin_grad=None, l2_norm_reg=None,
                 buffer_size=10000, replay_ratio=0.95, replay=True, gamma=1, sampling='x',
                 initial_dist='gaussian', temperature=1., temperature_trainable=False,
                 mh=False, reject_boundary=False, x_norm=False):
        super().__init__()
        self.net = net

        self.x_bound = x_bound
        self.l2_norm_reg = l2_norm_reg
        self.gamma = gamma
        self.sampling = sampling

        self.x_step = x_step
        self.x_stepsize = x_stepsize
        self.x_noise_std = x_noise_std
        self.x_noise_anneal = x_noise_anneal
        self.x_noise_anneal_full = x_noise_anneal_full
        self.x_bound = x_bound
        self.x_clip_langevin_grad = x_clip_langevin_grad
        self.mh = mh
        self.reject_boundary = reject_boundary
        self.x_norm = x_norm

        self.buffer_size = buffer_size
        self.replay_ratio = replay_ratio
        self.replay = replay

        self.buffer = SampleBufferV2(max_samples=buffer_size, replay_ratio=replay_ratio)

        self.x_shape = None
        self.initial_dist = initial_dist
        temperature = np.log(temperature)
        self.temperature_trainable = temperature_trainable
        if temperature_trainable:
            self.register_parameter('temperature_', nn.Parameter(torch.tensor(temperature, dtype=torch.float)))
        else:
            self.register_buffer('temperature_', torch.tensor(temperature, dtype=torch.float))

    @property
    def temperature(self):
        return torch.exp(self.temperature_)

    @property
    def sample_shape(self):
        return self.x_shape

    def forward(self, x):
        return self.net(x)

    def predict(self, x):
        return self.forward(x)

    def energy(self, x):
        return self.forward(x)

    def energy_T(self,x):
        return self.energy(x) / self.temperature

    def sample(self, x0=None, n_sample=None, device=None, replay=None):
        """sampling factory function. takes either x0 or n_sample and device
        """
        if x0 is not None:
            n_sample = len(x0)
            device = x0.device
        if replay is None:
            replay = self.replay

        if self.sampling == 'x':
            return self.sample_x(n_sample, device, replay=replay)
        elif self.sampling == 'cd':
            return self.sample_x(n_sample, device, x0=x0, replay=False)
        elif self.sampling == 'on_manifold':
            return self.sample_omi(n_sample, device, replay=replay)

    def sample_x(self, n_sample=None, device=None, x0=None, replay=False):
        if x0 is None:
            x0 = self.initial_sample(n_sample, device=device)
        d_sample_result = sample_langevin_v2(x0.detach(), self.energy, stepsize=self.x_stepsize, n_steps=self.x_step,
                                        noise_scale=self.x_noise_std,
                                        clip_x=self.x_bound, noise_anneal=self.x_noise_anneal,
                                        noise_anneal_full=self.x_noise_anneal_full,
                                        clip_grad=self.x_clip_langevin_grad, spherical=False,
                                        mh=self.mh, temperature=self.temperature, reject_boundary=self.reject_boundary, norm=self.x_norm)
        sample_result = d_sample_result['sample']
        if replay:
            self.buffer.push(sample_result)
        d_sample_result['sample_x'] = sample_result
        d_sample_result['sample_x0'] = x0
        return d_sample_result

    def initial_sample(self, n_samples, device):
        l_sample = []
        if not self.replay or len(self.buffer) == 0:
            n_replay = 0
        else:
            n_replay = (np.random.rand(n_samples) < self.replay_ratio).sum()
            l_sample.append(self.buffer.get(n_replay))

        shape = (n_samples - n_replay,) + self.sample_shape
        if self.initial_dist == 'gaussian':
            x0_new = torch.randn(shape, dtype=torch.float)
        elif self.initial_dist == 'uniform':
            x0_new = torch.rand(shape, dtype=torch.float)
            if self.sampling != 'on_manifold' and self.x_bound is not None:
                x0_new = x0_new * (self.x_bound[1] - self.x_bound[0]) + self.x_bound[0]
            elif self.sampling == 'on_manifold' and self.z_bound is not None:
                x0_new = x0_new * (self.z_bound[1] - self.z_bound[0]) + self.z_bound[0]

        l_sample.append(x0_new)
        return torch.cat(l_sample).to(device)

    def _set_x_shape(self, x):
        if self.x_shape is not None:
            return
        self.x_shape = x.shape[1:]

    def weight_norm(self, net):
        norm = 0
        for param in net.parameters():
            norm += (param ** 2).sum()
        return norm

    def train_step(self, x, opt):
        self._set_x_shape(x)

        # negative sample
        d_sample = self.sample(x)
        x_neg = d_sample['sample_x']

        opt.zero_grad()
        neg_e = self.energy(x_neg)

        # ae recon pass
        pos_e = self.energy(x)

        loss = (pos_e.mean() - neg_e.mean()) / self.temperature

        if self.gamma is not None:
            loss += self.gamma * (pos_e ** 2 + neg_e ** 2).mean()

        # weight regularization
        l2_norm = self.weight_norm(self.net)
        if self.l2_norm_reg is not None:
            loss = loss + self.l2_norm_reg * l2_norm

        loss.backward()
        opt.step()

        d_result = {'pos_e': pos_e.mean().item(), 'neg_e': neg_e.mean().item(),
                    'x_neg': x_neg.detach().cpu(), 'x_neg_0': d_sample['sample_x0'].detach().cpu(),
                    'loss': loss.item(), 'sample': x_neg.detach().cpu(),
                    'l2_norm': l2_norm.item()}
        return d_result

    def validation_step(self, x, y=None):
        pos_e = self.energy(x)
        loss = pos_e.mean().item()
        predict = pos_e.detach().cpu().flatten()
        return {'loss': pos_e, 'predict':predict}



class NAE(FFEBM):
    """Normalized Autoencoder"""
    # Need to check these MCMC parameters like stepsize, noise, etc ...
    def __init__(self, 
                 z_step=50, z_stepsize=0.2, z_noise_std=0.2, z_noise_anneal=None, z_noise_anneal_full=None,
                 x_step=50, x_stepsize=10, x_noise_std=0.05, x_noise_anneal=None, x_noise_anneal_full=None,
                 x_bound=(0, 1), z_bound=None,
                 z_clip_langevin_grad=None, x_clip_langevin_grad=None, 
                 l2_norm_reg=1.0e-8, l2_norm_reg_en=1.0e-8, spherical=True, z_norm_reg=None,
                 buffer_size=10000, replay_ratio=0.95, replay=True,
                 gamma= 0.01, sampling='on_manifold',
                 temperature=1., temperature_trainable=False,
                 initial_dist='gaussian', 
                 mh=True, mh_z=True, reject_boundary=False, reject_boundary_z=False, 
                 x_norm=False, z_norm=False):
        # Also added the gamma, which should be the negative energy regularization!!!
        """
        # Questions: Why is the lr for the NAE so low? 
        - appraently the low lr works =p. Now it looks like it does not diverge anymore
        """
        # Changes, mh and mhz are actiavted now. Temp trainable is false. l2 norm from zero to 1e-8
        # Also added PRelu as activation instead of ReLU
        # What does the initial distribution is?
        """
        encoder: An encoder network, an instance of nn.Module.
        decoder: A decoder network, an instance of nn.Module.

        **Sampling Parameters**
        sampling: Sampling methods.
                  'on_manifold' - on-manifold initialization.
                  'cd' - Contrastive Divergence.
                  'x' - Persistent CD.

        z_step: The number of steps in latent chain.
        z_stepsize: The step size of latent chain
        z_noise_std: The standard deviation of noise in latent chain
        z_noise_anneal: Noise annealing parameter in latent chain. If None, no annealing.
        mh_z: If True, use Metropolis-Hastings rejection in latent chain.
        z_clip_langevin_grad: Clip the norm of gradient in latent chain.
        z_bound: [z_min, z_max]

        x_step: The number of steps in visible chain.
        x_stepsize: The step size of visible chain
        x_noise_std: The standard deviation of noise in visible chain
        x_noise_anneal: Noise annealing parameter in visible chain. If None, no annealing.
        mh: If True, use Metropolis-Hastings rejection in latent chain.
        x_clip_langevin_grad: Clip the norm of gradient in visible chain.
        x_bouond: [x_min, x_bound]. 

        replay: Whether to use the replay buffer.
        buffer_size: The size of replay buffer.
        replay_ratio: The probability of applying persistent CD. A chain is re-initialized with the probability of
                      (1 - replay_ratio).
        initial_dist: The distribution from which initial samples are generated.
                      'Gaussian' or 'uniform'



        **Regularization Parameters**
        gamma: The coefficient for regularizing the negative sample energy.
        l2_norm_reg: The coefficient for L2 norm of decoder weights.
        l2_norm_reg_en: The coefficient for L2 norm of encoder weights.
        z_norm_reg: The coefficient for regularizing the L2 norm of Z vector.


        """
        super(NAE, self).__init__(net=None, x_step=x_step, x_stepsize=x_stepsize, x_noise_std=x_noise_std,
                                  x_noise_anneal=x_noise_anneal, x_noise_anneal_full=x_noise_anneal_full,
                                  x_bound=x_bound,
                                  x_clip_langevin_grad=x_clip_langevin_grad, l2_norm_reg=l2_norm_reg,
                                  buffer_size=buffer_size, replay_ratio=replay_ratio, replay=replay,
                                  gamma=gamma, sampling=sampling, initial_dist=initial_dist,
                                  temperature=temperature, temperature_trainable=temperature_trainable,
                                  mh=mh, reject_boundary=reject_boundary, x_norm=x_norm)
        # Defining a simple encoder and decoder!!
        
        self.encoder = nn.Sequential(
            nn.Linear(200, 128),
            nn.PReLU(),
            nn.Linear(128, 64),
            nn.PReLU(),
            nn.Linear(64, 16),
        )
        
        decoder = nn.Sequential(
            nn.Linear(16, 64),
            nn.PReLU(),
            nn.Linear(64, 128),
            nn.PReLU(),
            nn.Linear(128, 200),
        )
        
        # What does this DummyDistribution do??
        self.decoder = DummyDistribution(decoder)
        
        #self.encoder = encoder
        #self.decoder = DummyDistribution(decoder)
        
        self.z_step = z_step
        self.z_stepsize = z_stepsize
        self.z_noise_std = z_noise_std
        self.z_noise_anneal = z_noise_anneal
        self.z_noise_anneal_full = z_noise_anneal_full
        self.z_clip_langevin_grad = z_clip_langevin_grad
        self.mh_z = mh_z
        self.reject_boundary_z = reject_boundary_z
        self.z_norm = z_norm

        self.z_bound = z_bound
        self.l2_norm_reg = l2_norm_reg  # decoder
        self.l2_norm_reg_en = l2_norm_reg_en
        self.spherical = spherical
        self.sampling = sampling
        self.z_norm_reg = z_norm_reg

        self.z_shape = None
        self.x_shape = None

    @property
    def sample_shape(self):
        if self.sampling == 'on_manifold':
            return self.z_shape
        else:
            return self.x_shape

    def error(self, x, recon):
        """L2 error"""
        return ((x - recon) ** 2).view((x.shape[0], -1)).sum(dim=1)

    def forward(self, x):
        """ Computes error per dimension """
        D = np.prod(x.shape[1:])
        z = self.encode(x)
        recon = self.decoder(z)
        return self.error(x, recon) / D

    def energy_with_z(self, x):
        D = np.prod(x.shape[1:])
        z = self.encode(x)
        recon = self.decoder(z)
        return self.error(x, recon) / D, z

    def normalize(self, z):
        """normalize to unit length"""
        if self.spherical:
            if len(z.shape) == 4:
                z = z / z.view(len(z), -1).norm(dim=-1)[:, None, None, None]
            else:
                z = z / z.view(len(z), -1).norm(dim=1, keepdim=True)
            return z
        else:
            return z

    def encode(self, x):
        if self.spherical:
            return self.normalize(self.encoder(x))
        else:
            return self.encoder(x)

    def sample_omi(self, n_sample, device, replay=False):
        """using on-manifold initialization"""
        # Step 1: On-manifold initialization: LMC on Z space 
        z0 = self.initial_sample(n_sample, device)
        if self.spherical:
            z0 = self.normalize(z0)
        d_sample_z = self.sample_z(z0=z0, replay=replay)
        sample_z = d_sample_z['sample']

        sample_x_1 = self.decoder(sample_z).detach()
        if self.x_bound is not None:
            sample_x_1.clamp_(self.x_bound[0], self.x_bound[1])

        # Step 2: LMC on X space
        d_sample_x = self.sample_x(x0=sample_x_1, replay=False)
        sample_x_2 = d_sample_x['sample_x']
        return {'sample_x': sample_x_2, 'sample_z': sample_z.detach(), 'sample_x0': sample_x_1, 'sample_z0': z0.detach()} 

    def sample_z(self, n_sample=None, device=None, replay=False, z0=None):
        if z0 is None:
            z0 = self.initial_sample(n_sample, device)
        energy = lambda z: self.energy(self.decoder(z))
        d_sample_result = sample_langevin_v2(z0, energy, stepsize=self.z_stepsize, n_steps=self.z_step,
                                             noise_scale=self.z_noise_std, noise_anneal=self.z_noise_anneal,
                                             noise_anneal_full=self.z_noise_anneal_full,
                                             clip_x=self.z_bound, clip_grad=self.z_clip_langevin_grad,
                                             spherical=self.spherical, mh=self.mh_z,
                                             temperature=self.temperature, reject_boundary=self.reject_boundary_z, norm=self.z_norm)
        sample_z = d_sample_result['sample']
        if replay:
            self.buffer.push(sample_z)
        return d_sample_result 

    def _set_z_shape(self, x):
        if self.z_shape is not None:
            return
        # infer z_shape by computing forward
        with torch.no_grad():
            dummy_z = self.encode(x[[0]])
        z_shape = dummy_z.shape
        self.z_shape = z_shape[1:]

    # Shouldnt this use the sphere normalization in the laten space??
    def reconstruct(self, x):
        z = self.encode(x)
        return self.decoder(z)
    
    # This should be used to train the first step of the AE??
    # Why it is not sphere normalized??
    def train_step_ae(self, x, opt, clip_grad=None, validation=False):
        opt.zero_grad()
        z = self.encode(x)
        recon = self.decoder(z)
        z_norm = (z ** 2).mean()
        x_dim = np.prod(x.shape[1:])
        recon_error = self.error(x, recon).mean() / x_dim
        loss = recon_error

        # weight regularization
        decoder_norm = self.weight_norm(self.decoder)
        encoder_norm = self.weight_norm(self.encoder)
        if self.l2_norm_reg is not None:
            loss = loss + self.l2_norm_reg * decoder_norm
        if self.l2_norm_reg_en is not None:
            loss = loss + self.l2_norm_reg_en * encoder_norm
        
        if validation:
            opt.zero_grad()
            return loss.item()    
            
        loss.backward()

        if clip_grad is not None:
            torch.nn.utils.clip_grad_norm_(opt.param_groups[0]['params'], max_norm=clip_grad)
        opt.step()
        d_result = {'loss': loss.item(), 'z_norm': z_norm.item(), 'recon_error_': recon_error.item(),
                    'decoder_norm_': decoder_norm.item(), 'encoder_norm_': encoder_norm.item()}
        return d_result

    def train_step(self, x, opt, validation = False):
        self._set_z_shape(x)
        self._set_x_shape(x)

        # negative sample
        d_sample = self.sample(x)
        x_neg    = d_sample['sample_x']

        opt.zero_grad()
        neg_e, neg_z = self.energy_with_z(x_neg)

        # ae recon pass
        pos_e, pos_z = self.energy_with_z(x)

        loss = (pos_e.mean() - neg_e.mean()) 

        if self.temperature_trainable:
            loss = loss + (pos_e.mean() - neg_e.mean()).detach() / self.temperature


        # regularizing negative sample energy
        if self.gamma is not None:
            gamma_term = ((neg_e) ** 2).mean()
            loss += self.gamma * gamma_term

        # weight regularization
        decoder_norm = self.weight_norm(self.decoder)
        encoder_norm = self.weight_norm(self.encoder)
        if self.l2_norm_reg is not None:
            loss = loss + self.l2_norm_reg * decoder_norm
        if self.l2_norm_reg_en is not None:
            loss = loss + self.l2_norm_reg_en * encoder_norm
        if self.z_norm_reg is not None:
            z_norm = (torch.cat([pos_z, neg_z]) ** 2).mean()
            loss = loss + self.z_norm_reg * z_norm

        if validation:
            opt.zero_grad()
            return loss.item()

        loss.backward()
        opt.step()

        # for debugging
        x_neg_0 = d_sample['sample_x0']
        neg_e_x0 = self.energy(x_neg_0)  # energy of samples from latent chain
        recon_neg = self.reconstruct(x_neg)
        d_result = {'pos_e': pos_e.mean().item(), 'neg_e': neg_e.mean().item(),
                    'x_neg': x_neg.detach().cpu(), 'recon_neg': recon_neg.detach().cpu(),
                    'loss': loss.item(), 'sample': x_neg.detach().cpu(),
                    'decoder_norm': decoder_norm.item(), 'encoder_norm': encoder_norm.item(),
                    'neg_e_x0': neg_e_x0.mean().item(), 'x_neg_0': x_neg_0.detach().cpu(),
                    'temperature': self.temperature.item(), 
                    'pos_z': pos_z.detach().cpu(), 'neg_z': neg_z.detach().cpu()}
        if self.gamma is not None:
            d_result['gamma_term'] = gamma_term.item()
        if 'sample_z0' in d_sample:
            x_neg_z0 = self.decoder(d_sample['sample_z0'])
            d_result['neg_e_z0'] = self.energy(x_neg_z0).mean().item()
        return d_result

    def validation_step(self, x, y=None):
        z = self.encode(x)
        recon = self.decoder(z)
        energy = self.error(x, recon)
        loss = energy.mean().item()
        
        # lets keep this off for now, but need to reactivate it later
        #recon_img = make_grid(recon.detach().cpu(), nrow=10, range=(0, 1))
        #input_img = make_grid(x.detach().cpu(), nrow=10, range=(0, 1))
        
        return {'loss': loss, 'pos_e': loss} #, 'recon@': recon_img, 'input@': input_img}
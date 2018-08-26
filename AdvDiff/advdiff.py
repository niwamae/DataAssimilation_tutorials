import numpy as np


class AdvDiffModel:
    
    def __init__(self, **params):
        self.dx = params['dx']
        self.nx = params['nx']
        self.dt = params['dt']
        self.u = params['u']
        self.nu = params['nu']
        self.dxi = 1. / self.dx
        self.freq_euler = params['freq_euler']
        self.x = np.arange(self.nx) * self.dx


class Model(AdvDiffModel):

    def __init__(self, **params):
        super().__init__(**params)
        self.it = 0
        self.data = np.zeros(self.nx, dtype=np.float64)
        self.data_past = np.zeros(self.nx, dtype=np.float64)

    def reset(self):
        self.it = 0
        self.data = np.zeros(self.nx, dtype=np.float64)
        self.data_past = np.zeros(self.nx, dtype=np.float64)

    def advection(self, v):
        _v = np.r_[v[-1], v, v[0]]
        return -self.u * (_v[2:] - _v[:-2]) * 0.5 * self.dxi

    def diffusion(self, v):
        _v = np.r_[v[-1], v, v[0]]
        return self.nu * (_v[:-2] - 2. * _v[1:-1] + _v[2:]) * self.dxi * self.dxi

    def step_forward(self, force=0.):
        if self.it % self.freq_euler == 0:
            v0 = self.data.copy()
            alpha = 1.0
        else:
            v0 = self.data_past.copy()
            alpha = 2.0
        v1 = self.data.copy()
        v2 = v0 + (self.advection(v1) + self.diffusion(v0) + force) * alpha * self.dt
        self.data_past = self.data
        self.data = v2
        self.it += 1


class EnsembleModel(AdvDiffModel):

    def __init__(self, n_ensemble, **params):
        super().__init__(**params)
        self.it = 0
        self.n_ensemble = n_ensemble
        self.data = np.zeros((self.n_ensemble, self.nx), dtype=np.float64)
        self.data_past = np.zeros((self.n_ensemble, self.nx), dtype=np.float64)

    def advection(self, v):
        _v = np.c_[v[:,-1], v, v[:,0]]
        return -self.u * (_v[:,2:] - _v[:,:-2]) * 0.5 * self.dxi

    def diffusion(self, v):
        _v = np.c_[v[:,-1], v, v[:,0]]
        return self.nu * (_v[:,:-2] - 2.*_v[:,1:-1] + _v[:,2:]) * self.dxi * self.dxi
        
    def step_forward(self, force=0.):
        if self.it % self.freq_euler == 0:
            v0 = self.data.copy()
            alpha = 1.0
        else:
            v0 = self.data_past.copy()
            alpha = 2.0
        v1 = self.data.copy()
        v2 = v0 + (self.advection(v1) + self.diffusion(v0) + force) * alpha * self.dt
        self.data_past = self.data
        self.data = v2
        self.it += 1

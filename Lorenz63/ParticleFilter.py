import numpy as np

class ParticleFilter:

    def __init__(self, ensemble, H, obs, noise):
        self.ensemble = ensemble
        self.n_ensemble = ensemble.shape[0]
        self.H = H
        self.obs = obs
        self.noise = noise

    def fit(self):
        # calculate weight
        sigma = self.noise
        d = (self.H @ self.obs.T) - (self.H @ self.ensemble.T)
        likelihood = np.exp(-0.5*d*d / sigma**2) / np.sqrt(2.*np.pi*sigma**2)
        log_likelihood = np.log(likelihood)
        log_likelihood_max = log_likelihood.max()
        w = np.exp(log_likelihood - log_likelihood_max)
        w = w / w.sum()
        # resampling
        zeta = np.zeros(self.n_ensemble + 1)
        zeta[1:] = w.cumsum()
        eta = (np.arange(self.n_ensemble) + 0.8) / self.n_ensemble
        indices = np.zeros_like(eta, dtype=np.int64)
        for i, e in enumerate(eta):
            indices[i] = np.where(e - zeta > 0.)[0].max()
        return self.ensemble[indices, :]


class MergingParticleFilter:
    
    def __init__(self, ensemble, H, obs, noise):
        self.ensemble = ensemble
        self.n_ensemble = ensemble.shape[0]
        self.H = H
        self.obs = obs
        self.noise = noise

    def fit(self):
        # calculate weight
        sigma = self.noise
        d = (self.H @ self.obs.T) - (self.H @ self.ensemble.T)
        likelihood = np.exp(-0.5*d*d / sigma**2) / np.sqrt(2.*np.pi*sigma**2)
        log_likelihood = np.log(likelihood)
        log_likelihood_max = log_likelihood.max()
        w = np.exp(log_likelihood - log_likelihood_max)
        w = w / w.sum()
        # resampling
        zeta = np.zeros(self.n_ensemble + 1)
        zeta[1:] = w.cumsum()
        nens = self.n_ensemble * 3
        eta = (np.arange(nens) + 0.8) / nens
        indices = np.zeros_like(eta, dtype=np.int64)
        for i, e in enumerate(eta):
            indices[i] = np.where(e - zeta > 0.)[0].max()
        indices = np.random.permutation(indices)
        state = self.ensemble[indices,:].reshape((3, self.n_ensemble, 3))
        alpha = np.zeros_like(state)
        alpha[0,:,:] = 3./4.
        alpha[1,:,:] = (1. + np.sqrt(13.)) / 8.
        alpha[2,:,:] = (1. - np.sqrt(13.)) / 8.
        return (alpha * state).sum(axis=0)

import abc

import torch
import torch.nn as nn
from torch.nn.functional import conv2d


class Whitening2d(nn.Module):
    def __init__(self, num_features, momentum=0.01, track_running_stats=True, eps=0, dim=0):
        super(Whitening2d, self).__init__()
        self.num_features = num_features
        self.momentum = momentum
        self.track_running_stats = track_running_stats
        self.eps = eps
        self.dim = dim

        if self.track_running_stats and self.dim==0:
            self.register_buffer(
                "running_mean", torch.zeros([1, self.num_features, 1, 1])
            )
            self.register_buffer("running_variance", torch.eye(self.num_features))

    def forward(self, x):
        sigma_dim = x.size(0) if self.dim == 1 else self.num_features
        m = x.mean(self.dim)
        m = m.view(sigma_dim, -1) if self.dim == 1 else m.view(-1, sigma_dim)
        if not self.training and self.track_running_stats and self.dim==0:  # for inference
            m = self.running_mean
        xn = x - m  # [128, 64]

        T = xn if self.dim == 1 else xn.permute(1, 0)  # [128, 64] / [64, 128]
        f_cov = torch.mm(T, T.permute(1, 0)) / (T.shape[-1] - 1)  # [128, 128] / [64, 64]

        eye = torch.eye(sigma_dim).type(f_cov.type())  # [128, 128] / [64, 64]

        if not self.training and self.track_running_stats:  # for inference
            f_cov = self.running_variance

        sigma = (1 - self.eps) * f_cov + self.eps * eye

        wm = self.whiten_matrix(sigma, eye, sigma_dim)  # [128,64,1,1] [64,64,1,1] -> [128,64,1,1]
        wm = wm.reshape(sigma_dim, sigma_dim, 1, 1)

        if self.dim == 1:
            xn = xn.permute(1, 0).reshape(-1, sigma_dim, 1, 1)
            decorrelated = conv2d(xn, wm)  # [64,128,1,1] [128,128,1,1] -> [64,128,1,1]
            decorrelated = decorrelated.permute(1, 0, 2, 3)  # -> [128,64,1,1]
        else:
            xn = xn.reshape(-1, sigma_dim, 1, 1)
            decorrelated = conv2d(xn, wm)  # [128,64,1,1] [64,64,1,1] -> [128,64,1,1]

        if self.training and self.track_running_stats and self.dim==0:
            self.running_mean = torch.add(
                self.momentum * m.detach(),
                (1 - self.momentum) * self.running_mean,
                out=self.running_mean,
            )
            self.running_variance = torch.add(
                self.momentum * f_cov.detach(),
                (1 - self.momentum) * self.running_variance,
                out=self.running_variance,
            )

        return decorrelated.squeeze(2).squeeze(2)

    @abc.abstractmethod
    def whiten_matrix(self, sigma, eye, dim):
        pass

    def extra_repr(self):
        return "features={}, eps={}, momentum={}, dim={}".format(
            self.num_features, self.eps, self.momentum, self.dim
        )


class Whitening2dCholesky(Whitening2d):
    def whiten_matrix(self, sigma, eye, dim):  # x [128,64,1,1]
        wm = torch.triangular_solve(
            eye, torch.cholesky(sigma), upper=False
        )[0]
        return wm


class Whitening2dZCA(Whitening2d):
    def whiten_matrix(self, sigma, eye, dim):
        u, eig, _ = sigma.svd()
        scale = eig.rsqrt()
        wm = u.matmul(scale.diag()).matmul(u.t())
        return wm


class Whitening2dPCA(Whitening2d):
    def whiten_matrix(self, sigma, eye, dim):
        u, eig, _ = sigma.svd()
        scale = eig.rsqrt()
        wm = u.matmul(scale.diag())
        return wm


class Whitening2dIterNorm(Whitening2d):
    def __init__(self, num_features, momentum=0.01, track_running_stats=True, eps=0, dim=0, iterations=5):
        super(Whitening2dIterNorm, self).__init__(num_features,
                                                  momentum,
                                                  track_running_stats,
                                                  eps,
                                                  dim)
        self.iterations = iterations

    def whiten_matrix(self, sigma, eye, dim):
        trace = sigma.trace().reshape(1, 1, 1)
        sigma_norm = sigma.reshape(1, dim, dim) * trace.reciprocal()

        projection = eye.reshape(1, dim, dim)
        for k in range(self.iterations):
            projection = torch.baddbmm(1.5, projection, -0.5, torch.matrix_power(projection, 3), sigma_norm)
        wm = projection.mul_(trace.reciprocal().sqrt())
        return wm

    def extra_repr(self):
        return "features={}, eps={}, momentum={}".format(
            self.num_features, self.eps, self.momentum
        )

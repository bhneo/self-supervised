import abc

import torch
import torch.nn as nn


class Whitening2d(nn.Module):
    def __init__(self, momentum=0.01, track_running_stats=True, eps=0, axis=1, group=1):
        super(Whitening2d, self).__init__()
        self.momentum = momentum
        self.track_running_stats = track_running_stats
        self.eps = eps
        self.axis = axis
        self.group = group
        self.running_mean_registered = False
        self.running_variance_registered = False

    def forward(self, x):
        assert self.axis in (0, 1), "axis must be in (0, 1) !"
        w_dim = x.size(self.axis)

        assert w_dim % self.group == 0, "The dim for whitening should be divisible by group !"

        m = x.mean(0 if self.axis == 1 else 1)
        m = m.view(-1, w_dim) if self.axis == 1 else m.view(w_dim, -1)
        if self.track_running_stats:
            if not self.running_mean_registered:
                self.register_buffer(
                    "running_mean",
                    torch.zeros_like(m)
                )
                self.running_mean_registered = True
            if not self.training and self.axis == 1:  # for inference
                m = self.running_mean
        xn = x - m  # [128, 64]

        sigma_dim = w_dim // self.group
        eye = torch.eye(sigma_dim).type(xn.type()).reshape(1, sigma_dim, sigma_dim).repeat(self.group, 1, 1)  # [128, 128] / [64, 64]
        if self.axis == 1:
            xn_g = xn.reshape(-1, self.group, sigma_dim).permute(1, 0, 2)  # [4, 128, 16]
        else:
            xn_g = xn.reshape(self.group, sigma_dim, -1).permute(0, 2, 1)  # [4, 64, 32]
        f_cov = torch.bmm(xn_g.permute(0, 2, 1), xn_g) / (xn_g.shape[1] - 1)  # [4, 16, 128] * [4, 128, 16] -> [4, 16, 16] / [4, 32, 64] * [4, 64, 32] -> [4, 32, 32]
        sigma = (1 - self.eps) * f_cov + self.eps * eye

        if self.track_running_stats:
            if not self.running_variance_registered:
                self.register_buffer(
                    "running_variance",
                    torch.eye(sigma_dim).reshape(1, sigma_dim, sigma_dim).repeat(self.group, 1, 1)
                )
                self.running_variance_registered = True
            if not self.training:  # for inference
                sigma = self.running_variance

        matrix = self.whiten_matrix(sigma, eye)  # [4, 16, 16] / [4, 32, 32]
        decorrelated = torch.bmm(xn_g, matrix)  # [4, 128, 16] * [4, 16, 16] -> [4, 128, 16] / [4, 64, 32] * [4, 32, 32] -> [4, 64, 32]
    
        if self.axis == 1:
            decorrelated = decorrelated.permute(1, 0, 2).reshape(-1, w_dim)
        else:
            decorrelated = decorrelated.permute(0, 2, 1).reshape(w_dim, -1)
    
        if self.training and self.track_running_stats and self.axis == 0:
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
    
        return decorrelated

    @abc.abstractmethod
    def whiten_matrix(self, sigma, eye):
        pass

    def extra_repr(self):
        return "features={}, eps={}, momentum={}, axis={}, group={}".format(
            self.num_features, self.eps, self.momentum, self.axis, self.group
        )


class Whitening2dCholesky(Whitening2d):
    def whiten_matrix(self, sigma, eye):  # x [group, dim, dim]
        wm = torch.triangular_solve(
            eye, torch.cholesky(sigma), upper=False
        )[0]
        return wm


class Whitening2dZCA(Whitening2d):
    def whiten_matrix(self, sigma, eye):
        u, eig, _ = sigma.svd()
        scale = eig.rsqrt()
        wm = torch.bmm(u, torch.diag_embed(scale))
        wm = torch.bmm(wm, u.permute(0, 2, 1))
        return wm


class Whitening2dPCA(Whitening2d):
    def whiten_matrix(self, sigma, eye):
        u, eig, _ = sigma.svd()
        scale = eig.rsqrt()
        wm = torch.bmm(u, torch.diag_embed(scale))
        return wm


class Whitening2dIterNorm(Whitening2d):
    def __init__(self, momentum=0.01, track_running_stats=True, eps=0, axis=0, group=1, iterations=5):
        super(Whitening2dIterNorm, self).__init__(momentum,
                                                  track_running_stats,
                                                  eps,
                                                  axis,
                                                  group)
        self.iterations = iterations

    def whiten_matrix(self, sigma, eye):
        trace = sigma.diagonal(offset=0, dim1=-1, dim2=-2).sum(-1)
        trace = trace.reshape(sigma.size(0), 1, 1)
        sigma_norm = sigma * trace.reciprocal()

        projection = eye
        for k in range(self.iterations):
            # projection = torch.baddbmm(1.5, projection, -0.5, torch.matrix_power(projection, 3), sigma_norm)
            projection = torch.baddbmm(projection, torch.matrix_power(projection, 3), sigma_norm, beta=1.5, alpha=-0.5)
        wm = projection.mul_(trace.reciprocal().sqrt())
        return wm

    def extra_repr(self):
        return "features={}, eps={}, momentum={}, axis={}, group={}, iterations={}".format(
            self.num_features, self.eps, self.momentum, self.axis, self.group, self.iterations
        )


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import seaborn as sns

    axis = 1
    group = 1
    data = torch.rand(128, 64)

    # whiten = Whitening2dZCA(axis=axis, group=group)
    # whiten = Whitening2dPCA(axis=axis, group=group)
    # whiten = Whitening2dCholesky(axis=axis, group=group)
    whiten = Whitening2dIterNorm(axis=axis, group=group, iterations=5)
    decor_data = whiten(data)

    if axis == 1:
        raw_cov = torch.mm(data.permute(1, 0), data)
        decor_cov = torch.mm(decor_data.permute(1, 0), decor_data)
    else:
        raw_cov = torch.mm(data, data.permute(1, 0))
        decor_cov = torch.mm(decor_data, decor_data.permute(1, 0))

    plt.figure(figsize=(14, 5))
    plt.subplot(1, 2, 1)
    sns.heatmap(raw_cov)
    plt.subplot(1, 2, 2)
    sns.heatmap(decor_cov)
    plt.show()

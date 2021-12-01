import torch
import torch.nn.functional as F
from .whitening import Whitening2dCholesky, Whitening2dIterNorm, Whitening2dZCA, Whitening2dPCA
from .base import BaseMethod
from .norm_mse import norm_mse_loss


class WMSE(BaseMethod):
    """ implements W-MSE loss """

    def __init__(self, cfg):
        """ init whitening transform """
        super().__init__(cfg)
        if cfg.whiten == 'itn':
            self.whitening = Whitening2dIterNorm(eps=cfg.w_eps, track_running_stats=False, iterations=cfg.iter, axis=cfg.w_dim)
        elif cfg.whiten == 'zca':
            self.whitening = Whitening2dZCA(eps=cfg.w_eps, track_running_stats=False, axis=cfg.w_dim)
        elif cfg.whiten == 'pca':
            self.whitening = Whitening2dPCA(eps=cfg.w_eps, track_running_stats=False, axis=cfg.w_dim)
        elif cfg.whiten == 'qa':
            self.whitening = Whitening2dCholesky(eps=cfg.w_eps, track_running_stats=False, axis=cfg.w_dim)
        self.loss_f = norm_mse_loss if cfg.norm else F.mse_loss
        self.w_iter = cfg.w_iter
        self.w_size = cfg.bs if cfg.w_size is None else cfg.w_size

    def forward(self, samples):  # [512, 3, 32, 32] [512, 3, 32, 32]
        bs = len(samples[0])  # 512
        h = [self.model(x.cuda(non_blocking=True)) for x in samples]
        h = self.head(torch.cat(h))  # [1024, 64]
        loss = 0
        for _ in range(self.w_iter):
            z = torch.empty_like(h)
            perm = torch.randperm(bs).view(-1, self.w_size)  # [4, 128] 相当于分成4组做白化
            for idx in perm:  # idx 4组索引，每组128个
                for i in range(len(samples)):  # i -> [0, 1] 分别对变体1和变体2座白化
                    z[idx + i * bs] = self.whitening(h[idx + i * bs])
            for i in range(len(samples) - 1):
                for j in range(i + 1, len(samples)):
                    x0 = z[i * bs : (i + 1) * bs]
                    x1 = z[j * bs : (j + 1) * bs]
                    loss += self.loss_f(x0, x1)
        loss /= self.w_iter * self.num_pairs
        return loss

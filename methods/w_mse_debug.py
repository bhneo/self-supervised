import torch, sys
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import cfg, methods, datasets
from methods.whitening import Whitening2d, Whitening2dIterNorm
from methods.base import BaseMethod
from methods.norm_mse import norm_mse_loss



class WMSE(BaseMethod):
    """ implements W-MSE loss """

    def __init__(self, cfg):
        """ init whitening transform """
        super().__init__(cfg)
        if cfg.whiten == 'itn':
            self.whitening = Whitening2dIterNorm(cfg.emb, eps=cfg.w_eps, track_running_stats=False, iterations=cfg.iter, dim=cfg.w_dim)
        else:
            self.whitening = Whitening2d(cfg.emb, eps=cfg.w_eps, track_running_stats=False)
        self.loss_f = norm_mse_loss if cfg.norm else F.mse_loss
        self.w_iter = cfg.w_iter
        self.w_size = cfg.bs if cfg.w_size is None else cfg.w_size

    def forward(self, samples):
        bs = len(samples[0]) # [[512,3,32,32], [512,3,32,32]]
        h = [self.model(x.cuda(non_blocking=True)) for x in samples] # [[512,512], [512,512]]
        h = self.head(torch.cat(h)) # [1024, 64]
        loss = 0
        for _ in range(self.w_iter):
            z = torch.empty_like(h) # [1024, 64]
            perm = torch.randperm(bs).view(-1, self.w_size) # [4, 128]
            for idx in perm:
                for i in range(len(samples)):
                    z[idx + i * bs] = self.whitening(h[idx + i * bs])
            for i in range(len(samples) - 1):
                for j in range(i + 1, len(samples)):
                    x0 = z[i * bs : (i + 1) * bs]
                    x1 = z[j * bs : (j + 1) * bs]
                    loss += self.loss_f(x0, x1)
        loss /= self.w_iter * self.num_pairs
        return loss


if __name__ == "__main__":
    conf = cfg.get_cfg()
    # conf.whiten = 'itn'
    conf.w_dim = 0

    ds = datasets.get_ds(conf.dataset)(conf.bs, conf, conf.num_workers)
    model = methods.get_method(conf.method)(conf)
    model.cuda().train()

    for samples, _ in ds.train:
        loss = model(samples)
        break

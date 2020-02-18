import argparse
import torch
import wandb
from model import get_model
from eval_lbfgs import eval_lbfgs
from eval_sgd import eval_sgd
from torchvision import models
import cifar10
import stl10
import imagenet
DS = {'cifar10': cifar10, 'stl10': stl10, 'imagenet': imagenet}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--emb', type=int, default=128)
    parser.add_argument(
        '--arch', type=str, choices=dir(models), default='resnet50')
    parser.add_argument('--clf', type=str, default='sgd',
                        choices=['sgd', 'lbfgs'])
    parser.add_argument('--dataset', type=str, default='imagenet',
                        choices=['cifar10', 'stl10', 'imagenet'])
    parser.add_argument('--fname', type=str)
    parser.add_argument('--pretrained', action="store_true")
    cfg = parser.parse_args()
    wandb.init(project="white_ss", config=cfg)

    model, head = get_model(cfg.arch, cfg.emb, cfg.dataset, cfg.pretrained)
    if cfg.fname is None:
        print('evaluating random model')
    else:
        checkpoint = torch.load(cfg.fname)
        model.load_state_dict(checkpoint['model'])

    loader_clf = DS[cfg.dataset].loader_clf()
    loader_test = DS[cfg.dataset].loader_test()

    if cfg.clf == 'sgd':
        eval_sgd(model, head.module.in_features, loader_clf, loader_test)

    elif cfg.clf == 'lbfgs':
        acc = eval_lbfgs(model, loader_clf, loader_test)
        wandb.log({'acc': acc})
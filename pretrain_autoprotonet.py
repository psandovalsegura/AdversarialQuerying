# -*- coding: utf-8 -*-
import os
import argparse
import numpy as np
from tqdm import tqdm
import torch
from torchvision import transforms
from robustness.datasets import ImageNet

from models.autoprotonet import AutoProtoNetEmbedding

from utils import Timer, log


def get_model(options):
    # Choose the embedding network
    is_miniimagenet = options.dataset == 'miniImageNet'
    network = AutoProtoNetEmbedding(activation = options.activation, is_miniimagenet=is_miniimagenet).cuda()
    return network

def get_dataset(options):
    class Lighting(object):
        """
        Lighting noise (see https://git.io/fhBOc)
        """
        def __init__(self, alphastd, eigval, eigvec):
            self.alphastd = alphastd
            self.eigval = eigval
            self.eigvec = eigvec

        def __call__(self, img):
            if self.alphastd == 0:
                return img

            alpha = img.new().resize_(3).normal_(0, self.alphastd)
            rgb = self.eigvec.type_as(img).clone()\
                .mul(alpha.view(1, 3).expand(3, 3))\
                .mul(self.eigval.view(1, 3).expand(3, 3))\
                .sum(1).squeeze()

            return img.add(rgb.view(3, 1, 1).expand_as(img))

    IMAGENET_PCA = {
    'eigval':torch.Tensor([0.2175, 0.0188, 0.0045]),
    'eigvec':torch.Tensor([
        [-0.5675,  0.7192,  0.4009],
        [-0.5808, -0.0045, -0.8140],
        [-0.5836, -0.6948,  0.4203],
    ])
    }

    PRETRAIN_TRANSFORMS_IMAGENET = transforms.Compose([
            transforms.Resize(100),
            transforms.RandomCrop(84),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(
                brightness=0.1,
                contrast=0.1,
                saturation=0.1
            ),
            transforms.ToTensor(),
            Lighting(0.05, IMAGENET_PCA['eigval'], 
                        IMAGENET_PCA['eigvec'])
        ])


    VAL_TRANSFORMS_IMAGENET = transforms.Compose([
        transforms.Resize(100),
        transforms.CenterCrop(84),
        transforms.ToTensor(),
    ])

    ds = ImageNet(data_path='/fs/vulcan-datasets/imagenet',
                  transform_train=PRETRAIN_TRANSFORMS_IMAGENET,
                  transform_test=VAL_TRANSFORMS_IMAGENET)
    
    # Returns (train_loader, test_loader)
    loaders = ds.make_loaders(workers=options.workers, batch_size=options.batch_size)
    return loaders   

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num-epoch', type=int, default=20,
                            help='number of training epochs')
    parser.add_argument('--save-epoch', type=int, default=10,
                            help='frequency of model saving')
    parser.add_argument('--save-path', default='./experiments/exp_1')
    parser.add_argument('--disable_tqdm', action='store_true',
                        help='disables tqdm progress bar')

    parser.add_argument('--activation', type=str, default='LeakyReLU',
                        help='choose which activation function to use. only implemented for R2D2 and ProtoNet')
    parser.add_argument('--dataset', type=str, default='miniImageNet',
                            help='choose which classification head to use. miniImageNet, tieredImageNet, CIFAR_FS, FC100')
    parser.add_argument('--workers', type=int, default=1,
                        help='number of workers for data loaders')
    parser.add_argument('--batch-size', type=int, default=1024,
                        help='number of workers for data loaders')

    opt = parser.parse_args()

    # Use resized ImageNet as pretraining
    (dloader_train, dloader_val) = get_dataset(opt)
    
    if not os.path.isdir(opt.save_path):
        os.makedirs(opt.save_path+'/', )

    
    log_file_path = os.path.join(opt.save_path, "train_log.txt")
    if os.path.exists(log_file_path):
        os.remove(log_file_path)
    log(log_file_path, str(vars(opt)))

    embedding_net = get_model(opt)
    
    optimizer = torch.optim.SGD([{'params': embedding_net.parameters()}], 
                                lr=0.1, momentum=0.9, \
                                weight_decay=5e-4, nesterov=True)
    

    lambda_epoch = lambda e: 1.0 if e < 5 else (0.1 if e < 10 else 0.01 if e < 15 else (0.001))
    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_epoch, last_epoch=-1)

    min_val_loss = np.inf

    timer = Timer()
    mse_loss = torch.nn.MSELoss()
    
    for epoch in range(1, opt.num_epoch + 1):
        # Train on the training split
        
        # Fetch the current epoch's learning rate
        epoch_learning_rate = 0.1
        for param_group in optimizer.param_groups:
            epoch_learning_rate = param_group['lr']
            
        log(log_file_path, 'Train Epoch: {}\tLearning Rate: {:.4f}'.format(
                            epoch, epoch_learning_rate))
        
        embedding_net.train()
        train_losses = []
        for i, batch in enumerate(tqdm(dloader_train, disable=opt.disable_tqdm)):
            x, _ = batch
            x = x.cuda()
            recon_x = embedding_net.forward_plus_decoder(x)
            loss = mse_loss(recon_x, x)
                        
            train_losses.append(loss.item())
            if (i % 100 == 0):
                train_loss_avg = np.mean(np.array(train_losses))
                log(log_file_path, 'Train Epoch: {}\tBatch: [{}/{}]\tLoss: {:.4f}\tLoss Avg: {:.2f}'.format(
                            epoch, i, len(dloader_train), loss.item(), train_loss_avg))
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        lr_scheduler.step()

        # Evaluate on the validation split
        embedding_net.eval()
        val_losses = []
        with torch.no_grad():
            for i, batch in enumerate(tqdm(dloader_val, disable=opt.disable_tqdm)):
                x, _ = batch
                x = x.cuda()
                recon_x = embedding_net.forward_plus_decoder(x)
                loss = mse_loss(recon_x, x)

                val_losses.append(loss.item())

            val_loss_avg = np.mean(np.array(val_losses))
            val_loss_ci95 = 1.96 * np.std(np.array(val_losses)) / np.sqrt(len(val_losses))

            if val_loss_avg < min_val_loss:
                min_val_loss = val_loss_avg
                torch.save({'embedding': embedding_net.state_dict()},\
                           os.path.join(opt.save_path, 'best_model.pth'))
                log(log_file_path, 'Validation Epoch: {}\t\t\tLoss: {:.4f} ± {:.2f} (Best)'\
                    .format(epoch, val_loss_avg, val_loss_ci95))
            else:
                log(log_file_path, 'Validation Epoch: {}\t\t\tLoss: {:.4f} ± {:.2f} '\
                    .format(epoch, val_loss_avg, val_loss_ci95))

        torch.save({'embedding': embedding_net.state_dict()}\
                   , os.path.join(opt.save_path, 'last_epoch.pth'))

        if epoch % opt.save_epoch == 0:
            torch.save({'embedding': embedding_net.state_dict()}\
                       , os.path.join(opt.save_path, 'epoch_{}.pth'.format(epoch)))

        log(log_file_path, 'Elapsed Time: {}/{}\n'.format(timer.measure(), timer.measure(epoch / float(opt.num_epoch))))

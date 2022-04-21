import argparse
import os
import random
from unittest import loader
import numpy
import torch
import logging
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
import copy
from Network import Generator
from PIL import Image

class SRDataset(dset.VisionDataset):
    def __init__(self, root, transform=None, target_transform=None):
        super().__init__(root, transform=transform, target_transform=target_transform)
        samples = self.make_dataset(self.root)
        self.loader = loader
        self.samples = samples
        self.targets = [s[1] for s in samples]
    
    def __getitem__(self, index):
        data_path, target_path = self.samples[index]
        data_img = self.pil_loader(data_path)
        target_img = self.pil_loader(target_path)
        if self.transform is not None:
            data_img = self.transform(data_img)
        if self.target_transform is not None:
            target_img = self.target_transform(target_img)
        return data_img, target_img

    def __len__(self):
        return len(self.samples)

    def pil_loader(self, path):
        with open(path, "rb") as f:
            img = Image.open(f)
            return img.convert("RGB")

    def make_dataset(self, directory):
        data_folder = os.path.join(directory, "DIV2K_train_LR_bicubic", "X2")
        target_folder = os.path.join(directory, "DIV2K_train_HR")
        data_paths = [f for f in sorted(os.listdir(data_folder)) if f.endswith(".png")]
        target_paths = [f for f in sorted(os.listdir(target_folder)) if f.endswith(".png")]
        return list(zip(data_paths, target_paths))


def main():
    torch.set_printoptions(threshold=1)
    logging.basicConfig(level=logging.INFO, filename='train_log.txt')
    parser = argparse.ArgumentParser()
    parser.add_argument('--batchSize', type=int, default=16, help='input batch size')
    parser.add_argument('--imageSize', type=int, default=128, help='the height / width of the input image to network')
    parser.add_argument('--lr', type=float, default=2e-5, help='learning rate')
    parser.add_argument('--nz', type=int, default=512, help='size of the latent z vector')
    parser.add_argument('--workers', type=int, help='number of data loading workers', default=2)
    parser.add_argument('--outf', default='.', help='folder to output images and model checkpoints')

    model = None # Can change to load in a checkpoint

    opt = parser.parse_args()

    try:
        os.makedirs(opt.outf)
    except OSError:
        pass

    manualSeed = 42
    print("Random Seed: ", manualSeed)
    random.seed(manualSeed)
    torch.manual_seed(manualSeed)
    torch.cuda.manual_seed(manualSeed)
    torch.cuda.manual_seed_all(manualSeed)
    numpy.random.seed(manualSeed)

    cudnn.benchmark = True

    dataset = SRDataset('./DIV2K', transform=transforms.Compose([
        transforms.Resize(opt.imageSize),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        transforms.RandomHorizontalFlip(p=0.5)
    ]))

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=opt.batchSize,
                                            shuffle=True, num_workers=int(opt.workers), drop_last=True)

    device = torch.device("cuda:0")
    
    netG = Generator().to(device)
    optimizerG = optim.Adam(netG.parameters(), lr=opt.lr, betas=(0.9, 0.999))

    if model is not None:
        netG.load_state_dict(model['g_state_dict'], strict=True)
        optimizerG.load_state_dict(model['optimizerG_state_dict'])

    print('G params: ' + str(sum(p.numel() for p in netG.parameters() if p.requires_grad)))

    for epoch in range(0 if model is None else model['epoch'] + 1, 1000000):
        for i, sample in enumerate(dataloader, 0):
            netG.requires_grad = True
            netG.zero_grad()
            
            data_img_batch = sample[:, 0, :, :, :]
            target_img_batch = sample[:, 1, :, :, :]

            output_img_batch = netG(data_img_batch)

            loss = torch.nn.L1Loss()
            errG = loss(output_img_batch, target_img_batch)

            errG.backward()
            optimizerG.step()

            ###########################

            log_str = '[%d][%d/%d] Loss_G: %.4f' % (epoch, i, len(dataloader), errG.detach().item())
            print(log_str)
            logging.info(log_str)

            ###########################

            if i % 100 == 0:
                vutils.save_image(data_img_batch, '%s/lr_samples.png' % opt.outf, normalize=True, nrow=3)
                vutils.save_image(target_img_batch, '%s/hr_real_samples.png' % opt.outf, normalize=True, nrow=3)
                vutils.save_image(output_img_batch, '%s/hr_fake_samples.png' % opt.outf, normalize=True, nrow=3)

    torch.save({
        'epoch': epoch,
        'g_state_dict': netG.state_dict(),
        'optimizerG_state_dict': optimizerG.state_dict(),
        'loss_G': errG.detach().item(),
        }, '%s/epoch_%d.pth' % (opt.outf, epoch))

if __name__ == '__main__':
    main()
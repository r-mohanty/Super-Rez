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
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.utils.data import Dataset, DataLoader
from Network import Generator
from PIL import Image

"""
Expected Dataset File Structure:

DIV2K
-> DIV2K_train_HR
    -> HR images
-> DIV2K_train_LR_bicubic
    -> X4
        -> LR images

Download link for HR images: http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_train_HR.zip
Download link for LR images: http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_train_LR_bicubic_X4.zip
^ Paste these links into a web browser to download, then put the unzipped folders into a folder called "DIV2K" ^
"""

class SRDataset(Dataset):
    def __init__(self, root, cropSize):
        self.root = root
        samples = self.make_dataset(self.root)
        self.cropSize = cropSize
        self.samples = samples
    
    def __getitem__(self, index):
        data_path, target_path = self.samples[index]
        input_img = self.pil_loader(data_path)
        target_img = self.pil_loader(target_path)
        x, y = self.transform(input_img, target_img)
        return x, y

    def __len__(self):
        return len(self.samples)

    # We put all our transforms here because the input (LR) and target (HR) images must be transformed identically
    def transform(self, input_img, target_img):
        # Randomly crop both images (make sure that same corresponding regions are being cropped!!!)
        i, j, h, w = transforms.RandomCrop.get_params(input_img, output_size=(self.cropSize, self.cropSize))
        input_img = transforms.functional.crop(input_img, i, j, h, w)
        target_img = transforms.functional.crop(target_img, i*4, j*4, h*4, w*4)

        # Randomly hor flip both images
        if random.random() > 0.5:
            input_img = transforms.functional.hflip(input_img)
            target_img = transforms.functional.hflip(target_img)

        # Randomly 90 deg flip both images
        k = random.randint(0, 3)
        input_img = transforms.functional.rotate(input_img, k * 90)
        target_img = transforms.functional.rotate(target_img, k * 90)

        input_img = transforms.functional.to_tensor(input_img)
        target_img = transforms.functional.to_tensor(target_img)
        return input_img, target_img

    def pil_loader(self, path):
        with open(path, "rb") as f:
            img = Image.open(f)
            return img.convert("RGB")

    def make_dataset(self, directory):
        data_folder = os.path.join(directory, "DIV2K_train_LR_bicubic", "X4")
        target_folder = os.path.join(directory, "DIV2K_train_HR")
        data_paths = [os.path.join(data_folder, f) for f in sorted(os.listdir(data_folder)) if f.endswith(".png")]
        target_paths = [os.path.join(target_folder, f) for f in sorted(os.listdir(target_folder)) if f.endswith(".png")]
        return list(zip(data_paths, target_paths))

def main():
    torch.set_printoptions(threshold=1)
    logging.basicConfig(level=logging.INFO, filename='train_log.txt')
    parser = argparse.ArgumentParser()
    parser.add_argument('--batchSize', type=int, default=16, help='input batch size')
    parser.add_argument('--imageSize', type=int, default=48, help='the height / width of the input image to network')
    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
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

    dataset = SRDataset('./DIV2K', opt.imageSize)

    dataloader = DataLoader(dataset, batch_size=opt.batchSize,
                                            shuffle=True, num_workers=int(opt.workers), drop_last=True)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    netG = Generator().to(device)
    optimizerG = optim.Adam(netG.parameters(), lr=opt.lr, betas=(0.9, 0.999), eps=1e-08)

    if model is not None:
        netG.load_state_dict(model['g_state_dict'], strict=True)
        optimizerG.load_state_dict(model['optimizerG_state_dict'])

    print('G params: ' + str(sum(p.numel() for p in netG.parameters() if p.requires_grad)))

    netG.zero_grad()
    for epoch in range(0 if model is None else model['epoch'] + 1, 100000):
        for i, sample in enumerate(dataloader):
            netG.requires_grad = True
            netG.zero_grad()

            input_img_batch = sample[0]
            target_img_batch = sample[1]

            #### Quick test to make sure that we're cropping properly ####
            # import matplotlib.pyplot as plt
            # f, axarr = plt.subplots(1,2)
            # axarr[0].imshow(input_img_batch[0].permute(1, 2, 0))
            # axarr[1].imshow(target_img_batch[0].permute(1, 2, 0))
            # plt.show()
            # exit()

            output_img_batch = netG(input_img_batch)

            L1 = torch.nn.L1Loss()
            errG = L1(output_img_batch, target_img_batch)
            errG.backward()
            optimizerG.step()

            ###########################

            log_str = '[%d][%d/%d] Loss_G: %.4f' % (epoch, i, len(dataloader), errG.detach().item())
            print(log_str)
            logging.info(log_str)

            ###########################

            if (i + 1) % 100 == 0:
                vutils.save_image(input_img_batch, '%s/lr_samples.png' % opt.outf, normalize=True, nrow=3)
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
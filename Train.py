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
    -> 0001.png ... 0800.png
-> DIV2K_train_LR_bicubic
    -> X4
        -> 0001x4.png ... 0800x4.png
-> DIV2K_valid_HR
    -> 0801.png ... 0900.png
-> DIV2K_valid_LR_bicubic
    -> X4
        -> 0801x4.png ... 0900x4.png

Download link for HR images: http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_train_HR.zip
Download link for LR images: http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_train_LR_bicubic_X4.zip
Download link for validation set HR images: http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_valid_HR.zip
Download link for validation set LR images: http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_valid_LR_bicubic_X4.zip

^ Paste these links into a web browser to download, then put the unzipped folders into a folder called "DIV2K" ^
"""

class SRDataset(Dataset):
    def __init__(self, root, cropSize, input_dir, target_dir):
        self.root = root
        self.input_dir = input_dir
        self.target_dir = target_dir
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
        data_folder = os.path.join(directory, self.input_dir)
        target_folder = os.path.join(directory, self.target_dir)
        data_paths = [os.path.join(data_folder, f) for f in sorted(os.listdir(data_folder)) if f.endswith(".png")]
        target_paths = [os.path.join(target_folder, f) for f in sorted(os.listdir(target_folder)) if f.endswith(".png")]
        return list(zip(data_paths, target_paths))

def train(opt):
    model = None
    # model = torch.load(os.path.join(opt.outf, 'epoch_1.pth'), map_location='cpu')

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

    dataset = SRDataset('./DIV2K', opt.imageSize, os.path.join('DIV2K_train_LR_bicubic', 'X4'), 'DIV2K_train_HR')

    dataloader = DataLoader(dataset, batch_size=opt.batchSize,
                                            shuffle=True, num_workers=int(opt.workers), drop_last=True)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    netG = Generator().to(device)
    optimizerG = optim.Adam(netG.parameters(), lr=opt.lr, betas=(0.9, 0.999), eps=1e-08)

    if model is not None:
        netG.load_state_dict(model['g_state_dict'], strict=True)
        optimizerG.load_state_dict(model['optimizerG_state_dict'])
        print("Loaded model that was trained until epoch %d" % (model['epoch']))

    print('G params: ' + str(sum(p.numel() for p in netG.parameters() if p.requires_grad)))

    minibatch_updates = 0
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

            # Print 5 image batches per epoch. Will be overwritten per epoch, otherwise there would be too many output images saved
            if i % 10 == 0:
                vutils.save_image(input_img_batch, '%s/lr_samples_batch_%04d.png' % (opt.outf, i), normalize=True, nrow=4)
                vutils.save_image(target_img_batch, '%s/hr_real_samples_batch_%04d.png' % (opt.outf, i), normalize=True, nrow=4)
                vutils.save_image(output_img_batch, '%s/hr_fake_samples_batch_%04d.png' % (opt.outf, i), normalize=True, nrow=4)
            
            ###########################

            minibatch_updates += 1
            if (minibatch_updates + 1) % (2 * (10 ** 5)) == 0:
                # How to change learning rate: https://stackoverflow.com/questions/48324152/pytorch-how-to-change-the-learning-rate-of-an-optimizer-at-any-given-moment-no
                for g in optimizerG.param_groups:
                    g['lr'] /= 2

        if epoch % 10 == 0:
            torch.save({
                'epoch': epoch,
                'g_state_dict': netG.state_dict(),
                'optimizerG_state_dict': optimizerG.state_dict(),
                'loss_G': errG.detach().item(),
                }, '%s/epoch_%d.pth' % (opt.outf, epoch))

def validation(opt):
    dataset = SRDataset('./DIV2K', opt.imageSize, os.path.join('DIV2K_valid_LR_bicubic', 'X4'), 'DIV2K_valid_HR')

    dataloader = DataLoader(dataset, batch_size=opt.batchSize,
                                            shuffle=True, num_workers=int(opt.workers), drop_last=True)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    netG = Generator().to(device)
    optimizerG = optim.Adam(netG.parameters(), lr=opt.lr, betas=(0.9, 0.999), eps=1e-08)

    ### Model loaded in here, change to correct one! ###
    model = torch.load(os.path.join(opt.outf, 'epoch_1000.pth'), map_location='cpu')
    ####################################################

    netG.load_state_dict(model['g_state_dict'], strict=True)
    optimizerG.load_state_dict(model['optimizerG_state_dict'])
    print("Loaded model that was trained until epoch %d" % (model['epoch']))
    print('G params: ' + str(sum(p.numel() for p in netG.parameters() if p.requires_grad)))

    netG.eval()
    total_loss = 0

    with torch.no_grad():
        for i, sample in enumerate(dataloader):
            input_img_batch = sample[0]
            target_img_batch = sample[1]
            output_img_batch = netG(input_img_batch)

            L1 = torch.nn.L1Loss()
            errG = L1(output_img_batch, target_img_batch)
            loss = errG.detach().item()
            total_loss += loss
            log_str = '[VALIDATION][%d/%d] Loss_G: %.4f' % (i, len(dataloader), loss)
            print(log_str)
            logging.info(log_str)

            vutils.save_image(input_img_batch, '%s/validation_lr_samples_batch_%04d.png' % (opt.outf, i), normalize=True, nrow=3)
            vutils.save_image(target_img_batch, '%s/validation_hr_real_samples_batch_%04d.png' % (opt.outf, i), normalize=True, nrow=3)
            vutils.save_image(output_img_batch, '%s/validation_hr_fake_samples_batch_%04d.png' % (opt.outf, i), normalize=True, nrow=3)

    return total_loss / len(dataloader)


def main():
    torch.set_printoptions(threshold=1)
    logging.basicConfig(level=logging.INFO, filename='train_log.txt')
    parser = argparse.ArgumentParser()
    parser.add_argument('--batchSize', type=int, default=16, help='input batch size')
    parser.add_argument('--imageSize', type=int, default=48, help='the height / width of the input image to network')
    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
    parser.add_argument('--workers', type=int, help='number of data loading workers', default=2)
    parser.add_argument('--outf', default='./Output', help='folder to output images and model checkpoints')
    parser.add_argument('--mode', type=str, default='TRAIN', help='one of "TRAIN", "VALIDATION", or "TEST"')
    opt = parser.parse_args()

    if opt.mode == 'TRAIN':
        train(opt)
    elif opt.mode == 'VALIDATION':
        avg_loss = validation(opt)
        log_str = '[VALIDATION] Average loss: %.4f' % (avg_loss)
        print(log_str)
        logging.info(log_str)
    elif opt.mode == 'TEST':
        print("Test hasn't been implemented yet!")
    else:
        print('ERROR: Invalid mode! mode should be one of "TRAIN", "VALIDATION", or "TEST"')

if __name__ == '__main__':
    main()

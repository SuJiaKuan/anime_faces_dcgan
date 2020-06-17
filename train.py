import argparse
import os
import pathlib
import shutil

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.utils as vutils

from dcgan.data import load_data
from dcgan.model import Generator
from dcgan.model import Discriminator
from dcgan.model import init_weights


REAL_LABEL = 1
FAKE_LABEL = 0


def parse_args():
    parser = argparse.ArgumentParser(
        description='Training script for Simpson DCGAN',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        '--data_folder',
        type=str,
        default='anime_faces',
        help='Folder that contains training images of Simpsons faces',
    )
    parser.add_argument(
        '--data_workers',
        type=int,
        default=2,
        help='Number of workers to load data',
    )
    parser.add_argument(
        '--image_size',
        type=int,
        default=64,
        help='Size of generated images',
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=128,
        help='Batch size for training',
    )
    parser.add_argument(
        '--z_size',
        type=int,
        default=100,
        help='Size of generator input z',
    )
    parser.add_argument(
        '--lr',
        type=int,
        default=0.0002,
        help='Learning rate for training',
    )
    parser.add_argument(
        '--beta1',
        type=float,
        default=0.5,
        help='Beta1 for Adam optimizer',
    )
    parser.add_argument(
        '--num_epochs',
        type=int,
        default=20,
        help='Number of training epochs',
    )
    parser.add_argument(
        '--output',
        type=str,
        default='output',
        help='Output directory',
    )

    args = parser.parse_args()

    return args


def main(args):
    # Delete previous generated folder if it exists.
    if os.path.exists(args.output):
        shutil.rmtree(args.output)

    # Create a folder to put the generated images.
    generated_path = os.path.join(args.output, 'generated')
    pathlib.Path(generated_path).mkdir(parents=True, exist_ok=True)

    # Load real images from disk.
    dataloader = load_data(args.data_folder,
                           args.image_size,
                           args.batch_size,
                           args.data_workers)

    # Choose device (GPU or CPU).
    device = torch.device('cuda:0' if (torch.cuda.is_available()) else 'cpu')

    # Constructure the network architectures for Generator and Discrimiantor.
    net_g = Generator(args.z_size).to(device)
    net_d = Discriminator().to(device)

    # Initializa the weights of Generator and Discriminator.
    net_g.apply(init_weights)
    net_d.apply(init_weights)

    # Use Binary Cross Entropy as the loss function.
    criterion = nn.BCELoss()

    # Use Adam as optimizer for Generator and Discriminator.
    optimizer_g = optim.Adam(net_g.parameters(),
                             lr=args.lr,
                             betas=(args.beta1, 0.999))
    optimizer_d = optim.Adam(net_d.parameters(),
                             lr=args.lr,
                             betas=(args.beta1, 0.999))

    # Allocate a list of fixed noises to see the generated images for different
    # training iteration / epoch.
    fixed_noise = torch.randn(64, args.z_size, 1, 1, device=device)

    # Create list to record the loss of Generator and Discriminator for each
    # training iteration.
    losses_g = []
    losses_d = []

    for epoch in range(args.num_epochs):
        for batch_idx, data in enumerate(dataloader, 0):
            # Training for Discriminator, step 1: treat real images as postive
            # examples.
            net_d.zero_grad()
            imgs_real = data[0].to(device)
            batch_size = imgs_real.shape[0]
            labels = torch.full((batch_size,), REAL_LABEL, device=device)
            predictions = net_d(imgs_real).view(-1)
            err_d_real = criterion(predictions, labels)
            err_d_real.backward(retain_graph=True)
            d_x = predictions.mean().item()

            # Training for Discriminator, step 1: treat generated images as
            # negative examples.
            noise = torch.randn(batch_size, args.z_size, 1, 1, device=device)
            imgs_fake = net_g(noise)
            labels.fill_(FAKE_LABEL)
            predictions = net_d(imgs_fake).view(-1)
            err_d_fake = criterion(predictions, labels)
            err_d_fake.backward(retain_graph=True)
            d_g_z1 = predictions.mean().item()
            err_d = err_d_real + err_d_fake
            optimizer_d.step()

            # Training for Generator: treat generated images as postive
            # examples.
            net_g.zero_grad()
            labels.fill_(REAL_LABEL)
            predictions = net_d(imgs_fake).view(-1)
            err_g = criterion(predictions, labels)
            err_g.backward()
            d_g_z2 = predictions.mean().item()
            optimizer_g.step()

            # Print training information periodically.
            if batch_idx % 10 == 0:
                print(f'[{epoch}/{args.num_epochs}]'
                      f'[{batch_idx}/{len(dataloader)}]'
                      f'\tLoss D: {err_d.item():.4f}'
                      f'\tLoss G: {err_g.item():.4f}'
                      f'\tD(x): {d_x:.4f}'
                      f'\tD(G(z)): {d_g_z1:.4f} / {d_g_z2:.4f}')

            # Append current Generator and Discriminator loss to loss lists.
            losses_d.append(err_d.item())
            losses_g.append(err_g.item())

            # Generate fake image and save it to disk periodically.
            if (batch_idx % 10 == 0) or (batch_idx == len(dataloader) - 1):
                with torch.no_grad():
                    imgs_fake = net_g(fixed_noise).detach().cpu()
                    grid_img = vutils.make_grid(imgs_fake,
                                                padding=2,
                                                normalize=True)
                    save_path = os.path.join(generated_path,
                                             f'{epoch}_{batch_idx}.jpg')
                    vutils.save_image(grid_img, save_path)

    # Draw a plot for losses and save it to disk.
    plt.figure(figsize=(10, 5))
    plt.title('Generator and Discriminator Loss During Training')
    plt.plot(losses_g, label='Generator')
    plt.plot(losses_d, label='Discriminator')
    plt.xlabel('Iterations')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(os.path.join(args.output, 'loss.png'))


if __name__ == '__main__':
    args = parse_args()

    main(args)

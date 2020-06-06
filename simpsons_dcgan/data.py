import torch
import torchvision.datasets as dset
import torchvision.transforms as transforms


def load_data(data_folder, image_size, batch_size, data_workers):
    dataset = dset.ImageFolder(
        root=data_folder,
        transform=transforms.Compose([
            transforms.Resize(image_size),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]),
    )
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=data_workers,
    )

    return dataloader

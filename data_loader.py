import torch
from torchvision import datasets, transforms

def dataloader(train_batch_size):
    train_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder('data_supervised_shuffled/train/', transforms.Compose([
            transforms.Resize((84, 84)),  # width, height
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.618, 0.6373, 0.633), std=(0.2538, 0.2394, 0.2427))
        ])), batch_size=train_batch_size, num_workers=0, shuffle=True)

    valid_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder('data_supervised_shuffled/validation/', transforms.Compose([
            transforms.Resize((84, 84)),  # width, height
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.615, 0.631, 0.629), std=(0.252, 0.240, 0.242)),
        ])), batch_size=train_batch_size, num_workers=0, shuffle=True)

    test_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder('data_supervised_shuffled/test/', transforms.Compose([
            transforms.Resize((84, 84)),  # width, height
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.618, 0.6373, 0.633), std=(0.2538, 0.2394, 0.2427))
        ])), batch_size=1, num_workers=0, shuffle=False)
    return train_loader, valid_loader, test_loader
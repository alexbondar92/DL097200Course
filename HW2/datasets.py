import torchvision.datasets as DSet
import torchvision.transforms as transforms

transform_test = transforms.Compose([transforms.ToTensor(),
                                     transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])

def transform_train(mode=1):
    """Choose the transformation mode out of 4 available options"""
    if mode == 1:
        # Augmentation of random padding + cropping, random horizontal flip and normalization
        transform = transforms.Compose([transforms.RandomCrop(32, padding=4),
                                        transforms.RandomHorizontalFlip(),
                                        transforms.ToTensor(),
                                        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
    elif mode == 2:
        # Same as 1 but with random +- 50% brightness adjust
        transform = transforms.Compose([transforms.RandomCrop(32, padding=4),
                                        transforms.RandomHorizontalFlip(),
                                        transforms.ToTensor(),
                                        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                                        transforms.ColorJitter(brightness=0.5)])
    elif mode == 3:
        # Same as 1 but with random +- 50% contrast adjust
        transform = transforms.Compose([transforms.RandomCrop(32, padding=4),
                                        transforms.RandomHorizontalFlip(),
                                        transforms.ToTensor(),
                                        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                                        transforms.ColorJitter(contrast=0.5)])
    elif mode == 4:
        # 1 + 2 + 3 adjusts
        transform = transforms.Compose([transforms.RandomCrop(32, padding=4),
                                        transforms.RandomHorizontalFlip(),
                                        transforms.ToTensor(),
                                        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                                        transforms.ColorJitter(brightness=0.5, contrast=0.5)])
    return transform

def train_dataset(transform_mode=1):
    return DSet.CIFAR10(root='./data',
                        train=True,
                        transform=transform_train(transform_mode),
                        download=True)

def test_dataset():
    return DSet.CIFAR10(root='./data', train=False, transform=transform_test)

import torch
import torchvision.datasets as DSet
import torchvision.transforms as transforms
from sklearn.model_selection import train_test_split

test_size = 1/7

transform = transforms.Compose(
         [transforms.ToTensor(),
         transforms.Normalize((0.1307,), (0.3081,))])

def train_dataset():
    train_ds = DSet.MNIST(root='./data',
                          train=True,
                          transform=transform,
                          download=True)

    # Split the data and take only the training set
    X_train, _, y_train, _ = train_test_split(train_ds.train_data.tolist(), train_ds.train_labels.tolist(),
                                              test_size=test_size, random_state=42, stratify=train_ds.train_labels)

    # Convert to tensors
    train_ds.train_data = torch.ByteTensor(X_train)
    train_ds.train_labels = torch.LongTensor(y_train)

    return train_ds


def validation_dataset():
    validation_ds = DSet.MNIST(root='./data',
                               train=True,
                               transform=transform,
                               download=True)

    # Split the data and take only the validation set
    _, X_valid, _, y_valid = train_test_split(validation_ds.train_data.tolist(), validation_ds.train_labels.tolist(),
                                              test_size=test_size, random_state=42, stratify=validation_ds.train_labels)

    # Convert to tensors
    validation_ds.train_data = torch.ByteTensor(X_valid)
    validation_ds.train_labels = torch.LongTensor(y_valid)

    return validation_ds

def test_dataset():
    return DSet.MNIST(root='./data', train=False, transform=transform)
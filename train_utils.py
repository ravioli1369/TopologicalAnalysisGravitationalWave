from torch.utils.data import random_split
from model import TabularDataset
def dataset_split(feat, label, train_ratio = 0.7, val_ratio=0.15, test_ratio=0.15):


    dataset = TabularDataset(feat, label)

    dataset_size = len(dataset)
    train_size = int(train_ratio * dataset_size)
    val_size = int(val_ratio * dataset_size)
    test_size = dataset_size - train_size - val_size


    #train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

    return random_split(dataset, [train_size, val_size, test_size])

    
from torchvision import datasets, transforms


def get_transform(is_train=True):

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    if is_train:
        return transforms.Compose(
            [transforms.Pad(4),
             transforms.RandomSizedCrop(32),
             transforms.RandomHorizontalFlip(),
             transforms.ToTensor(),
             normalize]
        )
    else:
        return transforms.Compose(
            [transforms.ToTensor(),
             normalize]
        )


# cifar10_train_data = datasets.CIFAR10("./data", train=True, download=True, transform=get_transform(is_train=True))
# cifar10_test_data = datasets.CIFAR10("./data", train=False, download=True, transform=get_transform(is_train=False))
cifar100_train_data = datasets.CIFAR100("./data", train=True, download=True, transform=get_transform(is_train=True))
cifar100_test_data = datasets.CIFAR100("./data", train=False, download=True, transform=get_transform(is_train=False))

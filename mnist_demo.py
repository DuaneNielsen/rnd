import torch
import torch.nn as nn
from torch.nn.modules.loss import MSELoss
from torch.optim import Adam
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.sampler import Sampler
from torchvision.datasets import MNIST
import torchvision.transforms as transforms
from storm.vis import UniImageViewer
import statistics


if __name__ == '__main__':
    in_features = 28 * 28
    h_size = 20
    epochs = 20
    batch_size = 200

    rand_net = nn.Sequential(nn.Linear(in_features, h_size), nn.BatchNorm1d(h_size), nn.ReLU(), nn.Linear(h_size, 1), nn.ReLU())
    dist_net = nn.Sequential(nn.Linear(in_features, h_size), nn.BatchNorm1d(h_size), nn.ReLU(), nn.Linear(h_size, 1), nn.ReLU())


    class MNISTZeroSampler(Sampler):
        """
        Samples only zeros from MNIST
        """
        def __init__(self, datasource):
            super().__init__(datasource)
            self.datasource = datasource
            self.zero_elems = []
            for index in range(len(self.datasource)):
                image, label = self.datasource[index]
                if label.item() == 0:
                    self.zero_elems.append(index)

        def __iter__(self):
            return iter(self.zero_elems)

        def __len__(self):
            return len(self.zero_elems)


    class MNISTNonZeroSampler(Sampler):
        """
        Samples everything but zeros from MNIST
        """
        def __init__(self, datasource):
            super().__init__(datasource)
            self.datasource = datasource
            self.zero_elems = []
            for index in range(len(self.datasource)):
                image, label = self.datasource[index]
                if label.item() != 0:
                    self.zero_elems.append(index)

        def __iter__(self):
            return iter(self.zero_elems)

        def __len__(self):
            return len(self.zero_elems)


    mnist = MNIST('mnistdata', download=True,
                        transform=transforms.Compose([
                      transforms.ToTensor(),
                      transforms.Normalize((0.1307,), (0.3081,))
                  ]))
    mnist_zeros = DataLoader(mnist, batch_size=batch_size, sampler=MNISTZeroSampler(mnist))
    mnist_non_zeros = DataLoader(mnist, batch_size=batch_size, sampler=MNISTNonZeroSampler(mnist))
    mnist_normal = DataLoader(mnist, batch_size=batch_size)

    #viewer = UniImageViewer('mnist')

    criterion = MSELoss()
    optimizer = Adam(lr=1e-3, params=dist_net.parameters())

    for epoch in range(epochs):
        train_losses = []
        for image, label in mnist_zeros:
            flat_image = image.squeeze().view(-1, in_features)
            target = rand_net(flat_image)
            target = target.detach()
            optimizer.zero_grad()
            prediction = dist_net(flat_image)
            loss = criterion(prediction, target)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

        test_normal_losses = []
        for image, label in mnist_normal:
            flat_image = image.squeeze().view(-1, in_features)
            target = rand_net(flat_image)
            target = target.detach()
            prediction = dist_net(flat_image)
            loss = criterion(prediction, target)
            test_normal_losses.append(loss.item())

        test_zero_losses = []
        for image, label in mnist_zeros:
            flat_image = image.squeeze().view(-1, in_features)
            target = rand_net(flat_image)
            target = target.detach()
            prediction = dist_net(flat_image)
            loss = criterion(prediction, target)
            test_zero_losses.append(loss.item())

        test_nonzero_losses = []
        for image, label in mnist_non_zeros:
            flat_image = image.squeeze().view(-1, in_features)
            target = rand_net(flat_image)
            target = target.detach()
            prediction = dist_net(flat_image)
            loss = criterion(prediction, target)
            test_nonzero_losses.append(loss.item())

        mean_training_loss = statistics.mean(train_losses)

        mean_normal_loss = statistics.mean(test_normal_losses)
        sigma_normal_loss = statistics.stdev(test_normal_losses)

        mean_test_zero_loss = statistics.mean(test_zero_losses)
        mean_test_nonzero_loss = statistics.mean(test_nonzero_losses)

        def normalize(value, mean, sigma):
            return (value - mean) / sigma

        normalized_zero = normalize(mean_test_zero_loss, mean_normal_loss, sigma_normal_loss)
        normalized_nonzero = normalize(mean_test_nonzero_loss, mean_normal_loss, sigma_normal_loss)


        print(f'epoch: {epoch} '
              f'train:{mean_training_loss:.6f} '
              f'test_normal_mean:{mean_normal_loss:.6f} '
              f'test_normal_sigma:{sigma_normal_loss:.6f} '
              f'test_zero:{mean_test_zero_loss:.6f} '
              f'test_nonzero:{mean_test_nonzero_loss:.6f} '
              f'normalized_zero: {normalized_zero:.6f} '
              f'normalized_nonzero: {normalized_nonzero:.6f} ')

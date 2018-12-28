import torch.nn as nn
from torch.nn.modules.loss import MSELoss
from torch.optim import Adam
from torch.utils.data.dataloader import DataLoader
from torchvision.datasets import MNIST
import torchvision.transforms as transforms

from data import MNISTZeroSampler, MNISTNonZeroSampler, compute_covar
from storm.vis import UniImageViewer
import statistics

if __name__ == '__main__':
    in_features = 28 * 28
    h_size = 20
    epochs = 20
    batch_size = 200

    rand_net = nn.Sequential(nn.Linear(in_features, h_size), nn.BatchNorm1d(h_size), nn.ReLU(), nn.Linear(h_size, 1),
                             nn.ReLU())
    dist_net = nn.Sequential(nn.Linear(in_features, h_size), nn.BatchNorm1d(h_size), nn.ReLU(), nn.Linear(h_size, 1),
                             nn.ReLU())

    mnist = MNIST('mnistdata', download=True,
                  transform=transforms.Compose([
                      transforms.ToTensor()
                  ]))

    mean, stdev = compute_covar(mnist, index=0, batch_size=200)

    mnist_white = MNIST('mnistdata', download=True,
                        transform=transforms.Compose([
                            transforms.ToTensor(),
                            transforms.Normalize(mean, stdev)
                        ]))

    mnist_normal = DataLoader(mnist_white, batch_size=batch_size)
    mnist_zeros = DataLoader(mnist_white, batch_size=batch_size, sampler=MNISTZeroSampler(mnist))
    mnist_non_zeros = DataLoader(mnist_white, batch_size=batch_size, sampler=MNISTNonZeroSampler(mnist))

    viewer = UniImageViewer('mnist')

    criterion = MSELoss()
    optimizer = Adam(lr=1e-4, params=dist_net.parameters())

    for epoch in range(epochs):
        train_losses = []
        for image, label in mnist_zeros:
            # viewer.render(image, block=True)
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
            return (value - mean) + 1e-10 / (sigma + 1e-10)


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

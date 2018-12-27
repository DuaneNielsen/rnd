import torch
import torch.nn as nn
from torch.nn.modules.loss import MSELoss
from torch.optim import Adam
from torch.utils.data.dataloader import DataLoader
from torchvision.datasets import MNIST
import torchvision.transforms as transforms
from torch.nn.functional import softmax
import statistics

if __name__ == '__main__':
    in_features = 28 * 28
    h_size = 20
    epochs = 20
    batch_size = 200
    novel_batch_size = 2
    sample = False

    rand_net = nn.Sequential(nn.Linear(in_features, h_size), nn.BatchNorm1d(h_size), nn.ReLU(), nn.Linear(h_size, 1),
                             nn.ReLU())
    dist_net = nn.Sequential(nn.Linear(in_features, h_size), nn.BatchNorm1d(h_size), nn.ReLU(), nn.Linear(h_size, 1),
                             nn.ReLU())

    mnist = MNIST('mnistdata', download=True,
                  transform=transforms.Compose([
                      transforms.ToTensor(),
                      transforms.Normalize((0.1307,), (0.3081,))
                  ]))
    mnist = DataLoader(mnist, batch_size=batch_size)

    criterion = MSELoss()
    optimizer = Adam(lr=1e-3, params=dist_net.parameters())

    for epoch in range(epochs):
        train_losses = []
        for image, label in mnist:
            # compute the novelty value of the image
            flat_image = image.squeeze().view(-1, in_features)
            target = rand_net(flat_image)
            prediction = dist_net(flat_image)
            error = (prediction - target) ** 2
            weighted_novelty = softmax(error.squeeze(), dim=0)

            if sample:
                # sample based on novelty score (more novel = more likely)
                indices = torch.multinomial(weighted_novelty, novel_batch_size, replacement=False)
            else:
                values, indices = torch.topk(weighted_novelty, novel_batch_size)

            novel_images = flat_image[indices]
            novel_labels = label[indices]

            # train distillation network on selected images
            novel_targets = rand_net(novel_images)
            novel_targets.detach()

            optimizer.zero_grad()
            prediction = dist_net(novel_images)
            loss = criterion(prediction, novel_targets)
            loss.backward()
            optimizer.step()

            train_losses.append(loss.item())
            print(novel_labels, statistics.mean(train_losses))

import torch
import torch.nn as nn
from torch.nn.modules.loss import MSELoss
from torch.optim import Adam
from torch.utils.data.dataloader import DataLoader
from torchvision.datasets import MNIST
import torchvision.transforms as transforms
from torch.nn.functional import softmax
import statistics
from tensorboardX import SummaryWriter
from torch.utils.data.sampler import Sampler
import random
from data import compute_covar

if __name__ == '__main__':
    in_features = 28 * 28
    h_size = 50
    epochs = 30
    batch_size = 1000
    novel_batch_size = 200
    sample = False
    magnitude_norm = False
    L1_distance = True

    class MNISTBiasedSampler(Sampler):
        """
        Samples only zeros from MNIST
        """

        def __init__(self, datasource, index, drop_freq):
            super().__init__(datasource)
            self.datasource = datasource
            self.sampled_elems = []
            for index in range(len(self.datasource)):
                image, label = self.datasource[index]
                drop = random.random() < drop_freq
                if label.item() in {1, 2, 3}:
                    if not drop:
                        self.sampled_elems.append(index)
                else:
                    self.sampled_elems.append(index)

        def __iter__(self):
            return iter(self.sampled_elems)

        def __len__(self):
            return len(self.sampled_elems)


    mnist = MNIST('mnistdata', download=True,
                  transform=transforms.Compose([
                      transforms.ToTensor()
                  ]))

    mean, stdev = compute_covar(mnist, index=0, batch_size=200)

    mnist = MNIST('mnistdata', download=True,
                  transform=transforms.Compose([
                      transforms.ToTensor(),
                      transforms.Normalize(mean, stdev)
                  ]))
    mnist_biased = DataLoader(mnist, batch_size=batch_size, sampler=MNISTBiasedSampler(mnist, {1,2,3}, 0.2))
    mnist = DataLoader(mnist, batch_size=batch_size)

    for run in range(10):
        tb = SummaryWriter(f'runs/{run:2d}/')

        rand_net = nn.Sequential(nn.Linear(in_features, h_size),
                                 nn.BatchNorm1d(h_size),
                                 nn.ReLU(),
                                 nn.Linear(h_size, h_size),
                                 nn.BatchNorm1d(h_size),
                                 nn.ReLU(),
                                 nn.Linear(h_size, 1),
                                 nn.ReLU())

        dist_net = nn.Sequential(nn.Linear(in_features, h_size),
                                 nn.BatchNorm1d(h_size),
                                 nn.ReLU(),
                                 nn.Linear(h_size, h_size),
                                 nn.BatchNorm1d(h_size),
                                 nn.ReLU(),
                                 nn.Linear(h_size, 1),
                                 nn.ReLU())

        optimizer = Adam(lr=1e-5, params=dist_net.parameters())

        criterion = MSELoss()

        base_counts = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        base_freq = None
        novelty_counts = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        novelty_freq = None
        global_step = 0

        for epoch in range(epochs):
            train_losses = []
            for image, labels in mnist_biased:
                flat_image = image.squeeze().view(-1, in_features)

                for label in labels:
                    base_counts[label.item()] += 1
                total = sum(base_counts)
                base_freq = [x / total for x in base_counts]

                # compute the novelty value of the image
                target = rand_net(flat_image)
                prediction = dist_net(flat_image)
                if L1_distance:
                    error = (prediction - target)
                    error[error < 0] = error[error < 0] * -1
                else:
                    error = (prediction - target) ** 2
                weighted_novelty = softmax(error.squeeze(), dim=0)

                if sample:
                    # sample based on novelty score (more novel = more likely)
                    indices = torch.multinomial(weighted_novelty, novel_batch_size, replacement=False)
                else:
                    values, indices = torch.topk(weighted_novelty, novel_batch_size)

                novel_images = flat_image[indices]
                novel_labels = labels[indices]

                # train distillation network on selected images
                novel_targets = rand_net(novel_images)
                novel_targets.detach()

                optimizer.zero_grad()
                prediction = dist_net(novel_images)
                loss = criterion(prediction, novel_targets)
                loss.backward()
                optimizer.step()
                train_losses.append(loss.item())

                for label in novel_labels:
                    novelty_counts[label.item()] += 1
                novelty_total = sum(novelty_counts)
                novelty_freq = [x / novelty_total for x in novelty_counts]

                freq_diff = [base - novelty for base, novelty in zip(base_freq, novelty_freq)]
                print(freq_diff, loss.item())

                for i, freq in enumerate(freq_diff):
                    tb.add_scalar(f'freq_diff_{i}', freq, global_step)
                tb.add_scalar('loss', loss.item(), global_step)
                tb.add_histogram('labels', labels.cpu().numpy(), epoch)
                tb.add_histogram('novel_labels', novel_labels.cpu().numpy(), epoch)
                global_step += 1

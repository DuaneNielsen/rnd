import torch
from torch.utils.data import Sampler, DataLoader
from tqdm import tqdm


class MNISTZeroSampler(Sampler):
    """
    Samples only zeros from MNIST
    """

    def __init__(self, datasource):
        super().__init__(datasource)
        self.datasource = datasource
        self.zero_elems = []
        len_datasource_tq = tqdm(range(len(self.datasource)))
        len_datasource_tq.set_description('building index of zeroes')
        for index in len_datasource_tq:
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
        len_datasource_tq = tqdm(range(len(self.datasource)))
        len_datasource_tq.set_description('building index of non zeroes')
        for index in len_datasource_tq:
            image, label = self.datasource[index]
            if label.item() != 0:
                self.zero_elems.append(index)

    def __iter__(self):
        return iter(self.zero_elems)

    def __len__(self):
        return len(self.zero_elems)


def compute_covar(dataset, index, batch_size=200):
    """
    Compute covariance matrix
    :param dataset: a dataset to compute over
    :param index: index of item in dataset tuple to compute covar for
    :param batch_size: batch size to use for the computation
    :return: (mean, stdev) of covariance matrix
    """

    dataloader = DataLoader(dataset, batch_size=batch_size)

    example = dataset[0][index]
    sum_image = torch.zeros_like(example).double()
    sum_squares_dev = torch.zeros_like(example).double()

    mnist_normal_tq = tqdm(dataloader)
    mnist_normal_tq.set_description('computing covariance matrix - mean')
    for data in mnist_normal_tq:
        sum_image = sum_image + torch.sum(data[index].double(), dim=0)

    mean = sum_image / len(dataset)

    mnist_normal_tq = tqdm(dataloader)
    mnist_normal_tq.set_description('computing covariance matrix - stdev')
    for data in mnist_normal_tq:
        squared_dev = (data[index].double() - mean) ** 2
        sum_squares_dev = sum_squares_dev + torch.sum(squared_dev ** 2, dim=0)

    stdev = torch.sqrt((sum_squares_dev / (len(dataset) - 1)))
    stdev[stdev == 0.0] = 1e-12

    return mean.float(), stdev.float()


class Clip(object):
    def __init__(self, min, max):
        self.min = min
        self.max = max

    def __call__(self, x):
        return x.clamp(self.min, self.max)

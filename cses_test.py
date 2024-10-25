from torch.utils.data import DataLoader, Subset, ConcatDataset
from timesead.data.cses_dataset_flatX import CsesDataset as datasetflatX
from timesead.data.cses_dataset_flatX0 import CsesDataset as datasetflat0
from timesead.data.cses_dataset_polar import CsesDataset as datasetflatP
from timesead.data.cses_dataset_flat import CsesDataset as datasetflatXYZ
from timesead.models.baselines.kmeans import KMeansAD
import matplotlib.pyplot as plot
import numpy
import os
import random

def randindex(max, dataset):
    indexes = []
    if dataset != None:
        for _ in range(max):
            indexes.append(random.randint(0, len(dataset) - 1))

    return numpy.array(indexes, dtype=int)


def main():
    cses_train_ds1 = datasetflatX()
    cses_train_ds2 = datasetflat0()
    cses_train_ds3 = datasetflatP()
    cses_train_ds4 = datasetflatXYZ()

    #cses_test_ds = CsesDataset(training=False)

    b1 = randindex(50, cses_train_ds1)
    b2 = randindex(50, cses_train_ds2)
    b3 = randindex(50, cses_train_ds3)
    b4 = randindex(50, cses_train_ds4)

    subset1 = Subset(cses_train_ds1, b1)
    subset2 = Subset(cses_train_ds2, b2)
    subset3 = Subset(cses_train_ds3, b3)
    subset4 = Subset(cses_train_ds4, b4)

    cses_train_tot = ConcatDataset([subset1, subset2, subset3, subset4])

    cses_train_dl = DataLoader(cses_train_tot)
    # cses_test_dl = DataLoader(cses_test_ds, num_workers=1)

    Model = KMeansAD(256, 256000)
    Model.fit(cses_train_dl)

    plots_dir = 'Plots_kmeans_test'
    if not os.path.exists(plots_dir):
        os.makedirs(plots_dir)

    tot_score = []

    i = 0

    for item, idx in cses_train_dl:
        scores = Model.compute_online_anomaly_score(item)
        scores = scores.numpy()
        tot_score.append(scores)

        plot.figure(figsize=(12, 6))
        plot.plot(item[0, :, 0, 0].numpy(), label='E_normalized', color='blue', alpha=0.6)

        anomalies = numpy.where(scores > 1)[0]
        plot.scatter(anomalies, item[0, anomalies, 0, 0].numpy(), color='red', label='Anomalies', s=15)

        plot.title('Test')
        plot.xlabel('Time')
        plot.ylabel('E')

        plot.savefig(os.path.join(plots_dir, f'{i}.png'))
        plot.close()
        i += 1

    plot.figure(figsize=(12, 8))
    plot.hist(tot_score)
    plot.title('Test')
    plot.show()

if __name__ == '__main__':
    main()
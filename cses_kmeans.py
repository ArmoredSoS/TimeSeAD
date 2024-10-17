from torch.utils.data import DataLoader
from timesead.data.cses_dataset_polar import CsesDataset
from timesead.models.baselines.kmeans import KMeansAD
import matplotlib.pyplot as plot
import numpy
import os

#0.1 X
#0.1 polar

def main():
    cses_train_ds = CsesDataset()
    #cses_test_ds = CsesDataset(training=False)

    cses_train_dl = DataLoader(cses_train_ds)
    # cses_test_dl = DataLoader(cses_test_ds, num_workers=1)

    Model = KMeansAD(256, 256000)
    Model.fit(cses_train_dl)

    plots_dir = 'Plots_kmeans_polar_2'
    if not os.path.exists(plots_dir):
        os.makedirs(plots_dir)

    tot_score = []

    i = 0

    for item, idx in cses_train_dl:
        scores = Model.compute_online_anomaly_score(item)
        scores = scores.numpy()
        tot_score.append(scores)

        #appiattisci le dimensioni nel dataset e trattalo come il caso di solo E_X ma con 3 features
        plot.figure(figsize=(12, 6))
        plot.plot(item[0, :, 0, 0].numpy(), label='E_normalized', color='blue', alpha=0.6)

        anomalies = numpy.where(scores > 0.02)[0]
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
from torch.utils.data import DataLoader
from timesead.data.cses_dataset_flatX import CsesDataset
from timesead.models.baselines.knn import KNNAD
import matplotlib.pyplot as plot
import numpy
import os

def main():
    cses_train_ds = CsesDataset()
    #cses_test_ds = CsesDataset(training=False)
    cses_train_dl = DataLoader(cses_train_ds, num_workers=0)
    # cses_test_dl = DataLoader(cses_test_ds, num_workers=1)

    Model = KNNAD(5, 'mean')
    Model.fit(cses_train_dl)

    plots_dir = 'Plots_knn_X'
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

        anomalies = numpy.where(scores > 1.0)[0]
        plot.scatter(anomalies, item[0, anomalies, 0, 0].numpy(), color='red', label='Anomalies', s=10)

        plot.title('Test')
        plot.xlabel('Time')
        plot.ylabel('E')

        plot.savefig(os.path.join(plots_dir, f'{i}.png'))
        plot.close()

        i += 1

    plot.figure(figsize=(8, 3))
    plot.hist(tot_score)
    plot.title('Test')
    plot.show()
    
if __name__ == '__main__':
    main()
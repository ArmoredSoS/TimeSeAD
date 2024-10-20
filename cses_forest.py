from torch.utils.data import DataLoader
from timesead.data.cses_dataset import CsesDataset
from timesead.models.baselines.iforest import IForestAD
import matplotlib.pyplot as plot
import numpy
import os

def main():
    cses_train_ds = CsesDataset()
    #cses_test_ds = CsesDataset(training=False)

    cses_train_dl = DataLoader(cses_train_ds, num_workers=0)
    # cses_test_dl = DataLoader(cses_test_ds, num_workers=1)

    forest = IForestAD(n_trees=100)
    forest.model.contamination = 0.5
    forest.fit(cses_train_dl)

    plots_dir = 'Plots_0.3'
    if not os.path.exists(plots_dir):
        os.makedirs(plots_dir)

    for item, idx in cses_train_dl:
        scores = forest.compute_online_anomaly_score(item)
        scores = scores.numpy()

        #series = forest.format_online_targets(item)

        #appiattisci le dimensioni nel dataset e trattalo come il caso di solo E_X ma con 3 features
        plot.figure(figsize=(12, 6))
        plot.plot(item[0, :, 0, 0].numpy(), label='E_normalized', color='blue', alpha=0.6)
        plot.axhline(y=forest.model.threshold_, color='red', linestyle='--', label='Threshold')
        
        anomalies = numpy.where(scores > forest.model.threshold_)[0]
        plot.scatter(anomalies, item[0, anomalies, 0, 0].numpy(), color='red', label='Anomalies', s=15)

        plot.title('Test')
        plot.xlabel('Time')
        plot.ylabel('E')

        plot.savefig(os.path.join(plots_dir, f'{idx + 1}.png'))
        plot.close()

if __name__ == '__main__':
    main()
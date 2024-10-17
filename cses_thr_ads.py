from torch.utils.data import DataLoader
from timesead.data.cses_dataset_polar import CsesDataset
from timesead.models.baselines.threshold_ads import OOSAnomalyDetector,IQRAnomalyDetector
import matplotlib.pyplot as plot
import numpy
import os

def main():
    cses_train_ds = CsesDataset()
    #cses_test_ds = CsesDataset(training=False)

    cses_train_dl = DataLoader(cses_train_ds, num_workers=0)
    # cses_test_dl = DataLoader(cses_test_ds, num_workers=1)

    #thrasd = OOSAnomalyDetector() #thrX = 2.0 thrXYZ = 3.0 thrpol = 0.05
    thrasd = IQRAnomalyDetector() #thrX = 3.1  thrXYZ = 5.0 thrpol = 0.1
    thrasd.fit(cses_train_dl)

    plots_dir = 'Plots_IQR_polar'
    if not os.path.exists(plots_dir):
        os.makedirs(plots_dir)

    i = 0

    for item, idx in cses_train_dl:
        scores = thrasd.compute_online_anomaly_score(item)
        scores = scores.numpy()

        print(scores)

        threshold = 2.32

        plot.figure(figsize=(12, 6))
        plot.plot(item[0, :, 0, 0], label='E_X_normalized', color='blue', alpha=0.6)
        plot.axhline(y=threshold, color='red', linestyle='--', label='Threshold')
        
        anomalies = numpy.where(scores > threshold)
        plot.scatter(anomalies, item[0, anomalies, 0, 0], color='red', label='Anomalies', s=15)

        plot.title('Test')
        plot.xlabel('Time')
        plot.ylabel('E_X_res')

        plot.savefig(os.path.join(plots_dir, f'{i}.png'))
        plot.close()
        i += 1

if __name__ == '__main__':
    main()
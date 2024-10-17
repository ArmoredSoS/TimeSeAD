from torch.utils.data import DataLoader
from timesead.data.cses_dataset_polar import CsesDataset
from timesead.models.baselines.iforest import IForestAD
import matplotlib.pyplot as plot
import os
import numpy

#Modified source files to remove deprecated np.float32

#removed useless imports from __init__ files to allow easy execution

def main():
    cses_train_ds = CsesDataset()
    cses_train_dl = DataLoader(cses_train_ds)

    forest = IForestAD()
    forest.model.contamination = 0.4
    forest.fit(cses_train_dl)

    #cses_test_ds = CsesDataset(training=False)
    #cses_test_dl = DataLoader(cses_test_ds)

    plots_dir = 'Plots_iforest_polar'
    if not os.path.exists(plots_dir):
        os.makedirs(plots_dir)

    i = 0

    for item, _ in cses_train_dl:
        scores = forest.compute_online_anomaly_score(item)
        scores = scores.numpy()
        print("plotting")

        plot.figure(figsize=(12, 6))
        plot.plot(item[0, :, 0, 0].numpy(), label='E_X_normalized', color='blue', alpha=0.6)
        plot.axhline(y=forest.model.threshold_, color='red', linestyle='--', label='Threshold')
        
        anomalies = numpy.where(scores > forest.model.threshold_)[0]
        plot.scatter(anomalies, item[0, anomalies, 0, 0].numpy(), color='red', label='Anomalies', s=15)

        plot.title('Test')
        plot.xlabel('Time')
        plot.ylabel('E_X_res')

        plot.savefig(os.path.join(plots_dir, f'{i}.png'))
        plot.close() 
        i += 1

if __name__ == '__main__':
    main()
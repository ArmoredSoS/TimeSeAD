from torch.utils.data import DataLoader
from timesead.data.cses_dataset_X import CsesDataset
from timesead.models.baselines.iforest import IForestAD
import matplotlib.pyplot as plot
import os
import numpy

#Modified source files to remove deprecated np.float32

#removed useless imports from __init__ files to allow easy execution

def main():
    cses_train_ds = CsesDataset()
    cses_test_ds = CsesDataset(training=False)

    cses_train_dl = DataLoader(cses_train_ds)
    cses_test_dl = DataLoader(cses_test_ds)

    forest = IForestAD(n_trees=100, max_samples=256, max_features=0.5, bootstrap=False, input_shape= 'btf')

    forest.model.contamination = 0.5
    forest.fit(cses_train_dl)

    plots_dir = 'Plots'
    if not os.path.exists(plots_dir):
        os.makedirs(plots_dir)

    #all_scores = []
    #series = []

    for item, idx in cses_test_dl:
        scores = forest.compute_online_anomaly_score(item)
        scores = scores.numpy()
        #all_scores.extend(scores.numpy())
        series = item.numpy().squeeze()

        #series.extend(item.numpy())
        #series = numpy.array(series).squeeze()

        plot.figure(figsize=(12, 6))
        plot.plot(series.flatten(), label='E_X_res', color='blue', alpha=0.6)
        plot.axhline(y=forest.model.threshold_, color='red', linestyle='--', label='Threshold')
        
        anomalies = numpy.where(scores > forest.model.threshold_)[0]
        plot.scatter(anomalies, series.flatten()[anomalies], color='red', label='Anomalies')

        plot.title('Test')
        plot.xlabel('Time')
        plot.ylabel('E_X_res')

        plot.savefig(os.path.join(plots_dir, f'{idx + 1}.png'))
        plot.close() 

    #plt.figure(figsize=(10, 6))
    #sns.histplot(scores, bins=50, kde=True)
    #plt.title(f'Threshold: {forest.model.threshold_}')
    #plt.show()

if __name__ == '__main__':
    main()
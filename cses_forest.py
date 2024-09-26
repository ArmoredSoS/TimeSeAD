from torch.utils.data import DataLoader
from timesead.data.cses_dataset import CsesDataset
from timesead.models.baselines.iforest import IForestAD
import timesead.plots.dataset_plots as pl
import torch


def main():
    cses_train_ds = CsesDataset()
    cses_test_ds = CsesDataset(training=False)

    cses_train_dl = DataLoader(cses_train_ds, num_workers=1)
    cses_test_dl = DataLoader(cses_test_ds, num_workers=1)

    forest = IForestAD(n_trees=100, max_samples=256, max_features=1.0, bootstrap=False)

    print(len(cses_train_dl))

    forest.fit(cses_train_dl)

    tot_scores = []
    for item,_ in cses_train_dl:
        scores = forest.compute_online_anomaly_score(item)
        tot_scores.extend(scores.numpy())

    print(tot_scores)

    #Plots
    # save_path = 'C:\\Users\\Jacopo\\Documents\\Documenti\\Tesi\\Model_testing\\Plots_results'
    # pl.plot_features_against_anomaly(dataset=cses_train_dl.dataset, path=save_path, shape='tf', interval_size=100)
    # pl.plot_features_against_anomaly(dataset=cses_test_dl.dataset, path=save_path, shape='tf', interval_size=100)

    # pl.plot_anomaly_position_distribution(dataset=cses_train_dl.dataset, path=save_path)
    # pl.plot_anomaly_position_distribution(dataset=cses_test_dl.dataset, path=save_path)

    # pl.plot_anomaly_length_distribution(dataset=cses_train_dl.dataset, path=save_path)
    # pl.plot_anomaly_length_distribution(dataset=cses_test_dl.dataset, path=save_path)

    # pl.plot_mean_distribution(dataset=cses_train_dl.dataset, path=save_path)
    # pl.plot_mean_distribution(dataset=cses_test_dl.dataset, path=save_path)

if __name__ == '__main__':
    main()
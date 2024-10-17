from timesead.utils.metadata import DATA_DIRECTORY
import matplotlib.pyplot as plot

import h5py
import os
import numpy

def load_data():

    E_X = []
    path = os.path.join(DATA_DIRECTORY, 'CSES\\training')

    for file in os.listdir(path):

        cur_file = os.path.join(path, file)

        try:
            with h5py.File(cur_file, 'r') as current_file:

                E_X_temp = numpy.array(current_file['A111_W'][:], dtype=float)

                if E_X_temp.shape[0] < 1000:
                    E_X_temp = numpy.pad(E_X_temp, ((0, 1000 - E_X_temp.shape[0]), (0, 256 - E_X_temp.shape[1])), mode='constant', constant_values=0)
                    
                E_X_temp = E_X_temp.flatten()
                E_X.append(numpy.stack([E_X_temp[:256000], E_X_temp[:256000], E_X_temp[:256000]], axis=1))
                            
        except Exception as e:
            print(e)
            

    E_X = numpy.array(E_X, dtype=float)
    E_X = (E_X - E_X.mean())/E_X.std()
    E_X = numpy.expand_dims(E_X, 3)
    print(E_X.shape)

    return E_X
    

if __name__ == '__main__':
    
    E = load_data()
    E = E[:, :, 0, 0]

    plots_dir = 'Plots_posteriori_X_2'
    if not os.path.exists(plots_dir):
        os.makedirs(plots_dir)

    
    for item in range(len(E)):
        anomalies = numpy.zeros(E.shape[1])
        for data_point in range(len(E[item])):
            for i in range(100):
                if (abs(abs(E[item, data_point]) - abs(E[item, data_point - i])) > 0.015 and data_point >= 100):
                    anomalies[data_point] = 1

        plot.figure(figsize=(12, 6))
        plot.plot(E[item, :], label='E_normalized', color='blue', alpha=0.6)

        anomalies_graph = numpy.where(anomalies > 0)
        plot.scatter(anomalies_graph, E[item, anomalies_graph], color='red', label='Anomalies', s=10)

        plot.title('Test')
        plot.xlabel('Time')
        plot.ylabel('E')

        plot.savefig(os.path.join(plots_dir, f'{item}.png'))
        plot.close()
                
    
                

    
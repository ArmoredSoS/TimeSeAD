import torch.utils
import torch.utils.data
import torch.utils.data.dataloader
from .dataset import BaseTSDataset
from typing import Tuple, Union, Any, Dict, List
from timesead.utils.metadata import DATA_DIRECTORY
from sklearn.preprocessing import StandardScaler

import torch
import h5py
import os
import numpy

class CsesDataset(BaseTSDataset):

    def __init__(self, path: str = os.path.join(DATA_DIRECTORY, 'CSES'), training: bool = True):
        
        if not os.path.exists(path):
            raise NotADirectoryError

        if not os.path.exists(os.path.join(path, 'training')):
            raise NotADirectoryError
        train_dir = os.path.join(path, 'training')

        if not os.path.exists(os.path.join(path, 'test')):
            raise NotADirectoryError
        test_dir = os.path.join(path, 'test')

        self.work_files = []
        file_list = os.listdir(train_dir) if training else os.listdir(test_dir)
        work_dir = train_dir if training else test_dir
        for file in file_list:
            current_file = os.path.join(work_dir, file)
            self.work_files.append(current_file)

        self.training = training 
        self.E_X = self.load_data()

    def load_data(self):

        E_X = []

        for file in self.work_files:
            try:
                with h5py.File(file, 'r') as current_file:

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
    
    def __len__(self) -> int:
        return self.E_X.shape[0]
            
    @property
    def seq_len(self) -> Union[int, List[int]]:
        return self.E_X.shape[1]
    
    @property   
    def num_features(self) -> Union[int, Tuple[int, ...]]:
        return self.E_X.shape[2]
    
    @staticmethod
    def get_default_pipeline() -> Dict[str, Dict[str, Any]]:
        return {
            'identity': {
                'class': 'NoTransform',
                'args': {}
            }
        }
    
    @staticmethod
    def get_feature_names() -> List[str]:
        return ['E_X']
    
    def __getitem__(self, index: int) -> Tuple[Tuple[torch.Tensor, ...], Tuple[torch.Tensor, ...]]:
        dummy_targets = numpy.zeros((self.E_X.shape[1], self.E_X.shape[2], 1))  
        return torch.as_tensor(self.E_X[index]), torch.as_tensor(dummy_targets)
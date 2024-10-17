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
        self.E = self.load_data()

    def load_data(self):
        
        E = []

        for file in self.work_files:
            try:
                with h5py.File(file, 'r') as current_file:

                    E_temp = []

                    E_X_temp = numpy.array(current_file['A111_W'][:], dtype=int)
                    E_Y_temp = numpy.array(current_file['A112_W'][:], dtype=int)
                    E_Z_temp = numpy.array(current_file['A113_W'][:], dtype=int)    
                    
                    if E_X_temp.shape[0] < 1000:
                        E_X_temp = numpy.pad(E_X_temp, ((0, 1000 - E_X_temp.shape[0]), (0, 256 - E_X_temp.shape[1])), mode='constant', constant_values=0)
                    if E_Y_temp.shape[0] < 1000:
                        E_Y_temp = numpy.pad(E_Y_temp, ((0, 1000 - E_Y_temp.shape[0]), (0, 256 - E_Y_temp.shape[1])), mode='constant', constant_values=0)
                    if E_Z_temp.shape[0] < 1000:
                        E_Z_temp = numpy.pad(E_Z_temp, ((0, 1000 - E_Z_temp.shape[0]), (0, 256 - E_Z_temp.shape[1])), mode='constant', constant_values=0)

                    E_temp = numpy.stack([E_X_temp[:1000, :256], E_Y_temp[:1000, :256], E_Z_temp[:1000, :256]], axis=1)
                    E.append(E_temp)
                            
            except Exception as e:
                print(e)

        E = numpy.array(E, dtype=int)
        E = (E - E.mean())/E.std()

        return E
    
    def __len__(self) -> int:
        return self.E.shape[0]
            
    @property
    def seq_len(self) -> Union[int, List[int]]:
        return self.E.shape[1]
    
    @property   
    def num_features(self) -> Union[int, Tuple[int, ...]]:
        return self.E.shape[2]
    
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
        return ['E_X', 'E_Y', 'E_Z']
    
    def __getitem__(self, index: int) -> Tuple[Tuple[torch.Tensor, ...], Tuple[torch.Tensor, ...]]:
        dummy_targets = numpy.zeros((self.E.shape[1], self.E.shape[2], 1))
        return torch.as_tensor(self.E[index]), torch.as_tensor(dummy_targets)
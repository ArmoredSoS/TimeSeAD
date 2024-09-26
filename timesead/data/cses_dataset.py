import torch.utils
import torch.utils.data
import torch.utils.data.dataloader
from .dataset import BaseTSDataset
from typing import Tuple, Union, Any, Dict, List
from timesead.utils.metadata import DATA_DIRECTORY

import torch
import h5py
import pandas
import os
import numpy

class CsesDataset(BaseTSDataset):

    def __init__(self, path: str = os.path.join(DATA_DIRECTORY, 'CSES'), training: bool = True):
        
        if not os.path.exists(path):
            raise NotADirectoryError
        self.path = path

        if not os.path.exists(os.path.join(self.path, 'training')):
            raise NotADirectoryError
        self.train_dir = os.path.join(self.path, 'training')

        if not os.path.exists(os.path.join(self.path, 'test')):
            raise NotADirectoryError
        self.test_dir = os.path.join(self.path, 'test')

        self.training = training
        self.ts_length = []
        self.ds_train = numpy.zeros((1003, 256, 3))
        self.ds_test = numpy.zeros((1003, 256, 3))

    def load_data(self):

        if self.training:

            for file in os.listdir(self.train_dir):

                E_X = []
                E_Y = []
                E_Z = []
                #time = []

                try:
                    with h5py.File(os.path.join(self.train_dir, file), 'r') as current_file:
                        E_X = current_file['A111_W'][:]
                        E_Y = current_file['A112_W'][:]
                        E_Z = current_file['A113_W'][:]
                        #time = current_file['VERSE_TIME'][:]
                except Exception as e:
                    print(e)

                self.ts_length.append(len(E_X))
                #time = numpy.repeat(numpy.array(time, dtype=int).flatten(), 256)
                E_X = numpy.array(E_X, dtype=int)
                E_Y = numpy.array(E_Y, dtype=int)
                E_Z = numpy.array(E_Z, dtype=int)
                self.ds_train.append(numpy.stack(E_X, E_Y, E_Z), axis = -1)

            return self.ds_train
        
        for file in os.listdir(self.test_dir):

            E_X = []
            E_Y = []
            E_Z = []
            #time = []

            try:
                with h5py.File(os.path.join(self.test_dir, file), 'r') as current_file:
                    E_X = current_file['A111_W'][:]
                    E_Y = current_file['A112_W'][:]
                    E_Z = current_file['A113_W'][:]
                    #time = current_file['VERSE_TIME'][:]
            except Exception as e:
                    print(e)

            self.ts_length.append(len(E_X))
            #time = numpy.array(time, dtype=int)
            E_X = (numpy.array(E_X, dtype=int)).flatten()
            E_Y = (numpy.array(E_Y, dtype=int)).flatten()
            E_Z = (numpy.array(E_Z, dtype=int)).flatten()
            self.ds_test.append(numpy.stack(E_X, E_Y, E_Z), axis = -1)

        return self.ds_test
    
    def __len__(self) -> int:

        if self.training:
            return len(self.ds_train)
        return len(self.ds_test)
            
    @property
    def seq_len(self) -> Union[int, List[int]]:
        return self.ts_length
    
    @property   
    def num_features(self) -> Union[int, Tuple[int, ...]]:
        return 3
    
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

        if not (0 <= index < len(self)):
            raise KeyError('Out of bounds')

        if self.ds_test is None or self.ds_train is None:
            self.load_data()

        #Reshape to have 3 dimensions, as the fit method requires
        if not self.training:
            return torch.as_tensor(self.ds_test), index

        return torch.as_tensor(self.ds_train), index
    
    
    
    
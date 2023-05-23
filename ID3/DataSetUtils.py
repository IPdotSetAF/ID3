from pickle import FALSE
from pandas import read_csv
import numpy as np


class CSV_Loader:
    def __init__(self, fileName, verbose = 1):
        rawData = read_csv(fileName)
        if verbose:
            print(f'\nraw data : \n\n{rawData}')

        self.features = np.array(rawData.columns)[:-1]
        #print(f'\ncolumns are : {columns}')
    
        self.data = np.array(rawData)[:,:-1]
        #print(f'\nfeatures are : \n{data}')

        self.targets = np.array(rawData)[:,-1]
        #print(f'\ntarget is : {target}')
    
def train_test_dataSplit(data, target, ratio, random = FALSE):

    dataCount = len(data)

    #if random:
    #    pass
    #else:
    splitPoint =int(dataCount * ratio)
    train_d = data[0:splitPoint, : ]
    test_d = data[splitPoint:dataCount, : ]
    train_t = target[0:splitPoint]
    test_t = target[splitPoint:dataCount]

    return train_d, train_t, test_d, test_t
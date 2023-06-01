from pickle import FALSE
from pandas import read_csv
import numpy as np


class CSV_Loader:
    def __init__(self, fileName, verbose = 1):
        rawData = read_csv(fileName)

        self.features = np.array(rawData.columns)[:-1]
        #print(f'\ncolumns are : {columns}')
    
        self.data = np.array(rawData)[:,:-1]
        #print(f'\nfeatures are : \n{data}')

        self.targets = np.array(rawData)[:,-1]
        #print(f'\ntarget is : {target}')
        
        if verbose==1:
            print(f'\nraw data : \n\n{rawData}')
        elif verbose==2:
            uniques = self.unique()
            for i in range(len(rawData.columns)):
                print(f'{rawData.columns[i]} : {uniques[i]}')
        
    def convert(self, converters):
        if not len(converters) == (len(self.data[0]) + 1):
            raise InvalidShape(f'expected converter len={len(self.data[0]) + 1} but recieved len={len(converters)}')
        
        for i in range(len(self.data[0])):
            column = [converters[i][d] for d in self.data[:,i]]
            self.data[:,i] = column
            
        self.targets = [converters[-1][t] for t in self.targets]
            
    def unique(self):
        return [*[list(np.unique(self.data[:,i])) for i in range(len(self.data[0]))], list(np.unique(self.targets))]
      
class InvalidShape(Exception):
    pass
    
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
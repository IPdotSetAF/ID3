import numpy as np

class ID3:
    _Name = ''
    @property 
    def Name(self):
        return self._Name
    
    _Keys = []
    @property 
    def Keys(self):
        return self._Keys
    
    _Values = []
    @property 
    def Values(self):
        return self._Values
    
    __TotalEntropy = 0
    @property 
    def Entropy(self):
        return self.__TotalEntropy
        
    def __str__(self):
        return f'ID3{"{"}{self._Name},{self._Keys},{self._Values}{"}"}'
    
    def __repr__(self):
        return f'{self}'
    
    def _badCalssified():
       return 'ID3.badClassified'
    
    def __init__(self, id3 = 0):
        if(id3):
            for attr_name in id3.__dict__:
                setattr(self, attr_name, getattr(id3, attr_name))

            for i in range(len(self._Values)):
                if (isinstance(self._Values[i] , ID3)):
                    self._Values[i] = ID3(self._Values[i])

    def _calculate_entropy(self, target):
        totalEntries = len(target)
        entropy = 0
        for t in np.unique(target):
            targetCount = np.count_nonzero(target == t)
            targetProbability = targetCount/totalEntries
            entropy -= targetProbability * np.log2(targetProbability)
        return entropy
    
    def _slice_data_by_feature_index(self, data, featureIndex):
        featureData = np.array(data)[:,featureIndex]
        featurePossibleValues = np.unique(featureData)
        return featureData , featurePossibleValues
    
    def _calculate_best_feature(self, entropy, features, data, target):
        pass
    
    def _instantiate(self):
        pass
    
    def fit(self, features, data , target):
        self.__TotalEntropy = self._calculate_entropy(target)
        self._Name, featureIndex = self._calculate_best_feature( self.__TotalEntropy, features, data, target)
        
        featureData ,featurePossibleValues = self._slice_data_by_feature_index(data, featureIndex)
        self._Keys = featurePossibleValues
        
        newFeatures = np.delete(features, featureIndex, 0)
        newData = np.delete(data, featureIndex, 1)
        tmpValues = list([])
        for value in self._Keys:
            valueIndecies = np.where(featureData == value)
            newTarget = np.array(target)[valueIndecies]
            pTargets = np.unique(newTarget)
            if(len(pTargets) == 1):
                tmpValues.append(pTargets[0])
            else:
                tmpValues.append(self._instantiate())
                tmpValues[-1].fit(newFeatures, np.array(newData)[valueIndecies], newTarget)
        
        self._Values = np.array(tmpValues)
    
    def predict(self, features, data):
        index = np.where(features == self._Name)[0]
        value = self._Values[np.where(data[index] == self._Keys)]
        
        if (len(value) == 0):
            return type(self)._badCalssified()
        
        value = value[0]
        if isinstance(value , ID3):
            return value.predict(features, data)
        else:
            return value
        
    def score(self, features, x_test, y_text, verbose = 1):
        correct = 0 
        for i in range(len(x_test)):
            result = self.predict(features, x_test[i])
            if verbose:
                print(f'{x_test[i]}, {result}')
            if (result == y_text[i]):
                correct += 1
        return correct/len(x_test)

    def draw(self):
        from .Drawer import Drawer

        Drawer(self).Draw()
        
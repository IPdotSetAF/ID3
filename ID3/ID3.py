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
    
    def _CalculateEntropy(self, target):
        totalEntries = len(target)
        entropy = 0
        for t in np.unique(target):
            targetCount = np.count_nonzero(target == t)
            targetProbability = targetCount/totalEntries
            entropy -= targetProbability * np.log2(targetProbability)
        return entropy
    
    def _SliceDataByFeatureIndex(self, data, featureIndex):
        featureData = np.array(data)[:,featureIndex]
        featurePossibleValues = np.unique(featureData)
        return featureData , featurePossibleValues
    
    def _CalculateBestFeature(self, entropy, features, data, target):
        pass
    
    def _Instantiate(self):
        pass
    
    def Train(self, features, data , target):
        self.__TotalEntropy = self._CalculateEntropy(target)
        self._Name, featureIndex = self._CalculateBestFeature( self.__TotalEntropy, features, data, target)
        
        featureData ,featurePossibleValues = self._SliceDataByFeatureIndex(data, featureIndex)
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
                tmpValues.append(self._Instantiate())
                tmpValues[-1].Train(newFeatures, np.array(newData)[valueIndecies], newTarget)
        
        self._Values = np.array(tmpValues)
    
    def Resolve(self, features, data):
        index = np.where(features == self._Name)[0]
        value = self._Values[np.where(data[index] == self._Keys)]
        value = value[0]
        if isinstance(value , ID3):
            return value.Resolve(features, data)
        else:
            return value
        
class IG_ID3(ID3):
    
    def __init__(self):
        super()
        
    def _Instantiate(self):
        return IG_ID3()
    
    def _CalculateBestFeature(self, entropy, features, data, target):
        gains = np.zeros((len(features)), dtype=float)
        for featureIndex in range(len(features)):
            featureData ,featurePossibleValues = self._SliceDataByFeatureIndex(data, featureIndex)
            gains[featureIndex] = entropy
            for c in featurePossibleValues:
                valueIndecies = np.where(featureData == c)
                gains[featureIndex] -= (np.count_nonzero(featureData == c)/ len(data)) * self._CalculateEntropy(np.array(target)[valueIndecies])
        
#         print(features)
#         print(gains)
        featureIndex = gains.argmax()
        return features[featureIndex], featureIndex
    
class GR_ID3(ID3):
    
    def __init__(self):
        super()
        
    def _Instantiate(self):
        return GR_ID3()
    
    def _CalculateBestFeature(self, entropy, features, data, target):
        split = np.zeros((len(features)), dtype=float)
        for featureIndex in range(len(features)):
            featureData ,featurePossibleValues = self._SliceDataByFeatureIndex(data, featureIndex)
            for c in featurePossibleValues:
                valueIndecies = np.where(featureData == c)
                split[featureIndex] -= (np.count_nonzero(featureData == c)/ len(data)) * self._CalculateEntropy(np.array(target)[valueIndecies])
            split[featureIndex] = (entropy + split[featureIndex])/split[featureIndex]
#         print(features)
#         print(gains)
        featureIndex = split.argmax()
        return features[featureIndex], featureIndex
        
class GI_ID3(ID3):
    
    def __init__(self):
        super()
        
    def _Instantiate(self):
        return GI_ID3()
    
    def __CalculateGINI(self, target):
        totalEntries = len(target)
        gini = 1
        for t in np.unique(target):
            targetCount = np.count_nonzero(target == t)
            targetProbability = targetCount/totalEntries
            gini -= targetProbability * targetProbability
        return gini
    
    def _CalculateBestFeature(self, entropy, features, data, target):
        ginis = np.zeros((len(features)), dtype=float)
        for featureIndex in range(len(features)):
            featureData ,featurePossibleValues = self._SliceDataByFeatureIndex(data, featureIndex)
            for c in featurePossibleValues:
                valueIndecies = np.where(featureData == c)
                ginis[featureIndex] += (np.count_nonzero(featureData == c)/ len(data)) * self.__CalculateGINI(np.array(target)[valueIndecies])
#         print(features)
#         print(ginis)
        featureIndex = ginis.argmin()
        return features[featureIndex], featureIndex
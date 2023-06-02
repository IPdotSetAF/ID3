from .ID3 import ID3
import numpy as np

class GiniIndex_ID3(ID3):
    
    def __init__(self):
        super()
        
    def _instantiate(self):
        return GiniIndex_ID3()
    
    def __CalculateGINI(self, target):
        totalEntries = len(target)
        gini = 1
        for t in np.unique(target):
            targetCount = np.count_nonzero(target == t)
            targetProbability = targetCount/totalEntries
            gini -= targetProbability * targetProbability
        return gini
    
    def _calculate_best_feature(self, entropy, features, data, target):
        ginis = np.zeros((len(features)), dtype=float)
        for featureIndex in range(len(features)):
            featureData ,featurePossibleValues = self._slice_data_by_feature_index(data, featureIndex)
            for c in featurePossibleValues:
                valueIndecies = np.where(featureData == c)
                ginis[featureIndex] += (np.count_nonzero(featureData == c)/ len(data)) * self.__CalculateGINI(np.array(target)[valueIndecies])
#         print(features)
#         print(ginis)
        featureIndex = ginis.argmin()
        return features[featureIndex], featureIndex
from .ID3 import ID3
import numpy as np

class GainRatio_ID3(ID3):
    
    def __init__(self):
        super()
        
    def _instantiate(self):
        return GainRatio_ID3()
    
    def _calculate_best_feature(self, entropy, features, data, target):
        split = np.zeros((len(features)), dtype=float)
        for featureIndex in range(len(features)):
            featureData ,featurePossibleValues = self._slice_data_by_feature_index(data, featureIndex)
            for c in featurePossibleValues:
                valueIndecies = np.where(featureData == c)
                split[featureIndex] -= (np.count_nonzero(featureData == c)/ len(data)) * self._calculate_entropy(np.array(target)[valueIndecies])
            split[featureIndex] = (entropy + split[featureIndex])/split[featureIndex]
#         print(features)
#         print(gains)
        featureIndex = split.argmax()
        return features[featureIndex], featureIndex
        
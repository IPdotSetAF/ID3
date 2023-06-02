from .ID3 import ID3
import numpy as np

class InfoGain_ID3(ID3):
    
    def __init__(self):
        super()
        
    def _instantiate(self):
        return InfoGain_ID3()
    
    def _calculate_best_feature(self, entropy, features, data, target):
        gains = np.zeros((len(features)), dtype=float)
        for featureIndex in range(len(features)):
            featureData ,featurePossibleValues = self._slice_data_by_feature_index(data, featureIndex)
            gains[featureIndex] = entropy
            for c in featurePossibleValues:
                valueIndecies = np.where(featureData == c)
                gains[featureIndex] -= (np.count_nonzero(featureData == c)/ len(data)) * self._calculate_entropy(np.array(target)[valueIndecies])
        
#         print(features)
#         print(gains)
        featureIndex = gains.argmax()
        return features[featureIndex], featureIndex
    
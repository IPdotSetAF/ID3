from .InfoGain_ID3 import InfoGain_ID3 
from .GainRatio_ID3 import GainRatio_ID3 
from .GiniIndex_ID3 import GiniIndex_ID3 
from enum import Enum

class Algorithms(Enum) :
    INFO_GAIN = 0
    GAIN_RATIO = 1
    GINI_INDEX = 2

class Trainer :
    @property
    def ID3 (self):
        return self.__ID3

    def __init__(self, featurs, data, target, id3_Algorythm):
        if(id3_Algorythm == Algorithms.INFO_GAIN):
            self.__ID3 = InfoGain_ID3()
        elif(id3_Algorythm == Algorithms.GAIN_RATIO):
            self.__ID3 = GainRatio_ID3()
        elif(id3_Algorythm == Algorithms.GINI_INDEX):
            self.__ID3 = GiniIndex_ID3()
        else :
            print('invalid Algorythm')
            return
        
        self.__ID3.fit(featurs, data , target)

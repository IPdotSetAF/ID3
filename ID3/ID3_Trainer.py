from .ID3 import IG_ID3, GR_ID3, GI_ID3
from enum import Enum

class ID3_Algorythm(Enum) :
    INFO_GAIN = 0
    GAIN_RATIO = 1
    GINI_INDEX = 2

class ID3_Trainer :
    @property
    def ID3 (self):
        return self.__ID3

    def __init__(self, featurs, data, target, id3_Algorythm):
        if(id3_Algorythm == ID3_Algorythm.INFO_GAIN):
            self.__ID3 = IG_ID3()
        elif(id3_Algorythm == ID3_Algorythm.GAIN_RATIO):
            self.__ID3 = GR_ID3()
        elif(id3_Algorythm == ID3_Algorythm.GINI_INDEX):
            self.__ID3 = GI_ID3()
        else :
            print('invalid Algorythm')
            return
        
        self.__ID3.Train(featurs, data , target)

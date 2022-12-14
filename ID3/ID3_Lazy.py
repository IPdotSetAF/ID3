from ID3 import ID3 
import numpy as np

class ID3_Lazy(ID3):
    def __init__(self, id3):
        super()

        for attr_name in id3.__dict__:
            setattr(self, attr_name, getattr(id3, attr_name))

        for i in range(len(self._Values)):
            if (isinstance(self._Values[i] , ID3)):
                self._Values[i] = ID3_Lazy(self._Values[i])


    __badCalssified = 'ID3.badClassified'

    def Resolve(self, features, data, target):
        index = np.where(features == self._Name)[0]
        value = self._Values[np.where(data[index] == self._Keys)]
        
        if (len(value) == 0):
            value = self._Values[np.where('?' == self._Keys)]

        if (len(value) == 0):
            self.AddNewKeyValue(data[index] ,target)
            return ID3_Lazy.__badCalssified

        value = value[0]
        if isinstance(value , ID3):
            return value.Resolve(np.delete(features, [index]), np.delete(data, [index]), target)
        else:
            if (value == target):
                return value
            else:
                id3 = ID3()
                id3._Name = np.delete(features,[index])[0]
                id3._Keys = np.array([np.delete(data,[index])[0], '?'])
                id3._Values = np.array([target, value])
                id3 = ID3_Lazy(id3)

                oldIndex = np.where(data[index] == self._Keys)[0]
                self._Keys = np.delete(self._Keys, oldIndex)
                self._Values = np.delete(self._Values, oldIndex)
                self.AddNewKeyValue( data[index] ,id3)

                return ID3_Lazy.__badCalssified
    
    def AddNewKeyValue(self, key ,target):
        self._Keys = np.append(self._Keys , [key])
        self._Values = np.append(self._Values , [target])

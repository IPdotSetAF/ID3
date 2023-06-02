import sys
sys.path.append('../')

from ID3 import CSV_Loader , train_test_dataSplit
from ID3 import ID3_Trainer ,ID3_Algorythm
from ID3 import ID3_Drawer
from ID3 import ID3_Lazy
from ID3 import ID3

_dataSetFileName = 'PlayTennis.csv'

_dataSet = CSV_Loader(_dataSetFileName)
_features = _dataSet.features
_data = _dataSet.data
_targets = _dataSet.targets

_trainData ,_trainTargets , _testData, _testTargets = train_test_dataSplit(_data, _targets, 0.5)


trainer = ID3_Trainer(_features, _trainData, _trainTargets, ID3_Algorythm.INFO_GAIN)
#trainer = ID3_Trainer(_features, _trainData, _trainTargets, ID3_Algorythm.GAIN_RATIO)
#trainer = ID3_Trainer(_features, _trainData, _trainTargets, ID3_Algorythm.GINI_INDEX)

id3 = trainer.ID3

print('Tree :\n')
ID3_Drawer(id3).Draw()


print(id3.predict(_features, _testData[0]))

print(id3.score(_features, _testData, _testTargets))

id3 = ID3_Lazy(id3)

print(id3.score(_features, _testData, _testTargets))

ID3_Drawer(id3).Draw()

print(id3.score(_features, _testData, _testTargets))
    

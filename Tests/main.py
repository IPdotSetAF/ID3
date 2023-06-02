import sys
from tabnanny import verbose
sys.path.append('../')

from ID3 import CSV_Loader , train_test_dataSplit
from ID3 import Trainer ,Algorithms
from ID3 import Lazy_ID3
from ID3 import ID3

_dataSetFileName = 'PlayTennis.csv'

_dataSet = CSV_Loader(_dataSetFileName)
_features = _dataSet.features
_data = _dataSet.data
_targets = _dataSet.targets

_trainData ,_trainTargets , _testData, _testTargets = train_test_dataSplit(_data, _targets, 0.5)


trainer = Trainer(_features, _trainData, _trainTargets, Algorithms.INFO_GAIN)
#trainer = Trainer(_features, _trainData, _trainTargets, Algorithms.GAIN_RATIO)
#trainer = Trainer(_features, _trainData, _trainTargets, Algorithms.GINI_INDEX)

id3 = trainer.ID3

print('Tree :\n')
id3.draw()
id3.generalize()
id3.draw()

print(id3.predict(_features, _testData[0]))

print(id3.score(_features, _testData, _testTargets, verbose=1))

id3 = Lazy_ID3(id3)

print(id3.score(_features, _testData, _testTargets, verbose=1))

id3.draw()

print(id3.score(_features, _testData, _testTargets, verbose=1))
    

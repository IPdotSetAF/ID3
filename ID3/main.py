from DataSetUtils import CSV_Loader , train_test_dataSplit
from ID3_Trainer import ID3_Trainer ,ID3_Algorythm
from ID3_Drawer import ID3_Drawer
from ID3_Lazy import ID3_Lazy

_dataSetFileName = 'PlayTennis.csv'

_dataSet = CSV_Loader(_dataSetFileName)
_features = _dataSet.features
_data = _dataSet.data
_targets = _dataSet.targets

_trainData ,_trainTargets , _testData, _testTargets = train_test_dataSplit(_data, _targets, 0.5)


# trainer = ID3_Trainer(_features, _trainData, _trainTargets, ID3_Algorythm.INFO_GAIN)
trainer = ID3_Trainer(_features, _trainData, _trainTargets, ID3_Algorythm.GAIN_RATIO)
# trainer = ID3_Trainer(_features, _trainData, _trainTargets, ID3_Algorythm.GINI_INDEX)

id3 = ID3_Lazy(trainer.ID3)

print('Tree :\n')
ID3_Drawer(id3).Draw()

#testing resolve
for i in range(len(_testData)):
    print(f'{_testData[i]} , {_testTargets[i]}')
    print(id3.Resolve(_features, _testData[i], _testTargets[i]))    
    #print(id3)
    ID3_Drawer(id3).Draw()
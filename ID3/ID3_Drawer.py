import numpy as np
from .ID3 import ID3

c = ["│","─","┌","┐","└","┘","├","┤","┬","┴",">","<"]
# c = ["|","-","┌","+","└","┘","+","┤","┬","┴",">","<"]

class ID3_Drawer :
    
    def __init__(self, ID3):
        self.__ID3 = ID3
        
    def Draw(self):
        for line in self.__Tree(self.__ID3):
            print(line)
    
    def __Tree(self, id3):
        if isinstance(id3, ID3):
            tree = list([f'[{id3.Name}]{c[1] * 2}{c[3]}'])
            spaceing = len(tree[0])-1
            lastTempLength = 0
            for i in range(len(id3.Keys)):
                if (lastTempLength >1):
                    tree.append(f'{spaceing * " "}{c[0]}')
                tree.append(f'{" "*spaceing}{c[6]}{c[1]}{id3.Keys[i]}{c[1]*2}{c[10]}')
                
                spacing2 = len(tree[-1])+1
                tmp = self.__Tree(id3.Values[i])
                lastTempLength = len(tmp)
                tree[-1] += f' {tmp[0]}'
                for j in range(1 ,len(tmp)):
                    if (i == len(id3.Keys)-1):
                        tree.append(f'{spacing2 * " "}{tmp[j]}')
                    else:
                        tree.append(f'{spaceing * " "}{c[0]}{(spacing2-spaceing-1) * " "}{tmp[j]}')
                
            return np.array(tree)
        else :
            return np.array([id3])    
 
# -*- coding: utf-8 -*-
"""
Created on Wed May 25 15:26:19 2016

@author: brudervn
"""

f = open('/home/brudervn/data/resultsTRRojan/memClock/titanX/VolumeRender_TITAN X (Pascal)_SyntheticBox_1byte_1350.txt')
i = 0
"""
for i, l in enumerate(f):
    pass
        
numEntries = i + 1
"""
#xStep = np.empty(numEntries)

default = [] 
buff = []
shuff = []
shuffbuff = []

medianDefault = 0.0
medianBuff = 0.0
medianShuff = 0.0
medianShuffBuff = 0.0

for i,l in enumerate(f):
    lst = l.split()
    if len(lst) == 0:
        continue
    if i%2 == 1 or i < 3:
        continue
    if i < (64*2)*1+3:
        default.append(float(lst[0]))
    if i < (64*2)*2+3:
        buff.append(float(lst[0]))
    if i < (64*2)*3+3:
        shuff.append(float(lst[0]))
    if i < (64*2)*4+3:
        shuffbuff.append(float(lst[0]))

default.sort()
buff.sort()
shuff.sort()
shuffbuff.sort()

medianDefault = default[len(default) / 2]
print( "default " + str(medianDefault) )
medianBuff = buff[len(buff) / 2]
print( "buff " + str(medianBuff) )
medianShuff = shuff[len(shuff) / 2]
print( "shuff " + str(medianShuff) )
medianShuffBuff = shuffbuff[len(shuffbuff) / 2]
print( "shuff buff " + str(medianShuffBuff) )



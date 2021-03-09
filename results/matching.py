import csv
from os import listdir
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import re

def sorted_alphanumeric(data):
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ] 
    return sorted(data, key=alphanum_key)

outputPath = 'matchingGraphs\\'

path = 'evaluation\\MatchingScene\\'
files = sorted_alphanumeric(listdir(path))

xvalues = np.arange(1,1001,1)
sceneNames = defaultdict(list)

for file in files:
    with open(path+file) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0
        detectorName = ''
        sceneName = ' '
        for row in csv_reader:
            if line_count == 0:
                detectorName = row[0]
            elif len(row) == 1:
                sceneName = row[0]
            else:
                del(row[len(row)-1])#remove last element (its a space)
                sceneNames[sceneName].append([detectorName,np.array(row).astype(np.float)])
            line_count +=1
        print(f'Processed {line_count} lines.')

for key in sceneNames:
    for detectorResult in sceneNames[key]:
        plt.plot(xvalues,detectorResult[1],label=detectorResult[0][:-5])
    plt.xlim(xvalues[0]-1,xvalues[len(xvalues)-1])
    plt.ylim(0,1)
    plt.legend()
    plt.title(key[:-4])
    plt.savefig(outputPath + key + '.png')
    plt.clf()

path = 'evaluation\\MatchingAverage\\'
files = sorted_alphanumeric(listdir(path))

mapValues = {}
xvalues = np.arange(1,1001,1)



for file in files:
    with open(path+file) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0
        detectorName = ''
        for row in csv_reader:
            if line_count == 0:
                detectorName = row[0]
            else:
                del(row[len(row)-1])#remove last element (its a space)
                mapValues[detectorName]=np.array(row).astype(np.float)
            line_count +=1
        print(f'Processed {line_count} lines.')

for key in mapValues:
    plt.plot(xvalues,mapValues[key],label=key[:-5])
plt.xlim(xvalues[0]-1,xvalues[len(xvalues)-1])
plt.ylim(0,1)
plt.legend()
plt.title("Average matching score")
plt.savefig(outputPath + 'average.png')
#plt.show()



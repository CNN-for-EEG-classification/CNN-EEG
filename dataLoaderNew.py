# -*- coding: utf-8 -*-
"""
Dataloader program that creates a Pytorch compatible tensor from a raw tab-separated txt file (EP1 MindBigData dataset).
"""
import csv
from numpy.core.numeric import full
import pandas as pd
import numpy as np
import torch


def selectAndAddChannel(eventgrp, name):
    name = str(name)
    channel = eventgrp[eventgrp['channel'] == name]['data'].values
    channel_np = np.fromstring(channel[0], dtype=float, sep=",")[:250]
    return eventgrp, channel_np



# P9, AF7, AF8, and TP10 in a differnet study
def createTensorGroup(eventgrp, transform=True, exclude=[]):
    list_of_channels = ['AF3', 'AF4', 'F7', 'F8', 'F3', 'F4', 'FC5', 'FC6', 'P7', 'P8', 'T7', 'T8', 'O1', 'O2']
    [AF3np, AF4np, F7np, F8np, F3np, F4np, FC5np, FC6np, P7np, P8np, T7np, T8np, O1np, O2np] =\
        [None, None, None, None,None, None, None, None, None, None, None, None, None, None]                                                                                     
    NoneType = type(AF3np)

    exclude = set(exclude)
    # fullChannel = np.empty((1,250))
    if 'AF3' not in exclude:
        eventgrp, AF3np = selectAndAddChannel(eventgrp, 'AF3')
    if 'F7' not in exclude:
        eventgrp, F7np = selectAndAddChannel(eventgrp, 'F7')
    if 'F3' not in exclude:
        eventgrp, F3np = selectAndAddChannel(eventgrp, 'F3')
    if 'FC5' not in exclude:
        eventgrp, FC5np = selectAndAddChannel(eventgrp, 'FC5')
    if 'T7' not in exclude:
        eventgrp, T7np = selectAndAddChannel(eventgrp, 'T7')
    if 'P7' not in exclude:
        eventgrp, P7np = selectAndAddChannel(eventgrp, 'P7')
    if 'T7' not in exclude:
        eventgrp, T7np = selectAndAddChannel(eventgrp, 'T7')
    if 'AF4' not in exclude:
        eventgrp, AF4np = selectAndAddChannel(eventgrp, 'AF4')
    if 'F8' not in exclude:
        eventgrp, F8np = selectAndAddChannel(eventgrp, 'F8')
    if 'F4' not in exclude:
        eventgrp, F4np = selectAndAddChannel(eventgrp, 'F4')
    if 'FC6' not in exclude:
        eventgrp, FC6np = selectAndAddChannel(eventgrp, 'FC6')
    if 'T8' not in exclude:
        eventgrp, T8np = selectAndAddChannel(eventgrp, 'T8')
    if 'P8' not in exclude:
        eventgrp, P8np = selectAndAddChannel(eventgrp, 'P8')
    if 'O1' not in exclude:
        eventgrp, O1np = selectAndAddChannel(eventgrp, 'O1')
    if 'O2' not in exclude:
        eventgrp, O2np = selectAndAddChannel(eventgrp, 'O2')

    list_of_vars = [AF3np, AF4np, F7np, F8np, F3np, F4np, FC5np, FC6np, P7np, P8np, T7np, T8np, O1np, O2np]
    list_of_vars = [x for x in list_of_vars if type(x) != NoneType]  # remove None channels  #todo: the

    fullChannel = torch.tensor(np.vstack(list_of_vars))
    target = int(eventgrp.iloc[0]['code'])
    corrLen = True

    if list(fullChannel.size())[1] < 250:       # remove all instances shorter than 250 events
        print("Short data length")
        corrLen = False
    
    if len(list_of_vars) != (14-len(exclude)):
        print("Fewer channels than expected - ", len(list_of_vars))
        corrLen = False

    elif target < 0:
        print("-1 target val")
        corrLen = False

        
    # print("full channel", fullChannel, "\ntarget", target)

    return fullChannel, target, corrLen


filename = "/data/cs66/bmi/EP1.01.txt"  # change this line if the location of the file has changed]


def dataLoader(filename, exclude, outpathData, outpathTargets):
    lines = []
    lineMax = 0

    with open(filename) as tsv:
        for line in csv.reader(tsv, dialect="excel-tab"):
            if lineMax == 2000000:  # change this line to select how many examples the dataset to contain
                break
            lines.append(line)
            lineMax += 1

    dF = pd.DataFrame.from_records(lines, columns=['id', 'event', 'device', 'channel', 'code', 'size', 'data'])
    cleanDf = dF.drop(['id', 'device'], axis=1)
    cleanDf['size'] = pd.to_numeric(cleanDf['size'])
    cleanDf['event'] = pd.to_numeric(cleanDf['event'])
    minSize = cleanDf['size'].min()

    events = cleanDf['event'].drop_duplicates().to_numpy()
    numEvents = len(events)

    eventGroups = []
    targets = []
    eventTensors = []

    print("Grouping event channels")
    for event in events:
        eventGroups.append(cleanDf.loc[cleanDf['event'] == event])
        if len(eventGroups[-1]) != 14:
            print("Popped short event with len=" + str(len(eventGroups[-1])))
            eventGroups.pop()
        # print("Event: ", event)

    for eventGroup in eventGroups:
        tensor, target, corrLen = createTensorGroup(eventGroup, transform=True, exclude=exclude)
        if corrLen:
            eventTensors.append(tensor)
            targets.append(target)


    print("dims of the data tensor", eventTensors[0].shape, "type:", type(eventTensors))
    print("num of targets", len(targets), "type:", type(targets))
    fullDataTensor = torch.stack(eventTensors)
    targetTensor = torch.tensor(targets)
    print("shapes are", fullDataTensor.shape, fullDataTensor.shape[1], targetTensor.shape)

    torch.save(fullDataTensor, outpathData)
    torch.save(targetTensor, outpathTargets)
    print("Finished creating a tensor. Exclude=", exclude)


# name of the dataset
exclude = ['O1', 'O2']
outpathData = '/scratch/akazako1/10DigData_no' + exclude[0]+exclude[1] + '.pt'
outpathTargets = '/scratch/akazako1/10DigTarg_no' + exclude[0]+exclude[1] + '.pt'
dataLoader(filename, exclude, outpathData, outpathTargets)
 
exclude = ['AF3', 'AF4']
outpathData = '/scratch/akazako1/10DigData_no' + exclude[0] + exclude[1] + '.pt'
outpathTargets = '/scratch/akazako1/new_10DigTarg_no' + exclude[0] + exclude[1] + '.pt'
dataLoader(filename, exclude, outpathData, outpathTargets)


exclude = ['F7', 'F8']
outpathData = '/scratch/akazako1/10DigData_no' + exclude[0]+exclude[1] + '.pt'
outpathTargets = '/scratch/akazako1/new_10DigTarg_no' + exclude[0]+exclude[1] + '.pt'
dataLoader(filename, exclude, outpathData, outpathTargets)

exclude = ['F3', 'F4',]
outpathData = '/scratch/akazako1/10DigData_no' + exclude[0]+exclude[1] + '.pt'
outpathTargets = '/scratch/akazako1/new_10DigTarg_no' + exclude[0]+exclude[1] + '.pt'
dataLoader(filename, exclude, outpathData, outpathTargets)

exclude = ['FC5', 'FC6']
outpathData = '/scratch/akazako1/10DigData_no' + exclude[0]+exclude[1] + '.pt'
outpathTargets = '/scratch/akazako1/new_10DigTarg_no' + exclude[0]+exclude[1] + '.pt'
dataLoader(filename, exclude, outpathData, outpathTargets)

exclude = ['T7', 'T8']
outpathData = '/scratch/akazako1/10DigData_no' + exclude[0]+exclude[1] + '.pt'
outpathTargets = '/scratch/akazako1/new_10DigTarg_no' + exclude[0]+exclude[1] + '.pt'
dataLoader(filename, exclude, outpathData, outpathTargets)

exclude= ['P7', 'P8']
outpathData = '/scratch/akazako1/10DigData_no' + exclude[0]+exclude[1] + '.pt'
outpathTargets = '/scratch/akazako1/new_10DigTarg_no' + exclude[0]+exclude[1] + '.pt'
dataLoader(filename, exclude, outpathData, outpathTargets)



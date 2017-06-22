# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import json
curDir='C:/Users/Kicky/Documents/Nerd projectjes/170619 CNN met Rene/'
train_dic='fgvc4_iMat.train.data.json'

with open(curDir+train_dic) as json_data:
    d = json.load(json_data)


annotations = d['annotations']
images = d['images']

#Make dataframe of annotation imageId, labelId, taskId


#Reshape pictures to same size, fill up with zeros

#Load imagenet model


#Retr
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 20 11:25:02 2017

@author: Kicky
"""
import json
import numpy as np
import os
import cv2

def get_filepaths(directory):
    """
    This function will generate the file names in a directory 
    tree by walking the tree either top-down or bottom-up. For each 
    directory in the tree rooted at directory top (including top itself), 
    it yields a 3-tuple (dirpath, dirnames, filenames).
    """
    file_paths = []  # List which will store all of the full filepaths.

    # Walk the tree.
    for root, directories, files in os.walk(directory):
        for filename in files:
            # Join the two strings in order to form the full filepath.
            filepath = os.path.join(root, filename)
            file_paths.append(filepath)  # Add it to the list.

    return file_paths  # Self-explanatory.

def make_data(list_x, annot_dic, curDir):
    with open(curDir+annot_dic) as json_data:
      d = json.load(json_data)
    annotations = d['annotations']
    
    #Make dataframe of annotation imageId (index), labelId, taskId
    import pandas as pd    
    df=pd.DataFrame(annotations)
    df=df.set_index(['imageId'])
    
    #Permutate data
    x=[]
    idx = np.random.permutation(len(list_x))
    for i in idx:
        x.append(list_x[i])
    y_label=[]
    y_task=[]
    for i in x:
        imageID=os.path.splitext(os.path.basename(i))[0]
        y_label.append([*map(int, list(df.loc[imageID]['labelId']))])
        y_task.append([*map(int, list(df.loc[imageID]['taskId']))])
    return x,y_label, y_task,df
    
    
def hot_encoding(df, y_label, y_task):
    nr_labels=len(df['labelId'].value_counts())
    nr_tasks=len(df['taskId'].value_counts())
        
    hot_y_label=np.zeros((len(y_label),nr_labels+1), dtype=np.int)
    hot_y_task=np.zeros((len(y_task),nr_tasks+1), dtype=np.int)
    for i in range(len(y_label)):
      hot_y_label[i,y_label[i]] = 1
      hot_y_task[i,y_task[i]] = 1
    return hot_y_label, hot_y_task

def get_batch(x_list, batch_size, img_size):
  images = np.ndarray([batch_size, img_size, img_size, 3])
  for i in range(len(x_list)):
    img = cv2.imread(x_list[i])
    #flip image at random if flag is selected
    if np.random.random() < 0.5:
      img = cv2.flip(img, 1)
    
    images[i] = img
    
  return images
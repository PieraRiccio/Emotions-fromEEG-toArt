import numpy as np
import pandas as pd
import os
import scipy.io
from keras.utils import to_categorical
from scipy.spatial import distance_matrix

import torch
import torch_geometric

####################
#####PREPROCESS#####
####################

def emotionDL(eps=0.1):
  labels = { 
    '1' : to_categorical(np.array([1,2,3,0,2,0,0,1,0,1,2,1,1,1,2,3,2,2,3,3,0,3,0,3])), 
    '2' : to_categorical(np.array([2,1,3,0,0,2,0,2,3,3,2,3,2,0,1,1,2,1,0,3,0,1,3,1])), 
    '3' : to_categorical(np.array([1,2,2,1,3,3,3,1,1,2,1,0,2,3,3,0,2,3,0,0,2,0,1,0]))
  }

  for i in range(1, 4):
    for row in labels[str(i)]:
      if (row == [1, 0, 0, 0]).all():
        row[0] = row[0]-3/4*eps
        row[1:4] = row[1:4]+eps/4
      elif (row == [0, 1, 0, 0]).all():
        row[0] = row[0]+1/3*eps
        row[2] = row[2]+1/3*eps
        row[1] = row[1]-2/3*eps
      elif (row == [0, 0, 1, 0]).all():
        row[0] = row[0]+1/4*eps
        row[1] = row[1]+1/4*eps
        row[2] = row[2]-3/4*eps
        row[3] = row[3]+1/4*eps
      elif (row==[0, 0, 0, 1]).all():
        row[0] = row[0]+1/3*eps
        row[2] = row[2]+1/3*eps
        row[3] = row[3]-2/3*eps
  return labels

def load_data(seediv_path,labels):
  eeg_dict = {}
  for session in range (1, 4):
    #print("WE ARE IN SESSION: {:d}".format(session))
    session_labels = labels[str(session)]
    session_folder = os.path.join(seediv_path,str(session))
    for file in os.listdir(session_folder):
      p=int(file.split("_")[0])
      file_path = os.path.join(session_folder,file)
      #print("THIS IS PATIENT: {:d}".format(p))
      for videoclip in range(24):
        x = scipy.io.loadmat(file_path)['de_LDS{}'.format(videoclip+1)]
        if p not in eeg_dict.keys():
          eeg_dict[p] = {
            "neutral": {"train": [],"val": []},
            "sadness": {"train": [],"val": []},
            "fear": {"train": [],"val": []},
            "happiness": {"train": [],"val": []}
          }
        y=session_labels[videoclip]
        emotion=list(eeg_dict[p].keys())[np.argmax(y)]
        data=(x,y)
        if (len(eeg_dict[p][emotion]["train"]) - (session-1)*4) <4:
          eeg_dict[p][emotion]["train"].append(data)
        else:
          eeg_dict[p][emotion]["val"].append(data)
  for p in eeg_dict.keys():
    for emotion in eeg_dict[p].keys():
      for phase in ["train","val"]:
        data=[]
        for X,y in eeg_dict[p][emotion][phase]:
          y = torch.tensor([y]).float()
          edge_index = torch.tensor([np.arange(62*62)//62, np.arange(62*62)%62])
          for t in range(X.shape[1]):
            x = torch.tensor(X[:, t, :]).float()
            data.append(torch_geometric.data.Data(x=x, y=y, edge_index=edge_index))
        eeg_dict[p][emotion][phase]=data
  return eeg_dict

####################
#####DATALOADER#####
####################

def global_channels(adjacency_matrix, filtered_channels, channel_tuples):
  for channel1,channel2 in channel_tuples:
    adjacency_matrix[filtered_channels.index(channel1), filtered_channels.index(channel2)] = adjacency_matrix[filtered_channels.index(channel1), filtered_channels.index(channel2)] - 1
    adjacency_matrix[filtered_channels.index(channel2), filtered_channels.index(channel1)] = adjacency_matrix[filtered_channels.index(channel2), filtered_channels.index(channel1)] - 1
  return adjacency_matrix

def get_adjacency_matrix(channel_order_path,channel_location_path):
  channel_order = pd.read_excel(channel_order_path, header=None)
  channel_location = pd.read_csv(channel_location_path, sep= ",")
  filtered_df = pd.DataFrame(columns=["Channel", "X", "Y", "Z"])
  for channel in channel_location["Channel"]:
    for used in channel_order[0]:
      if channel.upper() == used:
        filtered_df = pd.concat([channel_location.loc[channel_location['Channel'] == channel], filtered_df], ignore_index=True)
  filtered_df = filtered_df.reindex(index=filtered_df.index[::-1]).reset_index(drop=True)
  filtered_matrix = np.asarray(filtered_df.values[:, 1:4], dtype=float)
  distances_matrix = distance_matrix(filtered_matrix, filtered_matrix)
  delta = 5
  adjacency_matrix = np.minimum(np.ones([62,62]), delta/(distances_matrix**2))
  filtered_channels = list(filtered_df["Channel"])
  adjacency_matrix = global_channels(adjacency_matrix, filtered_channels,
    [("Fp1", "Fp2"), ("AF3", "AF4"), ("F5", "F6"), ("FC5", "FC6"), ("C5", "C6"), ("CP5", "CP6"), ("P5", "P6"), ("PO5", "PO6"), ("O1", "O2")]
  )
  adjacency_matrix = np.absolute(adjacency_matrix)
  return adjacency_matrix

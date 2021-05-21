import os
import numpy as np
import pandas as pd
from PIL import Image
from io import BytesIO

import torch
import torchvision
from torchvision.transforms.functional import resize,center_crop,InterpolationMode

####################
#######IMPORT#######
####################

def getGroup(row,groups):
  for group in groups:
    if sum([row["ImageOnly: {}".format(emotion)] for emotion in group.split("_")])>0:
      return group
  return None

def download_wikiArtEmotion(path_wikiart):
  tsv=pd.read_csv("WikiArt-Emotions-Ag4.tsv",sep='\t')
  groups=["anger_disgust","fear_shame","happiness_optimism","sadness_pessimism"]
  lengths=[sum([sum(tsv["ImageOnly: {}".format(emotion)]) for emotion in group.split("_")]) for group in groups]
  groups=[groups[i] for i in np.argsort(lengths)]
  os.mkdir(path_wikiart)
  for group in groups: os.mkdir(os.path.join(path_wikiart,group))
  counter={group:0 for group in groups}
  for _,row in tsv.iterrows():
    group=getGroup(row,groups)
    if group is not None:
      flag=len(os.listdir(os.path.join(path_wikiart,group)))
      link=row["Image URL"].replace("use2-","")
      dst=os.path.join(path_wikiart,group,"{:04d}_{}".format(counter[group],link.split("/")[-1]))
      os.system("wget '{}' -O '{}' --no-check-certificate".format(link,dst))
      if(flag==len(os.listdir(os.path.join(path_wikiart,group)))):
        raise Exception("Something went wrong with the following link: {}".format(link))
      counter[group]+=1

####################
####AUGMENTATION####
####################

def getSortedList(group_path):
  sortedList=[]
  for image in sorted(os.listdir(group_path)):
    image=Image.open(os.path.join(group_path,image))
    sortedList.append(image.size[0]*image.size[1])
  return [sorted(os.listdir(group_path))[i] for i in np.argsort(sortedList)][::-1]

def split_image(image_path):
  image=Image.open(image_path)
  width,height=image.size
  if width>600 and height>600:
    for i,j in zip(np.arange(4)//2,np.arange(4)%2):
      sub_image=image.crop((i*width//2,j*height//2,(i+1)*width//2,(j+1)*height//2))
      sub_image.save(image_path.replace(".jpg","_{:1d}.jpg".format(i*2+j)))

def data_augmentation(path_wikiart,groups_to_augment,max):
  for group in groups_to_augment:
    group_path=os.path.join(path_wikiart,group)
    processed=[]
    while(len(os.listdir(group_path))!=len(processed) and len(os.listdir(group_path))<=max):
      sortedList=getSortedList(group_path)
      for image in sortedList:
        if image not in processed:
          processed.append(image)
          split_image(os.path.join(group_path,image))
          if(len(os.listdir(group_path))>max):
            break

####################
#####PREPROCESS#####
####################

def preprocess_wikiart(path_wikiart,size=128):
  wikiArt_dict = {
      "anger_disgust": {"images" : [], "labels" : []},
      "sadness_pessimism": {"images" : [], "labels" : []},
      "fear_shame": {"images" : [], "labels" : []},
      "happiness_optimism": {"images" : [], "labels" : []}
  }
  for label,folder in enumerate(os.listdir(path_wikiart)):
    label=torch.tensor(label)
    for f in os.listdir(os.path.join(path_wikiart,folder)):
      img_path = os.path.join(path_wikiart,folder,f)
      img = Image.open(img_path).convert("RGB")
      img = resize(img, size, InterpolationMode.LANCZOS)
      img = center_crop(img, size)
      buffer = BytesIO()
      img.save(buffer, format="jpeg", quality=100)
      val = buffer.getvalue()
      wikiArt_dict[folder]["images"].append(val)
      wikiArt_dict[folder]["labels"].append(label)
  return wikiArt_dict

####################
#####DATALOADER#####
####################

class WikiArtDataLoader(torch.utils.data.Dataset):
  def __init__(self,wikiArt_dict,transform=None):
    self.wikiArt_dict=wikiArt_dict
    self.transform=transform

  def __len__(self):
    return len(self.wikiArt_dict["labels"])
  
  def __getitem__(self, id):
    sample={
        "images": self.wikiArt_dict["images"][id],
        "labels": self.wikiArt_dict["labels"][id]
    }
    buffer = BytesIO(sample["images"])
    sample["images"] = Image.open(buffer)
    if self.transform:
      sample["images"] = self.transform(sample["images"])
    return sample

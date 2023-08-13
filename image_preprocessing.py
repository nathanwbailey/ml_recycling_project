import torch
import torchvision
import xml.etree.ElementTree as ET 
from PIL import Image
import os

names = []
labels = []

xml_filename = "image_labels.xml"
data_dir = 'compiled_dataset'
xml_file = open(data_dir+'/'+xml_filename, 'r')
xml_root = ET.fromstring(xml_file.read())

for item in xml_root.findall('Items/Item'):
    names.append(item.attrib['imageName'])
    labels.append(int(item.attrib['label']))
            
mapping = {0:'glass', 1:'metal', 2:'organic', 3:'paper', 4:'plastic'}
        
num_images = len(names)

for idx in range(num_images):
    image_name = names[idx]
    label = labels[idx]
    image = Image.open(data_dir+'/'+mapping[label]+'/'+image_name)
    image_tensor = torchvision.transforms.functional.pil_to_tensor(image)
    if int(image_tensor.size()[0]) == 1:
        print(image_name)
        os.remove(data_dir+'/'+mapping[label]+'/'+image_name)

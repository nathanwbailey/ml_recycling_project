import torch
import xml.etree.ElementTree as ET 
from sklearn.utils import shuffle
from PIL import Image

class RecyclingDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir, dataset_type, data_transforms):
        self.data_transforms = data_transforms
        xml_filename = "image_labels.xml"
        self.glass_dir = data_dir+'/'+'glass/'
        self.plastic_dir = data_dir+'/'+'plastic/'
        self.metal_dir = data_dir+'/'+'metal/'
        self.paper_dir = data_dir+'/'+'paper/'
        self.organic_dir = data_dir+'/'+'organic/'
        self.data_dir = data_dir
        xml_file = open(data_dir+'/'+xml_filename, 'r')
        xml_root = ET.fromstring(xml_file.read())
        self.names = []
        self.labels = []
        
        for item in xml_root.findall('Items/Item'):
            self.names.append(item.attrib['imageName'])
            self.labels.append(int(item.attrib['label']))
            
        self.mapping = {0:'glass', 1:'metal', 2: 'organic', 3:'paper', 4:'plastic'}
        
        valid_num = int(len(self.names)*0.5)
        test_num = int(len(self.names)*0.0)
        train_num = int(len(self.names)- (test_num+valid_num))
        
        self.names, self.labels = shuffle(self.names, self.labels, random_state=20)
        if dataset_type == 'train':
            self.names, self.labels = self.names[:train_num], self.labels[:train_num]
        elif dataset_type == 'valid':
            self.names, self.labels = self.names[train_num:train_num+valid_num], self.labels[train_num:train_num+valid_num]
        else:
            self.names, self.labels = self.names[train_num+valid_num:], self.labels[train_num+valid_num:]
            
    def __len__(self):
        return len(self.names)

    def __getitem__(self, idx):
        image_name = self.names[idx]
        label = self.labels[idx]
        image_orig = Image.open(self.data_dir+'/'+self.mapping[label]+'/'+image_name)
        image_path = self.data_dir+'/'+self.mapping[label]+'/'+image_name
        if self.data_transforms:
            image = self.data_transforms(image_orig)
        return (image, label, image_path)
            
        
        
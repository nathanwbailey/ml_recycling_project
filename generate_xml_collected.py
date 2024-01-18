import os 
import shutil
from lxml import etree


num_images_glass = len(os.listdir(path='collected_dataset_with_labels_no_5_no_6//glass'))
num_images_metal = len(os.listdir(path='collected_dataset_with_labels_no_5_no_6//metal'))
num_images_paper = len(os.listdir(path='collected_dataset_with_labels_no_5_no_6//paper'))
num_images_plastic = len(os.listdir(path='collected_dataset_with_labels_no_5_no_6//plastic'))

root = etree.Element("Images")
items = etree.SubElement(root, "Items", num_images = str(num_images_glass+num_images_metal+num_images_paper+num_images_plastic))

for idx, image_label in enumerate(['glass', 'metal', 'organic', 'paper', 'plastic']):
    try:
        list_files = os.listdir('collected_dataset_with_labels_no_5_no_6/'+image_label)
    except:
        continue
    list_files.sort()
    for f in list_files:
        id = f
        label = str(idx)
        item = etree.SubElement(items, "Item", imageName=id, label=label)

tree = etree.ElementTree(root)
tree.write("collected_dataset_with_labels_no_5_no_6//image_labels.xml", pretty_print=True)
    
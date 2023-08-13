import os 
import shutil
from lxml import etree


num_images_glass = len(os.listdir(path='compiled_dataset/glass'))
num_images_metal = len(os.listdir(path='compiled_dataset/metal'))
num_images_organic = len(os.listdir(path='compiled_dataset/organic'))
num_images_paper = len(os.listdir(path='compiled_dataset/paper'))
num_images_plastic = len(os.listdir(path='compiled_dataset/plastic'))

root = etree.Element("Images")
items = etree.SubElement(root, "Items", num_images = str(num_images_glass+num_images_metal+num_images_organic+num_images_paper+num_images_plastic))

for idx, image_label in enumerate(['glass', 'metal', 'organic', 'paper', 'plastic']):
    list_files = os.listdir('compiled_dataset/'+image_label)
    list_files.sort()
    for f in list_files:
        id = f
        label = str(idx)
        item = etree.SubElement(items, "Item", imageName=id, label=label)

tree = etree.ElementTree(root)
tree.write("compiled_dataset/image_labels.xml", pretty_print=True)
    
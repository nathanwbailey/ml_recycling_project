import torch
import torchvision
import sys
import get_council_data
import dataset
import network


#Grab the council, and what can be recycled or not, grab a test image and see
postcode = 'RG6 7DD'

council_data = get_council_data.get_from_database(get_council_data.get_council_data(postcode=postcode))

print(council_data)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

mean = torch.Tensor([0.6661, 0.6211, 0.5492])
std = torch.Tensor([0.2871, 0.2917, 0.3310])


transforms = torchvision.transforms.Compose([torchvision.transforms.Resize((150,150)), torchvision.transforms.ToTensor(), torchvision.transforms.Normalize(mean, std)])


test_dataset = dataset.RecyclingDataset(data_dir='compiled_dataset', dataset_type='test', data_transforms=transforms)
testloader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=False, pin_memory=True)

mapped_labels = {0:'glass', 1:'metal', 2:'organic', 3:'paper', 4:'plastic'}

model = network.RecycleNetwork(num_classes=5).to(device)
model.load_state_dict(torch.load('recycle_net.pt'))
model.eval()
with torch.no_grad():
    batch = next(iter(testloader))
    images = batch[0].to(device)
    labels = batch[1].to(device)
    output = model(images)
    softmaxed_output = torch.nn.functional.softmax(output, dim=1)
    predictions = torch.argmax(softmaxed_output, dim=1)
    predictions = list(predictions.cpu().numpy())
    for pred in predictions:
        material_type = mapped_labels[pred]
        if material_type in council_data['Recyclable']:
            print('Recyclable')
        elif council_data['Organics'] and material_type == 'organic':
            print('Food Waste')
        else:
            print('Non-recyclable')
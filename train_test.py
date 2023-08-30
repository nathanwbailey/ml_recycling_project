import torch
import sys
import numpy as np
from early import pytorchtools

def calculate_accuracy(outputs, ground_truth):
    softmaxed_output = torch.nn.functional.softmax(outputs, dim=1)
    predictions = torch.argmax(softmaxed_output, dim=1)
    ground_truth = torch.argmax(ground_truth, dim=1)
    num_correct = torch.sum(torch.eq(predictions, ground_truth)).item()
    return num_correct, ground_truth.size(0)


def calculate_accuracy_val(outputs, ground_truth):
    softmaxed_output = torch.nn.functional.softmax(outputs, dim=1)
    predictions = torch.argmax(softmaxed_output, dim=1)
    num_correct = torch.sum(torch.eq(predictions, ground_truth)).item()
    return num_correct, ground_truth.size(0)

def train_network(model, num_epochs, optimizer, loss_function, trainloader, validloader, device, patience=10, path_to_model='recycle_net', scheduler=None):
    print('Training Started')
    sys.stdout.flush()
    early_stop = pytorchtools.EarlyStopping(patience=patience, verbose=True,path=path_to_model+'.pt')
    for epoch in range(1, num_epochs+1):
        train_loss = []
        valid_loss = []
        num_correct_train = 0
        num_examples_train = 0
        num_correct_valid = 0
        num_examples_valid = 0
        model.train()
        for batch in trainloader:
            optimizer.zero_grad()
            images = batch[0].to(device)
            labels = batch[1].to(device)
            outputs = model(images)
            loss = loss_function(outputs, labels)
            #Calculate the gradients for parameters (backprop)
            loss.backward()
            #Update the values of the parameter using the optimizer
            optimizer.step()
            train_loss.append(loss.item())
            num_corr, num_ex = calculate_accuracy(outputs, labels)
            num_examples_train += num_ex
            num_correct_train += num_corr
        
        model.eval()
        with torch.no_grad():
            for batch in validloader:
                images = batch[0].to(device)
                labels = batch[1].to(device)
                outputs = model(images)
                loss = loss_function(outputs, labels)
                valid_loss.append(loss.item())
                num_corr, num_ex = calculate_accuracy_val(outputs, labels)
                num_examples_valid += num_ex
                num_correct_valid += num_corr
                
        
        print('Epoch: {}, Training Loss: {:4f}, Validation Loss: {:4f}, Training Accuracy: {:4f}, Validation Accuracy: {:4f}'.format(epoch, np.mean(train_loss), np.mean(valid_loss), num_correct_train/num_examples_train, num_correct_valid/num_examples_valid))
        
        if scheduler:
            scheduler.step(np.mean(valid_loss))
        
        if early_stop.counter == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'validation_loss': np.mean(valid_loss),
                'train_loss': np.mean(train_loss)
                }, 'full_model.tar')
            torch.save(model, 'model.pth')
        
        early_stop_loss = np.mean(valid_loss)
        early_stop(early_stop_loss, model)
        
        if early_stop.early_stop:
            print('Early Stopping at Epoch: {}'.format(epoch))
            break
    sys.stdout.flush()
            
    model.load_state_dict(torch.load(path_to_model+'.pt'))
    model.to(device)
    return model


def test_network(model, testloader, loss_function, device):
    test_loss = []
    num_examples = 0
    num_correct = 0
    model.eval()
    with torch.no_grad():
        for batch in testloader:
            images = batch[0].to(device)
            labels = batch[1].to(device)
            output = model(images)
            loss = loss_function(output, labels)
            test_loss.append(loss.item())
            num_corr, num_ex = calculate_accuracy_val(output, labels)
            num_examples += num_ex
            num_correct += num_corr

    print('Test Loss: {:4f}, Test Accuracy: {:.4f}'.format(np.mean(test_loss), num_correct/num_examples))
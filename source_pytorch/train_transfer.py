import argparse
import json
import os
import pandas as pd
import torch
import torch.optim as optim
import torch.utils.data

# pytorch
import torch
import torch.nn as nn
import torch.utils.data

import torchvision.transforms as transforms
from torchvision import datasets
from torch.utils.data.sampler import SubsetRandomSampler
import torchvision.models as models

import numpy as np

from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

# imports the model in model.py by name
from model import FruitClassifier

def model_fn(model_dir):
    """Load the PyTorch model from the `model_dir` directory."""
    print("Loading model.")

    # First, load the parameters used to create the model.
    model_info = {}
    model_info_path = os.path.join(model_dir, 'model_info_transfer.pth')
    with open(model_info_path, 'rb') as f:
        model_info = torch.load(f)

    print("model_info: {}".format(model_info))

    # Determine the device and construct the model.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = FruitClassifier(model_info['class_count'])

    # Load the stored model parameters.
    model_path = os.path.join(model_dir, 'model.pth')
    with open(model_path, 'rb') as f:
        model.load_state_dict(torch.load(f))

    # set to eval mode, could use no_grad
    model.to(device).eval()

    print("Done loading model.")
    return model

# Gets training data in batches from the train.csv file
def _get_train_data_loader(batch_size, training_dir):
    print("Get train data loader.")
    print(training_dir)
    normalize_transfer = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])


    #Need to resize images to 224x224 px for pretrained model
    resize_transform = transforms.Resize(256)
    centercrop_transform = transforms.CenterCrop(224)

    data_transforms_transfer = {
        'Training': transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.ColorJitter(),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ToTensor(),
            normalize_transfer
        ]),
        'Validation': transforms.Compose([
            resize_transform,
            centercrop_transform,
            transforms.ToTensor(),
            normalize_transfer
        ]), 
        'Test': transforms.Compose([
            resize_transform,
            centercrop_transform,
            transforms.ToTensor(),
            normalize_transfer
        ]),    
    }

    image_datasets_transfer = {x: datasets.ImageFolder(os.path.join(training_dir, x), data_transforms_transfer[x])
                              for x in ['Training', 'Test']}
    class_names = image_datasets_transfer['Training'].classes
    class_count = len(class_names)
    print(class_names)
    print("Class count %d" % (class_count))

    batch_size = 20
    num_workers = 0
    valid_size = 0.2

    # obtain training indices that will be used for validation
    num_train = len(image_datasets_transfer['Training'])
    indices = list(range(num_train))
    np.random.shuffle(indices)
    split = int(np.floor(valid_size * num_train))
    train_idx, valid_idx = indices[split:], indices[:split]

    train_sampler = SubsetRandomSampler(train_idx)
    valid_sampler = SubsetRandomSampler(valid_idx)    


    # prepare data loaders
    loaders_transfer = {}
    loaders_transfer['Training'] = torch.utils.data.DataLoader(image_datasets_transfer['Training'], batch_size=batch_size,
        sampler=train_sampler, num_workers=num_workers)
    loaders_transfer['Validation'] = torch.utils.data.DataLoader(image_datasets_transfer['Training'], batch_size=batch_size, 
        sampler=valid_sampler, num_workers=num_workers)
    loaders_transfer['Test'] = torch.utils.data.DataLoader(image_datasets_transfer['Test'], batch_size=batch_size, 
        num_workers=num_workers)

    return loaders_transfer

# Provided training function
def train(n_epochs, loaders, model, optimizer, criterion, use_cuda, save_path):
    """
    This is the training method that is called by the PyTorch training script. The parameters
    passed are as follows:
    model        - The PyTorch model that we wish to train.
    train_loader - The PyTorch DataLoader that should be used during training.
    epochs       - The total number of epochs to train for.
    criterion    - The loss function used for training. 
    optimizer    - The optimizer to use during training.
    device       - Where the model and data should be loaded (gpu or cpu).
    """
    
    loss_list = []
    acc_list = []

    # training loop is provided
    valid_loss_min = np.Inf 
    total_step = len(loaders['Training'])

    for epoch in range(1, n_epochs+1):
        # initialize variables to monitor training and validation loss
        train_loss = 0.0
        valid_loss = 0.0
        print('Started epoch')
        ###################
        # train the model #
        ###################
        model.train()
        for batch_idx, (data, target) in enumerate(loaders['Training']):
            # move to GPU
            if use_cuda:
                data, target = data.cuda(), target.cuda()
            data, target = data.to(device), target.to(device)
            ## find the loss and update the model parameters accordingly
            ## record the average training loss, using something like
      
            optimizer.zero_grad()
            # Get output
            output = model(data)               
            # Calculate loss
            loss = criterion(output, target)
            loss.backward()
            optimizer.step() 
            train_loss = train_loss + (1 / (batch_idx + 1)) * (loss.data - train_loss)
            
            # Track the accuracy
            total = target.size(0)
            _, predicted = torch.max(output.data, 1)
            correct = (predicted == target).sum().item()
            acc_list.append(correct / total)

            if (batch_idx + 1) % 100 == 0:
                print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Accuracy: {:.2f}%'
                      .format(epoch + 1, n_epochs, batch_idx + 1, total_step, loss.item(),
                              (correct / total) * 100))
            
        ######################    
        # validate the model #
        ######################
        model.eval()
        for batch_idx, (data, target) in enumerate(loaders['Validation']):
            # move to GPU
            if use_cuda:
                data, target = data.cuda(), target.cuda()
            output = model(data)
            loss = criterion(output, target)
            valid_loss = valid_loss + (1 / (batch_idx + 1)) * (loss.data - valid_loss)
        
        # print training/validation statistics 
        print('Epoch: {} \tTraining Loss: {:.6f} \tValidation Loss: {:.6f}'.format(
            epoch, 
            train_loss,
            valid_loss
            ))
        
        ## TODO: save the model if validation loss has decreased
        if valid_loss <= valid_loss_min:
            print('Saving model: {} \tNew Valid Loss: {:.6f} \tPrevious Valid Loss: {:.6f}'.format(
                epoch, 
                valid_loss,
                valid_loss_min
                ))
            torch.save(model.state_dict(), save_path)
            valid_loss_min = valid_loss   

def test(loaders, model, criterion, use_cuda):

    # monitor test loss and accuracy
    test_loss = 0.
    correct = 0.
    total = 0.

    model.eval()
    for batch_idx, (data, target) in enumerate(loaders['Test']):
        # move to GPU
        if use_cuda:
            data, target = data.cuda(), target.cuda()
        # forward pass: compute predicted outputs by passing inputs to the model
        output = model(data)
        # calculate the loss
        loss = criterion(output, target)
        # update average test loss 
        test_loss = test_loss + ((1 / (batch_idx + 1)) * (loss.data - test_loss))
         # convert output probabilities to predicted class
        pred = output.data.max(1, keepdim=True)[1]
        # compare predictions to true label
        correct += np.sum(np.squeeze(pred.eq(target.data.view_as(pred))).cpu().numpy())
        total += data.size(0)        
    print('Test Loss: {:.6f}\n'.format(test_loss))

    print('\nTest Accuracy: %2d%% (%2d/%2d)' % (
        100. * correct / total, correct, total))

## TODO: Complete the main code
if __name__ == '__main__':
    
    # All of the model parameters and training parameters are sent as arguments
    # when this script is executed, during a training job
    
    # Here we set up an argument parser to easily access the parameters
    parser = argparse.ArgumentParser()

    # SageMaker parameters, like the directories for training data and saving models; set automatically
    # Do not need to change
    parser.add_argument('--output-data-dir', type=str, default=os.environ['SM_OUTPUT_DATA_DIR'])
    parser.add_argument('--model-dir', type=str, default=os.environ['SM_MODEL_DIR'])
    parser.add_argument('--data-dir', type=str, default=os.environ['SM_CHANNEL_TRAIN'])
    
    # Training Parameters, given
    parser.add_argument('--batch-size', type=int, default=20, metavar='N',
                        help='input batch size for training (default: 10)')
    parser.add_argument('--epochs', type=int, default=20, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                        help='learning rate (default: 0.001)')
    
    ## TODO: Add args for the three model parameters: input_features, hidden_dim, output_dim
    # Model Parameters
    # Model parameters
    parser.add_argument('--class_count', type=int, default=1, metavar='OUT',
                        help='input dimension (default: 1)')
    parser.add_argument('--save_path', type=str, default='model_fruit_transfer.pt', metavar='OUT',
                        help='input dimension (default: 1)')

    # args holds all passed-in arguments
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device {}.".format(device))

    # Load the training data.
    train_loader = _get_train_data_loader(args.batch_size, args.data_dir)


    ## --- Your code here --- ##
    
    # Don't forget to move your model .to(device) to move to GPU , if appropriate
    model = models.vgg16(pretrained=True)
    for param in model.parameters():
        param.requires_grad = False

    fc_inputs = model.classifier[6].in_features

    # Trains the model (given line of code, which calls the above training function)
    use_cuda = torch.cuda.is_available()

    model.classifier[6] = nn.Sequential(
                      nn.Linear(fc_inputs, 256), 
                      nn.ReLU(), 
                      nn.Dropout(0.4),
                      nn.Linear(256, args.class_count),                   
                      nn.LogSoftmax(dim=1))

    if use_cuda:
        model.cuda()
    print(model)

    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.parameters())

    train(args.epochs, train_loader, model, optimizer, criterion, use_cuda, args.save_path)

    print("Finished training.Testing now")

    model.load_state_dict(torch.load(args.save_path))
    test(train_loader, model, criterion, use_cuda) 

    print("Finished testing.  Exiting")


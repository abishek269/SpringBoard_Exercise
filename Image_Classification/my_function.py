# Imports here
import pandas as pd
import torch
import PIL
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from torchvision import datasets, transforms, models
from torch import nn
from torch import optim
from collections import OrderedDict
import time
##%matplotlib inline
##%config InlineBackend.figure_format = 'retina'

## Define a function that loads training data ;; HINTS: Same as in Main file
def load_data_train(train_dir):
    train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                       transforms.RandomResizedCrop(224),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406], 
                                                            [0.229, 0.224, 0.225])])
    train_data = datasets.ImageFolder(train_dir, transform=train_transforms)
    trainloader = torch.utils.data.DataLoader(train_data, batch_size=50, shuffle=True)
    return train_data, trainloader

## Similar for test data
def load_data_test(test_dir):
    test_transforms = transforms.Compose([transforms.Resize(256),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406], 
                                                           [0.229, 0.224, 0.225])])
    test_data = datasets.ImageFolder(test_dir, transform=test_transforms)
    testloader = torch.utils.data.DataLoader(test_data, batch_size=50)
    return test_data, testloader  

## Similar for Validation data
def load_data_valid(valid_dir):
    valid_transforms = transforms.Compose([transforms.Resize(256),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406], 
                                                           [0.229, 0.224, 0.225])])
    valid_data = datasets.ImageFolder(valid_dir, transform=valid_transforms)
    validloader = torch.utils.data.DataLoader(valid_data, batch_size=50)
    return valid_data, validloader


## Define a function that creates a model
def built_model(arch='vgg19', pretrained=True):
    model = models.vgg19(pretrained=True)
     # Freeze parameters
    for param in model.parameters():
        param.requires_grad = False
    
    # Define a classifier, using ReLU activations and dropout
    classifier = nn.Sequential(OrderedDict([
                          ('fc1', nn.Linear(25088, 4096, bias=True)),
                          ('relu1', nn.ReLU()),
                          ('dropout1', nn.Dropout(p=0.5)),
                          ('fc2', nn.Linear(4096, 102, bias=True)),
                          ('output', nn.LogSoftmax(dim=1))
                          ]))
    model.classifier = classifier
    return model

def chose_device(use_gpu=False):
    if use_gpu:
        if torch.cuda.is_available():
            print('Using GPU')
            device = torch.device('cuda:0')
        else:
            print('GPU not available - falling back to CPU')
            device = torch.device('cpu')
    else:
        print('Using CPU')
        device = torch.device('cpu')
    return device

##Create criterion and optimizer
def criterion():
    return nn.NLLLoss()
### create optimizer:
def optimizer(model, alfa):
    return optim.Adam(model.classifier.parameters(), alfa)

# Calculate test_loss and accuracy
def validation(model, testloader, criterion):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device);
    test_loss = 0
    accuracy = 0
    
    for mm, (inputs, labels) in enumerate(testloader):
        
        inputs, labels = inputs.to(device), labels.to(device)
        
        output = model.forward(inputs)
        test_loss += criterion(output, labels).item()
        
        ps = torch.exp(output)
        equality = (labels.data == ps.max(dim=1)[1])
        accuracy += equality.type(torch.FloatTensor).mean()
    
    return test_loss, accuracy

def train_model(model, trainloader, validloader, testloader, criterion, epochs, optimizer, steps=0):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    print_every=25
    
    #model, criterion, optimizer, dataloader_train, dataloader_valid, args.epochs, args.gpu
    print("Training process started .....\n")
    for e in range(epochs):
        running_loss = 0
        #model.train() # Technically not necessary, setting this for good measure
        for ii, (inputs, labels) in enumerate(trainloader):
            steps += 1
            inputs, labels = inputs.to(device), labels.to(device)        
            optimizer.zero_grad()       
            # Forward and backward passes
            outputs = model.forward(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()   
            running_loss += loss.item()
            if steps % print_every == 0:
                model.eval()
                with torch.no_grad():
                    valid_loss, accuracy = validation(model, validloader, criterion)
                print("Epoch: {}/{} | ".format(e+1, epochs),
                    "Training_Loss: {:.4f} | ".format(running_loss/print_every),
                    "Validation_Loss: {:.4f} | ".format(valid_loss/len(testloader)),
                    "Validation_Accuracy: {:.4f}".format(accuracy/len(testloader)))
            
                running_loss = 0
                model.train()

    print("\n YAY!! Model training is completed")
    return model

      
def save_checkpoint(model, optimizer, train_data, epochs=5):   
    model.class_to_idx = train_data.class_to_idx
    checkpoint = {'epochs': epochs,'classifier': model.classifier,'state_dict': model.state_dict(),'optimizer_dict': optimizer.state_dict(),'class_to_idx': model.class_to_idx}
    torch.save(checkpoint, 'my_checkpoint.pth')
    print("Completed! Model saved as checkpoint\n")
    
def load_my_checkpoint(file_name='my_checkpoint.pth'):
    """
    Loads previously saved checkpoint.
    file name: give full path to ur "my_model_checkpoint.pth"
    
    """
    checkpoint = torch.load(file_name)
    model = models.vgg19(pretrained=True);
    for param in model.parameters(): param.requires_grad = False
    
    # Load stuff from my_model_checkpoint
    model.class_to_idx = checkpoint['class_to_idx']
    model.classifier = checkpoint['classifier']
    model.load_state_dict(checkpoint['state_dict'])   
    return model


def change_cat_to_name(ind):
    file_name="cat_to_name.json"
    import json
    with open(file_name, 'r') as f:
        c2n = json.load(f)
    return c2n[ind]
    
def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    import PIL
    test_image = PIL.Image.open(image)

    # Get original dimensions
    orig_width, orig_height = test_image.size
    #print("Original width of image: ", orig_width)
    #print("Original height of image: ", orig_height)

    # Find shorter size and create settings to crop shortest side to 256
    if orig_width < orig_height: resize_size=[256, 256**600]
    else: resize_size=[256**600, 256]
        
    test_image.thumbnail(size=resize_size)

    # Find pixels to crop on to create 224x224 image
    center = orig_width/4, orig_height/4
    left, top, right, bottom = center[0]-(244/2), center[1]-(244/2), center[0]+(244/2), center[1]+(244/2)
    test_image = test_image.crop((left, top, right, bottom))

    # Converrt to numpy - 244x244 image w/ 3 channels (RGB)
    np_image = np.array(test_image)/255 # Divided by 255 because imshow() expects integers (0:1)!!

    # Normalize each color channel based on provided means and stds
    normalise_means = [0.485, 0.456, 0.406]
    normalise_std = [0.229, 0.224, 0.225]
    np_image = (np_image-normalise_means)/normalise_std
        
    # Set the color to the first channel
    np_image = np_image.transpose(2, 0, 1)
    
    return np_image

def imshow(image, ax=None, title=None):
    if ax is None:
        fig, ax = plt.subplots()
    
    # PyTorch tensors assume the color channel is the first dimension
    # but matplotlib assumes is the third dimension
    image = image.transpose((1, 2, 0))
    
    # Undo preprocessing
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = std * image + mean
    
    # Image needs to be clipped between 0 and 1 or it looks like noise when displayed
    image = np.clip(image, 0, 1)
    
    ax.imshow(image)
    
    return ax


def predict(image_path, model, top_k=5):
    ''' use deep our model to Predict the class of an image. 
    
    image_path: Path to image (STRING) eg. "flowers/test/101/image_07952.jpg"
    model: pytorch neural network.
    top_k: integer. The top-K classes to be calculated; default value is 5
    
    returns top_probabilities(k), top_labels
    '''
    # CPU works better so switch to CPU
    model.to("cpu")
    
    # Set model to evaluate
    model.eval();

    # Convert image from numpy to torch
    torch_image = torch.from_numpy(np.expand_dims(process_image(image_path), axis=0)).type(torch.FloatTensor).to("cpu")

    # Find probabilities (log softmax means that its on a log scale)
    log_probs = model.forward(torch_image)

    # Converting to linear scale
    linear_probs = torch.exp(log_probs)

    # Find the top 5 prediction by model
    top_guess, top_labels = linear_probs.topk(top_k)
    
    # Detatch the details
    top_guess = np.array(top_guess.detach())[0]
    top_labels = np.array(top_labels.detach())[0]
    
    # Convert to classes
    idx_to_class = {val: key for key, val in model.class_to_idx.items()}
    top_labels = [idx_to_class[lab] for lab in top_labels]
    top_flowers = [change_cat_to_name(idd) for idd in top_labels]
    print(top_flowers)
    
    return top_guess , top_labels , top_flowers


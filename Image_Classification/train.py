## This file is written to run whole image classification process (train) from python command line
## uses: python train.py './path to image directory that contains test, train and valid sub-directories/'
#load libraries
# importing Required libraries
import my_function
import argparse
import torch

# lets pick some initial values
arch = 'vgg19'
hidden_units = 512
learning_rate = 0.001
epochs = 5
steps=0
print_every = 25

# Define basic parameter you want to use in command line
parser = argparse.ArgumentParser()
parser.add_argument('data_dir',action='store', help='provide image FOLDER path ("NOT image path")')
parser.add_argument('-arch','--arch',action='store',type=str, default='vgg19', help='Model used vgg19')
parser.add_argument('-hidden_units','--hidden_units',action='store',type=int, help='hidden units for 1st layer',
                   default = hidden_units)
parser.add_argument('-learning rate','--learning_rate',action='store',type=float, help='chose learning rate',
                   default=learning_rate)
parser.add_argument('-epochs','--epochs',action='store',type=int, help='no. of epoch',
                   default=epochs)
parser.add_argument('-save_dir','--save_dir',action='store', type=str, help='Where you wanna save the model? provide a directory name')
parser.add_argument('-gpu','--gpu',action='store_true',default=False, help='Use GPU')

args = parser.parse_args()
print("ARGS:", args)
#device = ("cuda" if results.gpu else "cpu")
#device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#model.to(device)


## Load data and transform it using my_function
data_train, dataloader_train = my_function.load_data_train(args.data_dir + "/train/")
data_test, dataloader_test = my_function.load_data_test(args.data_dir + "/test/")
data_valid, dataloader_valid = my_function.load_data_valid(args.data_dir + "/valid/")

###
# Create model
model = my_function.built_model(args.arch, True)
#if args.checkpoint:
#    model = my_function.load_my_checkpoint("./my_checkpoint.pth")
criterion = my_function.criterion()
optimizer = my_function.optimizer(model, args.learning_rate)

### Time to train the model
model = my_function.train_model(model, dataloader_train, dataloader_valid, dataloader_test, criterion, args.epochs,  optimizer, steps)
# Save the model
my_function.save_checkpoint(model, optimizer, data_train, epochs=5)
print("Succeed")





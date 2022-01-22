## This file is written to run whole image classification process (PREDICT) from python command line
## uses: python predict.py
#load libraries
import my_function  ### it contains all required functions
import argparse
import torch


parser = argparse.ArgumentParser(description='Predicting the model',)
parser.add_argument('--image_path', dest='image_path', action='store', default="./flowers/test/101/image_07952.jpg")
parser.add_argument('--checkpoint_path', dest='checkpoint_path', action='store', default='checkpoint.pth')
parser.add_argument('--top_k', dest='top_k', action='store', default=5, type=int)
parser.add_argument('--gpu', dest="mode", action="store", default="gpu")
args = parser.parse_args()

## Now lets use the functions to predict

saved_checkpoint_model = my_function.load_my_checkpoint(file_name="my_checkpoint.pth")
probs, classes, top_flowers = my_function.predict("./flowers/test/101/image_07952.jpg", saved_checkpoint_model, args.top_k)
print("Probabilities",probs)
print("Classes", classes)
print("top_flowers",top_flowers)
for i in range(args.top_k):
    print("Probability - {} \t Class - {} \t Flower -{}".format(probs[i], classes[i], top_flowers[i]))

import os 
import json
import argparse
from functions import predict,get_class_names

json_path = './label_map.json'   
image_path = './test_images/orange_dahlia.jpg'
top_k = 5
model_path = './flower_classifier.h5'

parser = argparse.ArgumentParser()

parser.add_argument('-m','--model_path',action='store',type=str)
parser.add_argument('-i','--image_path',action='store',type=str)
parser.add_argument('-k', '--top_k', action='store',type=int)
parser.add_argument('-j', '--category_names', action='store',type=str)

args = parser.parse_args()

if args.model_path:
    model_path = args.model_path
if args.image_path:
    image_path = args.image_path
if args.top_k:
    top_k = args.top_k
if args.category_names:
    json_path = args.category_names
    
   
predictions = predict(image_path,model_path,top_k)
classes = get_class_names(json_path,predictions[1])
print(classes)
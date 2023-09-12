# -*- coding: utf-8 -*-
"""M22RM007_QU3.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1Cd4A9PAkc4xo7FGuiyHciwsDDmquG-Yo

#Install Dependencies
"""

# Commented out IPython magic to ensure Python compatibility.
# clone YOLOv5 repository
!git clone https://github.com/ultralytics/yolov5  # clone repo
#change directory to yolov5
# %cd yolov5
!git reset --hard 064365d8683fd002e9ad789c1e91fa3d021b44f0

# install dependencies as necessary
!pip install -qr requirements.txt  # install dependencies 
import torch

from IPython.display import Image, clear_output  # to display images
from utils.downloads import attempt_download  # to download models/datasets

# clear_output()
print('Setup complete. Using torch %s %s' % (torch.__version__, torch.cuda.get_device_properties(0) if torch.cuda.is_available() else 'CPU'))

"""# Downloading the custom deer dataset"""

!pip install roboflow

# Paste the code snippet given by roboflow here.API key is personal to each individual
from roboflow import Roboflow
rf = Roboflow(api_key="BSh0bQp6fCTtpoWavLG2")
project = rf.workspace("iit-jodhpur-sqrtu").project("deer_detection")
dataset = project.version(5).download("yolov5")

# Commented out IPython magic to ensure Python compatibility.
# this is the YAML file Roboflow wrote for us that we're loading into this notebook with our data
# %cat {dataset.location}/data.yaml
#Check the number of classes and class name for our cas it is single class that is deer

"""# Define Model Configuration and Architecture

We will write a yaml script that defines the parameters for our model like the number of classes, anchors, and each layer.

"""

# define number of classes based on YAML
import yaml
with open(dataset.location + "/data.yaml", 'r') as stream:
    num_classes = str(yaml.safe_load(stream)['nc'])

# Commented out IPython magic to ensure Python compatibility.
#this is the model configuration we will use for our tutorial 
# %cat /content/yolov5/models/yolov5s.yaml

"""# Downloading the weights for yolov5 for pretraining and then we will finetune on deer dataset"""

!wget https://github.com/ultralytics/yolov5/releases/download/v5.0/yolov5s.pt

"""# Train Custom YOLOv5 Deer Detector


Here, we are able to pass a number of arguments:
- **img:** define input image size
- **batch:** determine batch size
- **epochs:** define the number of training epochs. (Note: often, 3000+ are common here!)
- **data:** set the path to our yaml file
- **cfg:** specify our model configuration
- **weights:** specify a custom path to weights. 
- **nosave:** only save the final checkpoint

# Training the model
"""

# Commented out IPython magic to ensure Python compatibility.
# # # train yolov5s on custom data for 100 epochs
# # # time its performance
# %%time
# %cd /content/yolov5/
# !python train.py --img 416 --batch 16 --epochs 100 --data {dataset.location}/data.yaml --cfg ./models/yolov5s.yaml --weights /content/yolov5/yolov5s.pt --name yolov5s_results --cache
#

"""# Evaluate Custom YOLOv5 Detector Performance"""

from utils.plots import plot_results  # plot results.txt as results.png
Image(filename='/content/yolov5/runs/train/yolov5s_results/results.png', width=1000)  # view results.png

"""###Visualize Our Validation Data with Labels


"""

# # first, display our ground truth data
print("GROUND TRUTH TRAINING DATA:")
Image(filename='/content/yolov5/runs/train/yolov5s_results/val_batch0_pred.jpg', width=900)

# Commented out IPython magic to ensure Python compatibility.
# trained weights are saved by default in our weights folder
# %ls runs/

# Commented out IPython magic to ensure Python compatibility.
# %ls runs/train/yolov5s_results/weights

"""# Testing the model on test images"""

# Commented out IPython magic to ensure Python compatibility.
# use the best weights
# %cd /content/yolov5/
!python detect.py --weights /content/yolov5/runs/train/yolov5s_results/weights/best.pt --img 416 --conf 0.4 --source /content/yolov5/deer_detection-5/test/images

#display inference on ALL test images
#this looks much better with longer training above

import glob
from IPython.display import Image, display

for imageName in glob.glob('/content/yolov5/runs/detect/exp2/*.jpg'): 
    display(Image(filename=imageName))
    print("\n")

"""# Now checking the performance on the dataset given for assignment 3"""

from google.colab import drive
drive.mount('/content/drive')

# Commented out IPython magic to ensure Python compatibility.
# use the best weights
# %cd /content/yolov5/
!python detect.py --weights /content/yolov5/runs/train/yolov5s_results/weights/best.pt --img 416 --conf 0.4 --source /content/drive/MyDrive/deer-test

import glob
from IPython.display import Image, display

for imageName in glob.glob('/content/yolov5/runs/detect/exp4/*.jpg'): 
    display(Image(filename=imageName))
    print("\n")
for imageName in glob.glob('/content/yolov5/runs/detect/exp4/*.png'): 
    display(Image(filename=imageName))
    print("\n")


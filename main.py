# Mabel Lu 
# Personal Proj
# CS 12
#
# This code takes images of either just rocks or people stuck in rocks and
# tries to identify wether or not there are any trapped humans amongst the
# debris.

# I don't think yolov5 was trained to do this, so it doesn't do quite that well.
# It tries it's best though.



import torch

# Model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s')

# Images
dir = 'images/'
imgs = [dir + f for f in ('rock1.jpg', 'trapped1.jpg')]  # batch of images

# Inference
results = model(imgs)
results.show()  # or .show(), .save()

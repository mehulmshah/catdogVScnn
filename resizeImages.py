# This program takes in each training image and resizes it to 224x224.
from scipy.misc import imread,imsave
import glob
import tensorflow as tf

trainImages = []
trainNames = []
arr = []
new_image_size = [224,224]
for filename in glob.glob('train/*.jpg'): #assuming gif
    im=imread(filename)
    trainImages.append(im)
    trainNames.append(filename)

sess = tf.Session()

for i,im in enumerate(trainImages):
    tensor = tf.pack(im)
    resized = tf.image.resize_images(tensor,new_image_size,method=1)
    arr.append(resized)

with sess.as_default():
    for i,a in enumerate(arr):
        output = a.eval()
        imsave(trainNames[i],output)

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: fernandoduarte
"""
import os
from derm_ita import get_ita
from PIL import Image

import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imread, imshow

from skimage import img_as_ubyte

def ground_truth(image, patch, mode='mean'):   
   """
   White balance image using Ground-truth algorithm
   Parameters
   ----------
   image : numpy array
           Image to white balancr
   patch : numpy array
           Patch of "true" white
   mode : mean or max, optional
          Adjust mean or max of each channel to match patch
  
   Returns
   -------
   
   image_wb : numpy array
              White-balanced image
   """
   image_patch = img_patch
   if mode == 'mean':
       image_gt = ((image * (image_patch.mean() / \
                   image.mean(axis=(0, 1))))\
                   .clip(0, 255)\
                   .astype(int))
   if mode == 'max':
       image_gt = ((image * 1.0 / image_patch.max(axis
                    (0,1))).clip(0, 1))
   #transparency channel
   if image.shape[2] == 4:
       image_gt[:,:,3] = 255
return image_gt


def percentile_whitebalance(image, percentile_value):
    fig, ax = plt.subplots(1,2, figsize=(12,6))
    for channel, color in enumerate('rgb'):
            channel_values = image[:,:,channel]
            value = np.percentile(channel_values, percentile_value)
            ax[0].step(np.arange(256), 
                       np.bincount(channel_values.flatten(), 
                       minlength=256)*1.0 / channel_values.size, 
                       c=color)
            ax[0].set_xlim(0, 255)
            ax[0].axvline(value, ls='--', c=color)
            ax[0].text(value-70, .01+.012*channel, 
                       "{}_max_value = {}".format(color, value), 
                        weight='bold', fontsize=10)
            ax[0].set_xlabel('channel value')
            ax[0].set_ylabel('fraction of pixels');
            ax[0].set_title('Histogram of colors in RGB channels')    
            whitebalanced = img_as_ubyte(
                    (image*1.0 / np.percentile(image, 
                     percentile_value, axis=(0, 1))).clip(0, 1))
            ax[1].imshow(whitebalanced);
            ax[1].set_title('Whitebalanced Image')
    return whitebalanced #ax

def gray_world(image):
    """
    White balance image using Gray-world algorithm
    Parameters
    ----------
    image : numpy array
            Image to white balance
    
    Returns
    -------
    image_wb : numpy array   
               White-balanced image
    """
    image_grayworld = ((image * (image.mean() / 
                      image.mean(axis=(0,1)))).
                      clip(0,255).astype(int))
    # for images having a transparency channel
    
    if image.shape[2] == 4:
        image_grayworld[:,:,3] = 255
    return image_grayworld
#call the function to implement gray world algorithm



def calculate_ita(file_path):
    # This function should perform the necessary task on the file
    # For demonstration, let's say it returns a float (e.g., file size in KB)
    whole_image_ita = get_ita(image=Image.open(file_path))
    return whole_image_ita

def main(directory, output_file):
    results = []

    # Iterate through all files in the specified directory
    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)

        # Check if it's a file (and not a directory)
        if os.path.isfile(file_path):
            result = calculate_ita(file_path)
            results.append((filename, result))

        print(result)

    # Write the results to a text file
    with open(output_file, 'w') as f:
        for filename, result in results:
            f.write(f"{filename}: {result}\n")

if __name__ == "__main__":
    path = os.getcwd()
    folder = path + "/ISIC-images/"
    image = folder + "ISIC_0079358.jpg"
    print(image)
    test_img = Image.open(image)
    test_img_np = np.array(test_img)
    test_img_np_gray = gray_world(test_img_np)
    
    test_img_np_whiteB = percentile_whitebalance(test_img_np,100)

    # Create a figure with two subplots
    fig, axs = plt.subplots(1, 3, figsize=(6, 3))
    
    # Display the original image in the first subplot
    axs[0].imshow(test_img_np)
    axs[0].set_title('Original Image')
    axs[0].axis('off')  # Hide the axis

    # Display the gray-world image in the second subplot
    axs[1].imshow(test_img_np_gray)
    axs[1].set_title('Gray-world Image')
    axs[1].axis('off')  # Hide the axis
    
    # Display the gray-world image in the second subplot
    axs[2].imshow(test_img_np_whiteB)
    axs[2].set_title('WhiteBalance Image')
    axs[2].axis('off')  # Hide the axis

    # Show the plot
    plt.tight_layout()
    plt.show()
    
    #percentile_whitebalance(test_img_np,90)
    
    
    #imshow(test_img_np)
    
    #imshow(gray_world(test_img_np))
    #image_grey = gray_world(test_img)



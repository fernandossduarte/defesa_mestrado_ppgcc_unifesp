#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: fernandoduarte
"""
import os
from derm_ita import get_ita
from PIL import Image

import numpy as np
import math
import matplotlib.pyplot as plt
from skimage.filters import threshold_otsu
from skimage.color import rgb2lab

from datetime import datetime




def calculate_ita(file_path):
    # This function should perform the necessary task on the file
    # For demonstration, let's say it returns a float (e.g., file size in KB)
    whole_image_ita = get_ita(image=Image.open(file_path))
    #whole_image_ita = get_ita(image)
    return whole_image_ita

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
                      image.mean(axis=(0, 1)))).
                      clip(0, 255).astype(np.uint8))
    
    # For images having a transparency channel
    if image.shape[2] == 4:
        image_grayworld[:, :, 3] = 255
    return image_grayworld

def compute_ita_from_lab(image):
    """
    This function computes the ITA value of the image along with ignore any pixels that are completely black
    :param lab_image:  input image formatted in LAB color space
    :return: the ITA value from the input image
    """
    lab_image = rgb2lab(image)
    
    # get the luminance and b values wihtin +- 1 std from mean
    l = lab_image[:, :, 0]
    l = np.where(l != 0, l, np.nan)
    std = np.nanstd(l)
    mean = np.nanmean(l)

    l = np.where(l >= mean - std, l, np.nan)
    l = np.where(l <= mean + std, l, np.nan)

    b = lab_image[:, :, 2]
    std = np.nanstd(b)
    mean = np.nanmean(b)
    b = np.where(b >= mean - std, b, np.nan)
    b = np.where(b <= mean + std, b, np.nan)

    ita = math.atan2(np.nanmean(l) - 50, np.nanmean(b)) * (180 / np.pi)
    return ita

def skin_classification(ita):
    
    classification = ""
    
    if ita > 55:
        classification = "I"
        return classification
    if 55 >= ita > 42:
        classification = "II"
        return classification
    if 42 >= ita > 35:
        classification = "III"
        return classification
    if 35 >= ita > 29:
        classification = "IV"
        return classification
    if 29 >= ita > 21:
        classification = "V"
        return classification
    if 21 >= ita:
        classification = "VI"
        return classification
    
    
    return classification

def skin_tone(ita):
    
    tone = ""
    
    if ita > 45:
        tone = "light"
        return tone
    if 45 >= ita:
        tone = "dark"
        return tone    
    
    return tone

def is_valid_image_pillow(file_name):
    try:
        with Image.open(file_name) as img:
            img.verify()
            return True
    except (IOError, SyntaxError):
        return False

def main(directory, output_file):
    results = []
    
    # Iterate through all files in the specified directory
    for filename in sorted(os.listdir(directory)):
        file_path = os.path.join(directory, filename)
        
        # Check if it's a file (and not a directory)
        if os.path.isfile(file_path) and is_valid_image_pillow(file_path):
            
            # Read the image
            input_img = Image.open(file_path)
            input_img_np = np.array(input_img)
            
            # Apply the gray-world algorithm
            input_img_np_gray = gray_world(input_img_np)
            
            #CHOSEN COLOR SCHEME
            IMG_CHOSEN = input_img_np_gray
            
            # Convert to grayscale if the image has multiple channels(ORIGINAL)
            input_to_greyscale = IMG_CHOSEN
            if input_to_greyscale.ndim == 3:
                input_to_greyscale = input_to_greyscale.mean(axis=2).astype(np.uint8)
            
            # Apply Otsu's threshold (ORIGINAL)
            thresh = threshold_otsu(input_to_greyscale)
            binary = input_to_greyscale > thresh
            
            # Apply the binary mask to the original image
            masked_image = np.zeros_like(IMG_CHOSEN)
            for i in range(3):  # Assuming RGB image
                masked_image[:,:,i] = IMG_CHOSEN[:,:,i] * binary
            
            #ita = math.floor(compute_ita_from_lab(masked_image))
            #ita = math.floor(compute_ita_from_lab(masked_image))
            ita = math.floor(calculate_ita(file_path))
            
            fitzpatric_type = skin_classification(ita)
            
            tone = skin_tone(ita)
            
            results.append((filename, ita, fitzpatric_type, tone))
            
            print(f"{filename}")
        

            """
            # Create subplots
            fig, axes = plt.subplots(ncols=4, figsize=(10, 4))
            ax = axes.ravel()
            
            ax[0].imshow(IMG_CHOSEN, cmap=plt.cm.gray)
            #ax[0].imshow(binary, cmap=plt.cm.gray, alpha=0.5)
            ax[0].set_title('Original')
            ax[0].set_xlabel(f"ITA: {ita}")
            ax[0].axis('on')
            
            ax[1].hist(input_to_greyscale.ravel(), bins=256)
            ax[1].set_title('Histogram')
            ax[1].axvline(thresh, color='r')
            
            ax[2].imshow(binary, cmap=plt.cm.gray)
            ax[2].set_title('Thresholded')
            ax[2].axis('off')
            
            ax[3].imshow(masked_image, cmap=plt.cm.gray)
            ax[3].set_title('Masked')
            ax[3].axis('off')
            
            plt.tight_layout()
            plt.show()
            #"""
            
            #print(f"ITA: {ita}\n")

        #print(results)
        
        

    # Write the results to a text file
    with open(output_file, 'w') as f:
        for filename, ita, fitzpatric_type, tone in results:
            f.write(f"{filename},{ita},{fitzpatric_type},{tone}\n")

if __name__ == "__main__":
    start_time = datetime.now()
    path = os.getcwd()
    #folder = path + "/ISIC-images/"
    folder = path + "/HAM10000 Dataset/dataverse_files/HAM10000_ALL/"
    #folder = path + "/Test_images_paper/"
    
    directory = folder  # replace with your folder path
    output_file = 'results_dermITA_HAM10000.txt'  # replace with your desired output file path
    main(directory, output_file)
    end_time = datetime.now()
    
    elapsed_time = end_time - start_time
    print(f"Elapsed time: {elapsed_time}")

################################################################################
#Zero padd, resize the images,and save images in images_resize
#################################################################################


# Check the image (greyscale or color)
image_path = "/local/data1/yanwa579/Data/imageswithextraAFF_anonymized_8bit/patient_AAJEHHHPOU_NFF_image_1.png" 
import os
#Install Pillow
from PIL import Image


# Load the image from folder
image_test = Image.open(image_path)
# Convert mode to check grayscale
if image_test.mode == "L":  # 'L' mode means grayscale in Pillow
    print("The image is grayscale.")

#install OpenCV

import os
import cv2


input_folder = '/local/data1/yanwa579/Data/imageswithextraAFF_anonymized_8bit'
#Count the number of images in the input folder
print(len(os.listdir(input_folder)))
#Create a folder manually in VS Code to store the resized images
output_folder = '/local/data1/yanwa579/Data/images_resize'

# Function to zero-pad and resize an image
def zeropad_and_resize(image):
    h, w = image.shape
    #get the largest dimention
    max_dimention = max(h, w)
    # Calculate padding for width and height
    zeropad_top = (max_dimention - h) // 2
    zeropad_bottom = max_dimention - h - zeropad_top
    zeropad_left = (max_dimention - w) // 2
    zeropad_right = max_dimention - w - zeropad_left
    # Add borders to the image
    zeropadd_image = cv2.copyMakeBorder(image, zeropad_top, zeropad_bottom, zeropad_left, zeropad_right, cv2.BORDER_CONSTANT, value=0)
    # Resize to 256x256
    resized_image = cv2.resize(zeropadd_image, (256,256))
    return resized_image

#Use os.listdir to obtain a list of images' names
for filename in os.listdir(input_folder):
    #Get the image path:   /local/data1/yanwa579/Data/../patient...png
    image_path = os.path.join(input_folder, filename)   
    # Read the image
    image = cv2.imread(image_path,cv2.IMREAD_GRAYSCALE)
    print(image.shape)
    image_resized = zeropad_and_resize(image)
    print(image_resized.shape)
    # Set output path
    output_path = os.path.join(output_folder, filename) 
    # Save the processed image
    cv2.imwrite(output_path, image_resized)  

#Count the number of images in a folder
print(len(os.listdir(input_folder)))
print(len(os.listdir(output_folder)))


################################################################################
#Split AFF and NFF and save them in AFF_resize and NFF_resize folders
#################################################################################

import shutil

# Define source and destination directories
#input_folder = '/local/data1/yanwa579/Data/imageswithextraAFF_anonymized_8bit'
source_dir = "/local/data1/yanwa579/Data/images_resize"  # Change to your actual path
#Create two folders manually in VS Code to store the resized AFF and NFF images
AFF_folder = "/local/data1/yanwa579/Data/AFF_resize"
NFF_folder = "/local/data1/yanwa579/Data/NFF_resize"

# Loop through all images in the input folder
for image in os.listdir(source_dir):
    if "_AFF_" in image:
    #copy the image from the input file into output file
        shutil.copy(os.path.join(source_dir, image), os.path.join(AFF_folder, image))
    elif "_NFF_" in image:
        shutil.copy(os.path.join(source_dir, image), os.path.join(NFF_folder, image))

#Count the number of images in a folder
print(len(os.listdir(AFF_folder)))
print(len(os.listdir(NFF_folder)))

##############################################
#Split the data into three dataset: AFF_train,AFF_valid,AFF_test
#NFF_train,NFF_valid,test,ratios are 70%,15%,15% respectively
##############################################


# Define the source folder where images are stored
AFF_source_folder = "/local/data1/yanwa579/Data/AFF_resize"
NFF_source_folder = "/local/data1/yanwa579/Data/NFF_resize"

# Define output folders for train, validation, and test
AFF_train_folder = "/local/data1/yanwa579/Data/AFF_train"
AFF_val_folder = "/local/data1/yanwa579/Data/AFF_valid"
AFF_test_folder = "/local/data1/yanwa579/Data/AFF_test"

NFF_train_folder = "/local/data1/yanwa579/Data/NFF_train"
NFF_val_folder = "/local/data1/yanwa579/Data/NFF_valid"
NFF_test_folder = "/local/data1/yanwa579/Data/NFF_test"

def spit_images(source_folder,train_folder,val_folder,test_folder):
    # Get all images'names
    image_files =[f for f in os.listdir(source_folder)]
    #Extract unique patient name from the image names
    #Get the patient names like:{'DCAUHSLHGH': ['patient_DCAUHSLHGH_AFF_image_1.png']}
    patient_names = {}
    for image in image_files:
        # Split the name based on _ 
        parts = image.split('_')
        # Filename format like: patient_id_AFF_image_1.png
        patient_name = parts[1]
        if patient_name not in patient_names:
            patient_names[patient_name] = []
        patient_names[patient_name].append(image)

    # Compute split sizes
    total_patients = len(patient_names)
    train_count = int(total_patients * 0.7)
    val_count = int(total_patients * 0.15)
    # Ensure all patients are used
    test_count = total_patients - (train_count + val_count)
    # Get all the unique patient names
    patient_list = list(patient_names.keys())
    #Split the patients base on the ratio
    train_patients = patient_list[:train_count]
    val_patients = patient_list[train_count:train_count + val_count]
    test_patients = patient_list[train_count + val_count:]
    # Function to copy images based on patient names
    def copy_patient_images(patient_list, output_folder):
        for patient in patient_list:
            # Get all images for this patient
            patient_images = patient_names[patient]
            for image in patient_images:
                shutil.copy(os.path.join(source_folder, image), os.path.join(output_folder, image))

    # Copy images for each dataset
    copy_patient_images(train_patients, train_folder)
    copy_patient_images(val_patients, val_folder)
    copy_patient_images(test_patients, test_folder)

    print(f"Total patients: {total_patients}")
    print(f"Train patients: {len(train_patients)}, Val patients: {len(val_patients)}, Test patients: {len(test_patients)}")
spit_images(AFF_source_folder,AFF_train_folder,AFF_val_folder,AFF_test_folder)
print(f"AFF Train : {len(os.listdir(AFF_train_folder ))}, AFF Valid: {len(os.listdir(AFF_val_folder ))}, AFF Test: {len(os.listdir(AFF_test_folder ))}")
spit_images(NFF_source_folder,NFF_train_folder,NFF_val_folder,NFF_test_folder)
print(f"NFF Train : {len(os.listdir(NFF_train_folder ))}, NFF Valid: {len(os.listdir(NFF_val_folder ))}, NFF Test: {len(os.listdir(NFF_test_folder ))}")

#############
# From results obtained above:247 patients in AFF, 914 patients in NFF, in total: 1161 patiens
#############

#Test the number of patiens in the imageswithextraAFF_anonmized_8bit

source_folder = '/local/data1/yanwa579/Data/imageswithextraAFF_anonymized_8bit'
# Get all images'names
image_files =[f for f in os.listdir(source_folder)]
#Extract unique patient name from the image names
patient_names = []
for image in image_files:
    parts = image.split('_')
    patient_name = parts[1]
    if patient_name not in patient_names:
        patient_names.append(patient_name)
#print(patient_names)
print(f"Total patients: {len(patient_names)}")





##############
##test if the image is greyscale
#############


import cv2
import os
image_path = "/local/data1/yanwa579/Data/imageswithextraAFF_anonymized_8bit/patient_AAJEHHHPOU_NFF_image_1.png" 

image = cv2.imread(image_path)
print(image.shape)


import os
from PIL import Image


# Load the image from folder
image = Image.open(image_path)
# Convert mode to check grayscale
if image.mode == "L":  # 'L' mode means grayscale in Pillow
    print("The image is grayscale.")


import cv2

# Load the image in grayscale mode (single-channel)
img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

# Check the shape
print(img.shape)



import cv2
from PIL import Image
# Load the image from folder
image_path = "/local/data1/yanwa579/Data/AFF_train/patient_AOOBKUOKNK_AFF_image_1.png"
image = Image.open(image_path)
# Convert mode to check grayscale
if image.mode == "L":  # 'L' mode means grayscale in Pillow
    print("grayscale.")
img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
print(img.shape)

###################
#Count the number of patients
#draw the histogram show how many patient own how mang images
###################
import os
# Define the source folder where images are stored
source_folder = "/local/data1/yanwa579/Data/imageswithextraAFF_anonymized_8bit"
# Get all images'names
image_files =[f for f in os.listdir(source_folder)]
#Extract unique patient name from the image names
#Get the patient names like:{'DCAUHSLHGH': ['patient_DCAUHSLHGH_AFF_image_1.png']}
patient_names = {}
for image in image_files:
    # split the name based on _ 
    parts = image.split('_')
    # filename format like: patient_id_AFF_image_1.png
    patient_name = parts[1]
    if patient_name not in patient_names:
        patient_names[patient_name] = []
    patient_names[patient_name].append(image)
# Compute split sizes
total_patients = len(patient_names)
# Get all the unique patient names
patient_list = list(patient_names.keys())
patients = {}
for patient in patient_list:
    patients[patient] = len(patient_names[patient])

print(f"Total patients: {total_patients}")
print(patients)

import matplotlib.pyplot as plt

values = list(patients.values())
plt.hist(values, bins=9,color='blue', edgecolor='black', alpha=0.7, density=True)

plt.xlabel("the number of images per patient")
plt.ylabel("Frequency")
plt.show()
max(patients.values())

############
#draw the image size distribution
############

import cv2
widths = []
heights = []
#use os.listdir to obtain a list of images' names
for filename in os.listdir(source_folder):
    #get the image path:   /local/data1/yanwa579/Data/../patient...png
    image_path = os.path.join(source_folder, filename)   
    # Read the image
    image = cv2.imread(image_path,cv2.IMREAD_GRAYSCALE)
    width, height = image.shape
    widths.append(width)
    heights.append(height)
print(np.min(widths))
print(np.max(widths))
print(np.min(heights))
print(np.max(heights))


plt.scatter(widths, heights, s=2, color='blue')
plt.xlabel("Width (pixels)")
plt.ylabel("Height (pixels)")
plt.title("Scatter Plot of Image Dimensions")
plt.grid(True)
plt.show()
import cv2
import os

# Set input and output folders
input_folder = '/proj/afraid/users/x_wayan/Data/NFF_test'
output_folder = '/proj/afraid/users/x_wayan/Data/NFF_test_color'
os.makedirs(output_folder, exist_ok=True)

# Process each PNG image
for filename in os.listdir(input_folder):
    input_path = os.path.join(input_folder, filename)
    output_path = os.path.join(output_folder, filename)

    # Read image in grayscale
    gray_img = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)
    # Resize to 64x64
    #gray_img = cv2.resize(gray_img, (64, 64), interpolation=cv2.INTER_AREA)
    # Replicate to 3 channels
    new_img = cv2.merge([gray_img, gray_img, gray_img])

    # Save image
    cv2.imwrite(output_path, new_img)
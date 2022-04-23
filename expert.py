import os
import cv2 as cv
import copy
from skimage import io
import numpy as np
from skimage.color import rgb2gray
from skimage.feature import hog
from sklearn.model_selection import train_test_split
from sklearn import svm

# import more libraries as you need
from face_landmark_detection import getface

# T1  start _______________________________________________________________________________
# Read in Dataset

# change the dataset path here according to your folder structure
dataset_path = "./Dataset_1"
X = []
y = []

for subject_name in os.listdir(dataset_path):
    y.append(subject_name)
    subject_images_dir = os.path.join(dataset_path, subject_name)

    temp_x_list = []

    for img_name in os.listdir(subject_images_dir):
        img = cv.imread(os.path.join(subject_images_dir, img_name), 0)
        img = cv.equalizeHist(img)
        # cv.imshow('img', img)
        # cv.waitKey()
        temp_x_list.append(img)
        # add the img to temp_x_list
    X.append(temp_x_list)
    # add the temp_x_list to X

print(np.array(temp_x_list).shape)  # verify the shape of array
print(np.array(X).shape)
# T1 end ____________________________________________________________________________________

# T2 start __________________________________________________________________________________
# Preprocessing
X_processed = []  # X_processed: without mask
X_masked = []  # X_masked: with mask
for element_1, x_list in enumerate(X):
    temp_X_processed = []
    temp_X_masked = []
    for element_2, x in enumerate(x_list):
        # write the code to detect face in the image (x) using dlib facedetection library
        left, top, right, bottom, mask_region = getface('shape_predictor_68_face_landmarks.dat', x)

        # write the code to crop the image (x) to keep only the face, resize the cropped image to 150x150
        mask_region = np.array(mask_region)
        cropped_x = copy.deepcopy(x)
        cv.fillPoly(cropped_x, mask_region, (255))
        top_1 = max(0, top - 100)
        bottom_1 = min(480, bottom + 100)
        left_1 = max(0, left - 100)
        right_1 = min(640, right + 100)
        if bottom_1 - top_1 != right_1 - left_1:
            right_1 = left_1 + (bottom_1 - top_1)
        cropped_x = cropped_x[top_1:bottom_1, left_1:right_1]
        x = x[top_1:bottom_1, left_1:right_1]
        cropped_x = cv.resize(cropped_x, (150, 150))
        x = cv.resize(x, (150, 150))

        # cropped_x=cv.equalizeHist(rgb2gray(cropped_x))
        # x=cv.equalizeHist(rgb2gray(x))
        # cropped_x = rgb2gray(cropped_x)
        # print(cropped_x)
        # cv.equalizeHist(cropped_x, cropped_x)
        # x = rgb2gray(x)

        temp_X_masked.append(cropped_x)
        temp_X_processed.append(x)

        path = "C:\\Users\\huang\\Desktop\\nus_project\\Dataset_2_processed\\s"
        if element_1 + 1 < 10:
            path += "0"
        path = path + str(element_1 + 1)
        if not os.path.exists(path):
            os.makedirs(path)
        path += "\\"
        if element_2 + 1 < 10:
            path += "0"
        path = path + str(element_2 + 1) + ".jpg"
        print(path)
        cv.imwrite(path, cropped_x)

        path = "C:\\Users\\huang\\Desktop\\nus_project\\Dataset_2\\s"
        if element_1 + 1 < 10:
            path += "0"
        path = path + str(element_1 + 1)
        if not os.path.exists(path):
            os.makedirs(path)
        path += "\\"
        if element_2 + 1 < 10:
            path += "0"
        path = path + str(element_2 + 1) + ".jpg"
        print(path)
        cv.imwrite(path, x)

        # append the converted image into temp_X_processed
    X_masked.append(temp_X_masked)
    X_processed.append(temp_X_processed)
    # append temp_X_processed into  X_processed

# T2 end ____________________________________________________________________________________

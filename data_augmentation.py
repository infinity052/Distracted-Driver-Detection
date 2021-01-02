import cv2
import glob
import os

from sklearn.cross_validation import KFold
os.mkdir("augmented_dataset")
os.mkdir("augmented_dataset/flipped")

for i in range(0,10):
    os.mkdir("augmented_dataset/flipped/c%d" %i)
    inputFolder = "./dataset/imgs/train/c%d" %i
    folderLen = len(inputFolder)
    for img in glob.glob(inputFolder + "/*.jpg"):
        image = cv2.imread(img)
        imgResized = cv2.flip(image, 1)
        cv2.imwrite("augmented_dataset/flipped/c%d" %i + "/" + img[folderLen:] , imgResized)

os.mkdir("augmented_dataset/blurred")
for i in range(0,10):
    os.mkdir("augmented_dataset/blurred/c%d" %i)
    inputFolder = "./dataset/imgs/train/c%d" %i
    folderLen = len(inputFolder)
    for img in glob.glob(inputFolder + "/*.jpg"):
        image = cv2.imread(img)
        imgResized = cv2.resize(image,None, fx=0.2, fy=0.2, interpolation = cv2.INTER_CUBIC)
        cv2.imwrite("augmented_dataset/blurred/c%d" %i + "/" + img[folderLen:], imgResized);

os.mkdir("augmented_dataset/blurred_grayscale")
for i in range(0,10):
    os.mkdir("augmented_dataset/blurred_grayscale/c%d" %i)
    inputFolder = "./augmented_dataset/blurred/c%d" %i
    folderLen = len(inputFolder)
    for img in glob.glob(inputFolder + "/*.jpg"):
        image = cv2.imread(img)
        imgResized = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        cv2.imwrite("augmented_dataset/blurred_grayscale/c%d" %i + "/" + img[folderLen:], imgResized);

os.mkdir("augmented_dataset/blurred_flipped_grayscale")
for i in range(0,10):
    os.mkdir("augmented_dataset/blurred_flipped_grayscale/c%d" %i)
    inputFolder = "./augmented_dataset/blurred_grayscale/c%d" %i
    folderLen = len(inputFolder)
    for img in glob.glob(inputFolder + "/*.jpg"):
        image = cv2.imread(img)
        imgResized = cv2.flip(image, 1)
        cv2.imwrite("augmented_dataset/blurred_flipped_grayscale/c%d" %i + "/" + img[folderLen:], imgResized);

cv2.destroyAllWindows()
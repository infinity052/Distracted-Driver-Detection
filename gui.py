from tkinter import *
from tkinter.filedialog import askopenfilename
from PIL import ImageTk, Image
import cv2
import os
import numpy as np
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Flatten
from keras.layers.convolutional import Conv2D, MaxPooling2D, \
                                       ZeroPadding2D




print("Loading model please wait")
model = Sequential()

model.add(Conv2D(filters=64, kernel_size=2, padding='same', activation='relu', input_shape=(64,64,3), kernel_initializer='glorot_normal'))
model.add(MaxPooling2D(pool_size=2))
model.add(Conv2D(filters=128, kernel_size=2, padding='same', activation='relu', kernel_initializer='glorot_normal'))
model.add(MaxPooling2D(pool_size=2))
model.add(Conv2D(filters=256, kernel_size=2, padding='same', activation='relu', kernel_initializer='glorot_normal'))
model.add(MaxPooling2D(pool_size=2))
model.add(Conv2D(filters=512, kernel_size=2, padding='same', activation='relu', kernel_initializer='glorot_normal'))
model.add(MaxPooling2D(pool_size=2))
model.add(Dropout(0.5))
model.add(Flatten())
model.add(Dense(500, activation='relu', kernel_initializer='glorot_normal'))
model.add(Dropout(0.5))
model.add(Dense(10, activation='softmax', kernel_initializer='glorot_normal'))
model.load_weights(os.path.join('.','cache','model','distracted-09-1.00.hdf5'))
model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
print("Model successfully loaded")
def load_img(path):
    img = cv2.imread(path)
    resized = cv2.resize(img, (64, 64))
    return resized

def normalize_img(img):
    img_array = np.array(img, dtype=np.uint8)
    img_array = img_array.astype('float32')
    mean_pixel = [103.-939, 116.779, 123.68]
    for c in range(3):
        img_array[:, :, :, c] = img_array[:, :, :, c] - mean_pixel[c]
    img_array /= 255
    return img_array

def uploadImage():
    filename = askopenfilename()
    return filename

def getPrediction(num):
    prediction = "Prediction : "
    if num == 0:
        prediction+="Driver Focussed"
    elif num == 1 or num == 3:
        prediction+="Driver is Texting"
    elif num == 2 or num == 4:
        prediction += "Driver is on the phone"
    elif num == 5:
        prediction += "Driver is operating the radio"
    elif num == 6:
        prediction += "Driver is drinking"
    elif num == 7:
        prediction += "Driver is reaching behind"
    elif num == 8:
        prediction += "Driver is looking in the mirror"
    else:
        prediction += "Driver is talking to the passenger"
    return prediction


def display_result(predicted_class, path):
    for widget in frame.winfo_children():
       widget.destroy()    
    img = ImageTk.PhotoImage(Image.open(path))
    imgLabel = Label(frame , image = img)
    imgLabel.image = img
    imgLabel.pack()
    
    txt = StringVar()
    ans = Label(frame, textvariable = txt)
    txt.set(getPrediction(int(predicted_class[0])))
    ans.pack()

    

def predict():
    
    path = uploadImage()
    if path == '':
        return
    root.title('Please wait...')
    img = load_img(path)
    array = [img]
    img_array = normalize_img(array)
    predicted_class = model.predict_classes(img_array)
    display_result(predicted_class, path)
    root.title('Distracted Driver Detection')
    return

root = Tk()
root.title('Distracted Driver Detection')
root.minsize(400, 50) 
frame = Frame(root)
frame.pack(fill = BOTH, expand = True)

btn = Button(root, text = "Choose image" , command = predict)
btn.pack()

root.mainloop()


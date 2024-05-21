import tensorflow as tf
import numpy as np
import os
from PIL import ImageOps,Image
import cv2


def predict_with_model(model,img_path):
    
    if not isinstance(img_path, str):
        size = (60,60)    
        image = ImageOps.fit(img_path, size, Image.AFFINE)
        image = np.asarray(image)
        img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        #img_resize = (cv2.resize(img, dsize=(75, 75),    interpolation=cv2.INTER_CUBIC))/255.
        
        image = img[np.newaxis,...]
    else:

        # Check if file exists
        if not os.path.isfile(img_path):
            raise FileNotFoundError(f"No such file: '{img_path}'")
        image = tf.io.read_file(img_path)
        image = tf.image.decode_png(image,channels=3)
        # will scale image from 0 to 255 -> 0 to 1
        image = tf.image.convert_image_dtype(image,dtype=tf.float32)
        image = tf.image.resize(image,[60,60]) #(60,60,3)
        image  = tf.expand_dims(image,axis=0)# (1,60,60,3)

    predictions = model.predict(image) #[0.005,0.00003,0.99,0...]
    score = tf.nn.softmax(predictions[0])
    predictions = np.argmax(predictions) # 2

    return predictions,score


if __name__ =="__main__":
    # img_path= "D:\\Projects Computer Vision\\Computer vision Project 2 (GTSRB)\\GTSRB\\archive\\Trainning_data\\val\\2\\00002_00001_00011.png"
    # img_path= "D:\\Projects Computer Vision\\Computer vision Project 2 (GTSRB)\\GTSRB\\archive\\Trainning_data\\val\\8\\00008_00001_00022.png"

    model = tf.keras.models.load_model("Models.keras")
    # prediction = predict_with_model(model,img_path)



    # print(f"prediction ={prediction} ")
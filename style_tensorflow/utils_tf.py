from PIL import Image
import numpy as np
from tensorflow.python.keras.preprocessing import image as kp_image
import tensorflow as tf


def load_img(path_to_img):
    max_dim = 512
    img = Image.open(path_to_img)
    img_size = max(img.size)
    scale = max_dim/img_size
    img = img.resize((round(img.size[0]*scale),
                      round(img.size[1] * scale)), Image.ANTIALIAS)
    img = kp_image.img_to_array(img)
    # We need to broadcast the image array such that it has a batch dimension
    img = np.expand_dims(img, axis=0)

    # preprocess raw images to make it suitable to be used by VGG19 model
    out = tf.keras.applications.vgg19.preprocess_input(img)

    return tf.convert_to_tensor(out)


def deprocess_img(processed_img):
    x = processed_img.copy()
    # perform the inverse of the preprocessiing step
    x[:, :, 0] += 103.939
    x[:, :, 1] += 116.779
    x[:, :, 2] += 123.68
    x = x[:, :, ::-1]
    x = np.clip(x, 0, 255).astype('uint8')
    return x

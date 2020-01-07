import numpy as np
from keras import backend as K # keras version 2.2.4
from keras.applications import vgg19 # for now we only use vgg19
#from scipy.misc import imsave
from scipy.optimize import fmin_l_bfgs_b

#from utils import preprocess_image, deprocess_image

def content_loss(base, output):
    """
    Compute the content loss for style transfer.
    Inputs:
    - base: features of the content image, Tensor with shape [height, width, channels]
    - output: features of the generated image, Tensor with shape[H,W,C]

    Returns:
    - Scalar of content loss
    """

    return K.sum(K.square(output - base))

def test_content_loss():
    np.random.seed(1)
    base = np.random.randn(10,10,3)
    output = np.random.randn(10,10,3)
    a = K.constant(base)
    b = K.constant(output)
    test = content_loss(a, b)
    print('Result: ', K.eval(test))
    print('Expected result: ', 605.6219)

def gram_matrix(x):
    """
    Computes the outer-product of the input tensor x.

    Input:
    - x: input tensor of shape [H,W,C]

    Returns:
    Tensor of shape [C,C] corresponding to the Gram matrix
    """
    features = K.batch_flatten(K.permute_dimensions(x, (2,0,1)))
    return K.dot(features, K.transpose(features))

def test_gram_matrix():
    np.random.seed(1)
    x_np = np.random.randn(10,10,3)
    x = K.constant(x_np)
    test = gram_matrix(x)
    print('Result:\n', K.eval(test))
    print('Expected:\n', np.array([[99.75723, -9.96186, -1.4740534],
    [-9.96186, 86.854324, -4.141108 ], 
    [-1.4740534, -4.141108, 82.30106  ]]))

def style_loss(base, output):
    """
    Computes the style reconstruction loss.

    Inputs:
    - base: features at given layer of the style image.
    - output: features of the generated image.

    Returns:
    - style_loss: scalar style loss
    """
    H, W = int(base.shape[0]), int(base.shape[1])
    gram_base = gram_matrix(base)
    gram_output = gram_matrix(output)
    factor = 1.0 / float((2*H*W)**2) #factorised implementation of equation
    out = factor * K.sum(K.square(gram_output - gram_base))
    return out

def test_style_loss():
    np.random.seed(1)
    x = np.random.randn(10,10,3)
    y = np.random.randn(10,10,3)
    a = K.constant(x)
    b = K.constant(y)
    test = style_loss(a, b)
    print('Result:  ', K.eval(test))
    print('Expected:', 0.09799164)

def total_variation_loss(x):
    """
    Total variation loss encourages smoothness in the image.
    Acts as a regularizer

    Inputs:
    - x: image with pixels 1 x H x W x C

    Returns:
    - total variation loss
    """

    a = K.square(x[:, :-1, :-1, :] - x[:, 1:, :-1, :])
    b = K.square(x[:, :-1, :-1, :] - x[:, :-1, 1:, :])
    return K.sum(a + b)

def test_variation_loss():
    np.random.seed(1)
    x_np = np.random.randn(1,10,10,3)
    x = K.constant(x_np)
    test = total_variation_loss(x)
    print('Result:  ', K.eval(test))
    print('Expected:', 937.0538)
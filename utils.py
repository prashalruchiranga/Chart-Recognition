import os
import cv2
import matplotlib.pyplot as plt
from copy import deepcopy
import operator

def annotshow(image, annotation, radius=5, tl_color=(255, 0, 0), br_color=(0, 0, 255), thickness=-1):
    image_copy = deepcopy(image)
    for dic in annotation:
        x0, y0, width, height = [int(x) for x in dic['bbox']]
        cv2.circle(image_copy, (x0, y0), radius, tl_color, thickness)
        cv2.circle(image_copy, (x0+width, y0+height), radius, br_color, thickness)
    fig, ax = plt.subplots(1)
    ax.imshow(image_copy)
    ax.axis('off')
    plt.show()


def scale(annotation, input_shape, output_shape):
    deep_copy = deepcopy(annotation)
    a, b = input_shape
    c, d = output_shape
    sx = d / b
    sy = c / a
    for dic in deep_copy:
        dic['bbox'] = list(map(operator.mul, dic['bbox'], [sy, sx, sy, sx]))
    return deep_copy


def resize_image(file, new_shape):
    rgb_img = cv2.cvtColor(cv2.imread(file), cv2.COLOR_BGR2RGB) 
    return cv2.resize(rgb_img, new_shape) 



    
    
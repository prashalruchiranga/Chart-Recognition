import os
import cv2
import matplotlib.pyplot as plt

def annotshow(image, annotation, radius=5, tl_color=(255, 0, 0), br_color=(0, 0, 255), thickness=-1):
    for dic in annotation:
        x0, y0, width, height = [int(x) for x in dic['bbox']]
        cv2.circle(image, (x0, y0), radius, tl_color, thickness)
        cv2.circle(image, (x0+width, y0+height), radius, br_color, thickness)
    fig, ax = plt.subplots(1)
    ax.imshow(image)
    ax.axis('off')
    plt.show()
    
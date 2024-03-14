import os
import cv2
import matplotlib.pyplot as plt

def annotshow(annotation, image, file_path, radius=5, tl_color=(255, 0, 0), br_color=(0, 0, 255), thickness=-1):
    full_path = os.path.join(file_path, image['file_name'])
    print(f"file name: {image['file_name']}")
    print(f"image id: {image['id']}")
    img = cv2.cvtColor(cv2.imread(full_path), cv2.COLOR_BGR2RGB)
    for dic in annotation:
        x0, y0, width, height = [int(x) for x in dic['bbox']]
        cv2.circle(img, (x0, y0), radius, tl_color, thickness)
        cv2.circle(img, (x0+width, y0+height), radius, br_color, thickness)
    fig, ax = plt.subplots(1)
    ax.imshow(img)
    ax.axis('off')
    plt.show()





    
    
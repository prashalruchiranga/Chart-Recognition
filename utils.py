import os
import cv2
import matplotlib.pyplot as plt

TRAIN_PATH_BAR = 'DeepRuleDataset/bardata(1031)/bar/images/train2019'
VAL_PATH_BAR = 'DeepRuleDataset/bardata(1031)/bar/images/val2019'

def annotshow(annotations, image_id, train=True, radius=3, thickness=-1):
    tl_color = (255, 0, 0)
    br_color = (0, 0, 255)
    image_id = image_id if train==True else image_id - annotations['images'][0]['id']
    if image_id >= 0:
        print(image_id)
        file_name = annotations['images'][image_id]['file_name']
        print(file_name)
        file_path = os.path.join(TRAIN_PATH_BAR, file_name) if train==True else os.path.join(VAL_PATH_BAR, file_name)
        img = cv2.cvtColor(cv2.imread(file_path), cv2.COLOR_BGR2RGB)
        for dic in annotations['annotations'][image_id]:
            x0, y0, width, height = [int(x) for x in dic['bbox']]
            cv2.circle(img, (x0, y0), radius, tl_color, thickness)
            cv2.circle(img, (x0+width, y0+height), radius, br_color, thickness)
        fig, ax = plt.subplots(1)
        ax.imshow(img)
        ax.axis('off')
        plt.show()
    else:
        print('Image id not found')
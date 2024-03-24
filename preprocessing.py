import json
from copy import deepcopy
import operator
import cv2

def clean(file):
    '''
    Removes images without annotations and groups annotations by image id.

    Parameters: 
    file (str): Annotations file path.

    Returns:
    dict: Cleaned annotations file. 
    '''
    with open(file) as f:
        data = json.load(f)

    grouped_annotations = {}
    for d in data['annotations']:
        grouped_annotations.setdefault(d['image_id'], []).append(d)
    annotations = list(grouped_annotations.values())

    img_ids_from_annotations = list(grouped_annotations.keys())
    img_ids_from_images = [dic['id'] for dic in data['images']]
    missing_annotations = [img_id for img_id in img_ids_from_images if img_id not in img_ids_from_annotations]
    images = [dic for dic in data['images'] if dic['id'] not in missing_annotations]

    return {'licenses': data['licenses'], 'images': images, 'annotations': annotations, 'categories': data['categories']}


def scale(annotation, input_shape, output_shape):
    '''
    Scale annotations according to output shape. 
    '''
    deep_copy = deepcopy(annotation)
    a, b = input_shape
    c, d = output_shape
    sx = d / b
    sy = c / a
    for dic in deep_copy:
        x0, y0, width, height = dic['bbox']
        dic['bbox'] = list(map(operator.mul, dic['bbox'], [sy, sx, sy, sx]))
    return deep_copy


def resize_image(file, new_shape):
    rgb_img = cv2.cvtColor(cv2.imread(file), cv2.COLOR_BGR2RGB) 
    return cv2.resize(rgb_img, new_shape) 


def transform(annotation, input_shape, output_shape):
    '''
    Scale annotations according to output shape and construct bottom right annotation.
    '''
    deep_copy = deepcopy(scale(annotation, input_shape, output_shape))
    for dic in deep_copy:
        x0, y0, width, height = dic['bbox']
        dic['bbox'] = [x0, y0, x0+width, y0+height]
    return deep_copy


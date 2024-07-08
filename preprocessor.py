import json
from copy import deepcopy
import operator
import cv2
import tensorflow as tf
import numpy as np
from itertools import chain
import os
import random

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
    foo = lambda x, y: np.floor(operator.mul(x, y)).astype(np.int32)
    for dic in deep_copy:
        x0, y0, width, height = dic['bbox']
        dic['bbox'] = list(map(foo, dic['bbox'], [sy, sx, sy, sx]))    
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


def generate_2d_guassian(height, width, y0, x0, sigma=1, scale=12):
    """
    "The same technique as Tompson et al. is used for supervision. A MeanSquared Error (MSE) loss is
    applied comparing the predicted heatmap to a ground-truth heatmap consisting of a 2D gaussian
    (with standard deviation of 1 px) centered on the keypoint location."

    https://github.com/princeton-vl/pose-hg-train/blob/master/src/util/img.lua#L204

    This code generates a 2D Gaussian heatmap centered at a specified point (y0, x0) in a grid of size height by width. 
    The Gaussian is defined by its standard deviation sigma, and its scale is adjusted by the scale parameter. 
    This heatmap is used for supervision in training neural networks, particularly for tasks like pose estimation.
    
    """
    heatmap = tf.zeros((height, width))

    # this gaussian patch is 7x7, let's get four corners of it first
    xmin = x0 - 3 * sigma
    ymin = y0 - 3 * sigma
    xmax = x0 + 3 * sigma
    ymax = y0 + 3 * sigma
    # if the patch is out of image boundary we simply return nothing according to the source code
    # [1]"In these cases the joint is either truncated or severely occluded, so for
    # supervision a ground truth heatmap of all zeros is provided."
    if xmin >= width or ymin >= height or xmax < 0 or ymax < 0:
        return heatmap

    size = 6 * sigma + 1
    x, y = tf.meshgrid(tf.range(0, 6*sigma+1, 1), tf.range(0, 6*sigma+1, 1), indexing='xy')

    # the center of the gaussian patch should be 1
    center_x = size // 2
    center_y = size // 2

    # generate this 7x7 gaussian patch
    gaussian_patch = tf.cast(tf.math.exp(-(tf.square(x - center_x) + tf.math.square(y - center_y)) / (tf.math.square(sigma) * 2)) * scale, dtype=tf.float32)

    # part of the patch could be out of the boundary, so we need to determine the valid range
    # if xmin = -2, it means the 2 left-most columns are invalid, which is max(0, -(-2)) = 2
    patch_xmin = tf.math.maximum(0, -xmin)
    patch_ymin = tf.math.maximum(0, -ymin)
    # if xmin = 59, xmax = 66, but our output is 64x64, then we should discard 2 right-most columns
    # which is min(64, 66) - 59 = 5, and column 6 and 7 are discarded
    patch_xmax = tf.math.minimum(xmax, width) - xmin
    patch_ymax = tf.math.minimum(ymax, height) - ymin

    # also, we need to determine where to put this patch in the whole heatmap
    heatmap_xmin = tf.math.maximum(0, xmin)
    heatmap_ymin = tf.math.maximum(0, ymin)
    heatmap_xmax = tf.math.minimum(xmax, width)
    heatmap_ymax = tf.math.minimum(ymax, height)

    # finally, insert this patch into the heatmap
    indices = tf.TensorArray(tf.int32, 1, dynamic_size=True)
    updates = tf.TensorArray(tf.float32, 1, dynamic_size=True)

    count = 0

    for j in tf.range(patch_ymin, patch_ymax):
        for i in tf.range(patch_xmin, patch_xmax):
            indices = indices.write(count, [heatmap_ymin+j, heatmap_xmin+i])
            updates = updates.write(count, gaussian_patch[j][i])
            count += 1
                
    heatmap = tf.tensor_scatter_nd_update(heatmap, indices.stack(), updates.stack())

    # unfortunately, the code below doesn't work because 
    # tensorflow.python.framework.ops.EagerTensor' object does not support item assignment
    # heatmap[heatmap_ymin:heatmap_ymax, heatmap_xmin:heatmap_xmax] = gaussian_patch[patch_ymin:patch_ymax,patch_xmin:patch_xmax]

    return heatmap


def make_heatmaps(keypoint_tl, keypoint_br):
    # Define SCALE as heatmap.shape[0] / image.shape[0] 
    # SCALE = 64 / 256
    #SCALE = tf.cast(0.25, dtype=tf.float32)
    tl = tf.cast(tf.math.round(keypoint_tl), dtype=tf.int32) 
    br = tf.cast(tf.math.round(keypoint_br), dtype=tf.int32) 
    count = len(tl) 
    heatmap_array_tl = tf.TensorArray(tf.float32, count)
    heatmap_array_br = tf.TensorArray(tf.float32, count)
    for i in range(0, count, 2):
        gaussian = generate_2d_guassian(256, 256, tl[i+1], tl[i])
        heatmap_array_tl = heatmap_array_tl.write(i, gaussian)
        gaussian = generate_2d_guassian(256, 256, br[i+1], br[i])
        heatmap_array_br = heatmap_array_br.write(i, gaussian)
    tl_heatmaps = tf.math.reduce_sum(heatmap_array_tl.stack(), axis=0)
    br_heatmaps = tf.math.reduce_sum(heatmap_array_br.stack(), axis=0)
    heatmap = tf.stack([tl_heatmaps, br_heatmaps], axis=2)
    return heatmap


def label_generator(annotations, batch_size):
    num_samples = len(annotations)
    indices = np.arange(num_samples)
    keypoints_tl = [list(chain(*[dic['bbox'][0:2] for dic in annotation])) for annotation in annotations]
    keypoints_br = [list(chain(*[dic['bbox'][2:4] for dic in annotation])) for annotation in annotations]
    for start_idx in range(0, num_samples, batch_size):
        end_idx = min(start_idx + batch_size, num_samples)
        batch_keypoints_tl = keypoints_tl[start_idx:end_idx]
        batch_keypoints_br = keypoints_br[start_idx:end_idx]
        batch_labels = tf.stack(list(map(make_heatmaps, batch_keypoints_tl, batch_keypoints_br)))
        yield tf.stack(batch_labels)


def label_generator2(annotations, batch_size):
    num_samples = len(annotations)
    indices = np.arange(num_samples)
    keypoints_tl = [annotation['bbox'][0:2] for annotation in annotations]
    keypoints_br = [annotation['bbox'][2:4] for annotation in annotations]
    for start_idx in range(0, num_samples, batch_size):
        end_idx = min(start_idx + batch_size, num_samples)
        batch_keypoints_tl = keypoints_tl[start_idx:end_idx]
        batch_keypoints_br = keypoints_br[start_idx:end_idx]
        batch_labels = tf.stack(list(map(make_heatmaps, batch_keypoints_tl, batch_keypoints_br)))
        yield tf.stack(batch_labels)


def divide_list_of_dicts(dict_list, chunk_size):
    # Use list comprehension with slicing to create chunks of the specified size
    return [dict_list[i:i + chunk_size] for i in range(0, len(dict_list), chunk_size)]


def shuffle_lists_together(list1, list2):
    if len(list1) != len(list2):
        raise ValueError("Both lists must have the same length.")
    combined = list(zip(list1, list2)) # Combine the lists into a list of tuples
    random.shuffle(combined) # Shuffle the combined list
    list1_shuffled, list2_shuffled = zip(*combined) # Unzip the combined list back into two lists
    return list(list1_shuffled), list(list2_shuffled) # Convert the tuples back to lists and return


def shuffle(file_list, heatmaps, chunk_size):
    grouped_file_list = divide_list_of_dicts(file_list, chunk_size)
    shuffled_grouped_file_list, shuffled_heatmaps = shuffle_lists_together(grouped_file_list, heatmaps)
    return (list(chain(*shuffled_grouped_file_list)), shuffled_heatmaps)


def data_generator(file_list, heatmaps, batch_size, output_shape, file_path):
    num_samples = len(file_list)
    indices = np.arange(num_samples)
    while True:
        file_list, heatmaps = shuffle(file_list, heatmaps, chunk_size=batch_size)
        iteration = 0
        for start_idx in range(0, num_samples, batch_size):
            end_idx = min(start_idx + batch_size, num_samples)
            batch_indices = indices[start_idx:end_idx]
            batch_images = []
            for idx in batch_indices:
                file_name = file_list[idx]['file_name']
                img = resize_image(os.path.join(file_path, file_name), output_shape)
                batch_images.append(img)  
            batch_labels = np.load(heatmaps[iteration])['batch']
            iteration += 1
            yield (tf.stack(batch_images), tf.convert_to_tensor(batch_labels)) 


def data_generator2(file_list, heatmaps, batch_size, output_shape, file_path):
    num_samples = len(file_list)
    indices = np.arange(num_samples)
    while True:
        #file_list, heatmaps = shuffle(file_list, heatmaps, chunk_size=batch_size)
        iteration = 0
        for start_idx in range(0, num_samples, batch_size):
            end_idx = min(start_idx + batch_size, num_samples)
            batch_indices = indices[start_idx:end_idx]
            batch_images = []
            for idx in batch_indices:
                file_name = file_list[idx]['file_name']
                img = resize_image(os.path.join(file_path, file_name), output_shape)
                batch_images.append(img)  
            batch_labels = np.sum(np.load(heatmaps[iteration])['batch'], axis=-1)
            batch_labels = np.expand_dims(batch_labels, axis=-1)
            iteration += 1
            yield (tf.stack(batch_images), tf.convert_to_tensor(batch_labels)) 
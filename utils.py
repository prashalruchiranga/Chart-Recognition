import os
import cv2
import matplotlib.pyplot as plt
from copy import deepcopy
import numpy as np
import re
import pytesseract
from scipy.ndimage import center_of_mass


def annotshow(image, annotation, radius=5, tl_color=(255, 0, 0), br_color=(0, 0, 255), thickness=-1):
    image_copy = deepcopy(image)
    for point in annotation:
        point = [int(x) for x in point]
        x0, y0, width, height = point
        cv2.circle(image_copy, (x0, y0), radius, tl_color, thickness)
        cv2.circle(image_copy, (x0+width, y0+height), radius, br_color, thickness)
    fig, ax = plt.subplots(1)
    ax.imshow(image_copy)
    ax.axis('off')
    plt.show()


def delete_contents(directory):
    contents = os.listdir(directory)
    if len(contents) == 0:
        print(f'The {directory} is already empty')
        return
    print(f'Deleting the contents in the {directory}')
    for file in contents:
        path = os.path.join(directory, file)
        os.remove(path)
    return


def extract_y_axis(file):
    # Load the image
    image = cv2.imread(file)
    # Check if the image is loaded correctly
    if image is None:
        raise ValueError('Image not loaded. Check the path.')
    # Get the dimensions of the image
    height, width = image.shape[:2]
    # Define a threshold to consider a line too close to the border
    border_threshold = 10  

    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian blur
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Apply Canny edge detector
    edges = cv2.Canny(blurred, threshold1=50, threshold2=150)

    # Detect lines using Hough Line Transform
    lines = cv2.HoughLinesP(edges, rho=1, theta=np.pi/180, threshold=100, minLineLength=100, maxLineGap=10)

    # Create a copy of the original image to draw lines on
    line_image = np.copy(image)

    # Initialize variables to find the leftmost vertical line
    min_x = float('inf')
    y_axis_line = None

    angles_in_degrees = []

    # Filter and identify the y-axis line
    for line in lines:
        for x1, y1, x2, y2 in line:
            # Calculate the angle of the line
            angle = np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi
            angles_in_degrees.append(angle)
            # Check if the line is nearly vertical
            if abs(angle) == 90 and min(x1, x2) > border_threshold and max(x1, x2) < (width - border_threshold):
                # Take the minimum x-coordinate of the line
                if min(x1, x2) < min_x:
                    min_x = min(x1, x2)
                    y_axis_line = ((x1, y1), (x2, y2))

    # Draw the y-axis line on the image if found
    if y_axis_line:
        cv2.line(line_image, y_axis_line[0], y_axis_line[1], (0, 0, 255), 2)
    
    # Print the y-axis line coordinates if found
    if y_axis_line:
        print('Y-Axis Line Coordinates:', y_axis_line)
    else:
        print('No Y-Axis Line detected')

    return y_axis_line, line_image


def extract_text(image, y_axis_line):
    starting_point, _ = y_axis_line
    x, _ = starting_point
    roi = image[: , 0:x]
    text = pytesseract.image_to_string(roi)
    return text.strip()


def find_min_max(txt):
    # splits = re.split(r'\n\n|\n| ', txt)
    txt = txt.replace(',', '')
    splits = txt.split('\n')
    #numbers_and_spaces = [re.sub(r'\D', '', item) for item in splits]
    numbers_only = [int(num) for num in splits if num.isdigit()]
    return (min(numbers_only), max(numbers_only))


def extract_centroids(predicted_heatmap, threshold):
    retval, dst = cv2.threshold(src=predicted_heatmap, thresh=threshold, maxval=1, type=cv2.THRESH_BINARY)
    centroids_for_both_channels = []
    for i in range(2):
        heatmap = dst[:,:,i].astype(np.uint8)
        # Find contours (islands)
        contours, _ = cv2.findContours(heatmap, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # Calculate the centroid for each contour
        centroids = []
        for contour in contours:
            # Calculate the centroid using moments
            M = cv2.moments(contour)
            if M['m00'] != 0:
                cx = int(M['m10'] / M['m00'])
                cy = int(M['m01'] / M['m00'])
                centroids.append((cx, cy))
        centroids_for_both_channels.append(centroids)
    return centroids_for_both_channels


def match_bar_coordinates(centroids):
    top_left, bottom_right = centroids
    matched_coordinates = list(zip(sorted(top_left), sorted(bottom_right)))
    return matched_coordinates


def calculate_scaling_factor(src_image_height, heatmap_height, y_axis_line, data_range):
    top_left, bottom_right = y_axis_line
    _, axis_y1 = top_left
    _, axis_y2 = bottom_right
    image_scale = heatmap_height / src_image_height
    return data_range / (abs(axis_y1-axis_y2) * image_scale)


def bar_values(matched_coordinates, scaling_factor):
    return [round(scaling_factor * abs(y1 - y2)) for ((x1, y1), (x2, y2)) in matched_coordinates]


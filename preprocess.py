import json

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

    img_ids_from_annotations = list(grouped_annotations.keys())
    img_ids_from_images = [dic['id'] for dic in data['images']]
    missing_annotations = [img_id for img_id in img_ids_from_images if img_id not in img_ids_from_annotations]
    images = [dic for dic in data['images'] if dic['id'] not in missing_annotations]

    return {'licenses': data['licenses'], 'images': images, 'annotations': grouped_annotations, 'categories': data['categories']}
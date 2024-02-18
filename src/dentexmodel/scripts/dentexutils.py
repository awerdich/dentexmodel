import os
from dentexmodel.dentexdata import DentexData

# Data directory (change as needed)
dentex_dir = os.path.join(os.environ['HOME'], 'data', 'dentex')
data_dir = os.path.join(dentex_dir, 'dentex_detection')

# Json file
annotation_file_name = 'train_quadrant_enumeration.json'
json_file = os.path.join(data_dir, 'quadrant_enumeration', annotation_file_name)

dtx = DentexData(data_dir=data_dir)
annotations = dtx.load_annotations(json_file=json_file)


def create_rcnn_anntations(annotations, file):
    """
    Create RCNN annotations for a given file.
    Parameters:
    - annotations (dict): The annotations dictionary containing information about images and annotations.
    - file (str): The path of the file for which RCNN annotations are to be created.
    Returns:
    - dict: The image annotation dictionary with RCNN annotations.
    """
    file_name = os.path.basename(file)
    im_annotation = {}
    # Verify that the image exists
    if is_image(file):

        # Find the image annotation
        im_annotation = [dct.copy() for dct in annotations.get('images') if dct.get('file_name') == file_name][0]

        # Replace the file_name field in the annotation dictionary with the full path
        im_annotation.update({'file_name': file})

        # We also need an 'image_id' field. We can replace the original id.
        image_id = im_annotation.get('id')
        im_annotation.update({'image_id': image_id})
        im_annotation.pop('id')

        # Find the list of annotations for one image
        annotation_list = [an_dict for an_dict in annotations.get('annotations') if an_dict.get('image_id') == image_id]

        # We pull out just the information that we need into a new list
        record_list = []
        for an_dict in annotation_list:
            im_dict = {'id': an_dict.get('id'),
                       'area': an_dict.get('area'),
                       'iscrowd': an_dict.get('iscrowd'),
                       'bbox': an_dict.get('bbox'),
                       'bbox_mode': BoxMode.XYWH_ABS,
                       'category_id': 0}
            record_list.append(im_dict)

        # Add this list of annotations to the image annotation dictionary
        im_annotation.update({'annotations': record_list})

    return im_annotation

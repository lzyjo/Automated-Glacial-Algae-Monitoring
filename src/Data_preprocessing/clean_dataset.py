import os
import pathlib
import xml.etree.ElementTree as ET
import sys
import matplotlib.pyplot as plt
from label_map import label_map_ODOC
import argparse

"""
This script processes XML annotation files to collate object types, remove unwanted labels, and convert labels to a standard format.
"""



parser = argparse.ArgumentParser(description='Label Handling for Datasets')
parser.add_argument('--annotation_folder', type=str, required=True, help='Path to the folder containing XML annotation files')
parser.add_argument('--remove_labels', type=str, nargs='+', default=['UNKNOWN', 'Question', 'None'], help='List of labels to remove from annotations (case-sensitive). You must enter the labels exactly as they appear in the XML files, e.g., UNKNOWN, Question, None')
parser.add_argument('--clean_labels', action='store_true', help='Flag to clean labels (remove unwanted labels and convert uncertain labels)')
parser.add_argument('--convert_labels', action='store_true', \
                    help='Flag to convert labels to a standard format for the model type. ' \
                    'If cell_detection, converts to "cell"; if species_classification, converts to "AA", "AN", "A_morph"; ' \
                    'if count_classification, converts to counts; ' \
                    'if detection_and_classification, keeps original labels. ')
parser.add_argument('--model_type', type=str, choices=['cell_detection', 'species_classification', 'cell_counter'], help='Only required if --convert_labels is set. Specifies the type of model for which label handling is to be performed.')
# parser.add_argument('--split', action='store_false', help='Flag to indicate whether to split the dataset into train, val, and test sets. Default is False')
args = parser.parse_args()

"""
This script processes XML annotation files to collate object types, remove unwanted labels, and convert labels to a standard format.

Usage:
python clean_dataset.py --annotation_folder /path/to/annotations \\
                                --remove_labels UNKNOWN Question None \\
                                    --clean_labels \\
                                        --convert_labels \\
                                            --model_type cell_detection \\
                                                # --split
"""



# -------------------------------------------------- OBJECT SUMMARY FUNCTIONS --------------------------------------------------

def collate_object_types(annotation_folder):
    """
    Collates and counts the different object types present in XML annotation files within a specified folder.
    This function iterates through all XML files in the given folder, parses each file to extract object labels,
    and counts the occurrences of each unique label. The results are printed to the console.
    Args:
        annotation_folder (str): The path to the folder containing XML annotation files.
    Returns:
        dict: A dictionary with object labels as keys and their respective counts as values.
        int: Total number of objects found.
    """
    object_counts = {}
    total_objects = 0

    for filename in os.listdir(annotation_folder):
        if filename.endswith('.xml'):
            file_path = os.path.join(annotation_folder, filename)
            tree = ET.parse(file_path)
            root = tree.getroot()

            for obj in root.findall('object'):
                label = obj.find('name').text
                object_counts[label] = object_counts.get(label, 0) + 1
                total_objects += 1

    for label, count in object_counts.items():
        print(f"Label: {label}, Count: {count}")
    print(f"Total objects: {total_objects}")

    return object_counts, total_objects


def object_count_for_cells(annotation_folder):
    annotation_folder = pathlib.Path(annotation_folder)
    if not annotation_folder.exists():
        raise FileNotFoundError(f"The folder '{annotation_folder}' does not exist. Please check the path.")
    count = 0
    for filename in os.listdir(annotation_folder):
        if filename.endswith('.xml'):
            file_path = os.path.join(annotation_folder, filename)
            tree = ET.parse(file_path)
            root = tree.getroot()
            for obj in root.findall('object'):
                name = obj.find('name').text
                if name != 'UNKNOWN':
                    count += 1
    
    print(f"Number of objects that are cells (excluding 'UNKNOWN'): {count}")
    return count



# --------------------------------------------------- LABEL REMOVAL FUNCTION --------------------------------------------------

def remove_objects_from_labels(annotation_folder, remove_labels=None):
    """
    Removes objects with specified labels from XML annotation files in the given folder.
    Args:
        annotation_folder (str): Path to the folder containing XML annotation files.
        remove_labels (list or set): Object labels to remove (e.g., ['UNKNOWN', 'question']).
    Returns:
        None
    """
    if remove_labels is None:
        sys.exit("Please specify labels to remove, e.g., ['UNKNOWN', 'question']")
    remove_labels = set(remove_labels)

    for filename in os.listdir(annotation_folder):
        if filename.endswith('.xml'):
            file_path = os.path.join(annotation_folder, filename)
            tree = ET.parse(file_path)
            root = tree.getroot()

            objects = root.findall('object')
            to_remove = []
            for obj in objects:
                label = obj.find('name').text
                if label in remove_labels:
                    to_remove.append(obj)

            for obj in to_remove:
                root.remove(obj)

            tree.write(file_path)

    print(f"Removed labels {remove_labels} from {annotation_folder}.")



# -------------------------------------------------- LABEL HANDLING FUNCTIONS --------------------------------------------------

def convert_uncertain_labels(annotation_folders):
    """
    Updates specific uncertain labels in XML annotation files within specified folders.
    - If label is 'possible_AN1_but_labelled_as_AA1', it is updated to 'AA1'.
    - If label is 'possible_particle_but_labelled_AA_1', the object is removed (we assume it is a particle).
    The modified XML file is then saved back to its original location.

    Args:
        annotation_folders (list or iterable): The paths to the folders containing XML annotation files.
    Returns:
        None
    """
    for folder in annotation_folders:
        annotation_folder = pathlib.Path(folder)
        for filename in os.listdir(annotation_folder):
            if filename.endswith('.xml'):
                file_path = os.path.join(annotation_folder, filename)
                tree = ET.parse(file_path)
                root = tree.getroot()

                to_remove = []
                changed = False
                for obj in root.findall('object'):
                    label_elem = obj.find('name')
                    if label_elem is not None:
                        label = label_elem.text
                        if label == 'possible_particle_but_labelled_AA_1': # remove this object
                            to_remove.append(obj)
                            changed = True
                        elif label == 'possible_AN1_but_labelled_as_AA1':
                            label_elem.text = 'AA_1'
                            changed = True
                        if label == 'possible AN_2_but_labelled_AA_2':
                            label_elem.text = 'AA_2'
                            changed = True

                for obj in to_remove:
                    root.remove(obj)

                if changed:
                    tree.write(file_path)

        print(f"Labels in {annotation_folder} have been updated for uncertain cases.")


def convert_labels_to_cell(annotation_folder):
    """
    Updates the labels in XML annotation files within a specified folder.
    This function iterates through all XML files in the given folder, and for each
    file, it parses the XML structure to find all object elements. If the label
    ('name' tag) of an object is not 'UNKNOWN', it updates the label to 'cell'.
    The modified XML file is then saved back to its original location.
    Args:
        annotation_folder (str): The path to the folder containing XML annotation files.
    Returns:
        None
    """
    for filename in os.listdir(annotation_folder):
        if filename.endswith('.xml'):
            file_path = os.path.join(annotation_folder, filename)
            tree = ET.parse(file_path)
            root = tree.getroot()

            objects_to_change = []
            for obj in root.findall('object'):
                label = obj.find('name').text
                if label != 'UNKNOWN':
                    objects_to_change.append(obj)

            for obj in objects_to_change:
                obj.find('name').text = 'cell'

            tree.write(file_path)
    
    print(f"Labels in {annotation_folder} have been updated to 'cell' for all objects except 'UNKNOWN'.")


def convert_labels_to_species(annotation_folder):  
    """
    Updates the labels in XML annotation files within a specified folder to 'AA' or 'AN'.
    This function iterates through all XML files in the given folder, and for each
    file, it parses the XML structure to find all object elements. If the label
    ('name' tag) of an object is not 'UNKNOWN', it updates the label to 'AA' or 'AN'
    based on the original label. The modified XML file is then saved back to its original location.
    Args:
        annotation_folder (str): The path to the folder containing XML annotation files.
    Returns:
        None
    """
    for filename in os.listdir(annotation_folder):
        if filename.endswith('.xml'):
            file_path = os.path.join(annotation_folder, filename)
            tree = ET.parse(file_path)
            root = tree.getroot()

            for obj in root.findall('object'):
                label_elem = obj.find('name')
                if label_elem is not None:
                    label = label_elem.text
                    if label == 'UNKNOWN':
                        continue
                    elif 'AA' in label:
                        label_elem.text = 'AA'
                    elif 'AN' in label:
                        label_elem.text = 'AN'
                    elif 'A_morph' in label:
                        label_elem.text = 'A_morph'

            tree.write(file_path)

    print(f"Labels in {annotation_folder} have been updated to 'AA' or 'AN' based on the original label.")


def convert_labels_to_counts(annotation_folder):
    """
    Updates the labels in XML annotation files within a specified folder to '1', '2', or '3'.
    This function iterates through all XML files in the given folder, and for each
    file, it parses the XML structure to find all object elements. If the label
    ('name' tag) of an object is not 'UNKNOWN', it updates the label to '1', '2', or '3'
    based on the original label. The modified XML file is then saved back to its original location.
    Args:
        annotation_folder (str): The path to the folder containing XML annotation files.
    Returns:
        None
    """ 
    for filename in os.listdir(annotation_folder):
        if filename.endswith('.xml'):
            file_path = os.path.join(annotation_folder, filename)
            tree = ET.parse(file_path)
            root = tree.getroot()

            for obj in root.findall('object'):
                label_elem = obj.find('name')
                if label_elem is not None:
                    label = label_elem.text
                    if label == 'UNKNOWN':
                        continue
                    elif 'AA' in label:
                        # Extract the number after the last underscore
                        parts = label.split('_')
                        if parts and parts[-1].isdigit():
                            label_elem.text = parts[-1]
                        else:
                            label_elem.text = 'AA'
                    elif 'AN' in label:
                        parts = label.split('_')
                        if parts and parts[-1].isdigit():
                            label_elem.text = parts[-1]
                        else:
                            label_elem.text = 'AN'
                    elif 'A_morph' in label:
                        parts = label.split('_')
                        if parts and parts[-1].isdigit():
                            label_elem.text = parts[-1]
                        else:
                            label_elem.text = 'A_morph'

            tree.write(file_path)

    print(f"Labels in {annotation_folder} have been updated to counts based on the original label.")


# --------------------------------------------------- NESTED FUNCTIONS --------------------------------------------------


def remove_unwanted_labels(annotation_folders,
                            remove_labels=['UNKNOWN', 'Question', 'None']):
    """
    Cleans annotation labels in the specified folders by removing unwanted object labels.

    For each folder in `annotation_folders`, this function:
        1. Prints the folder being processed.
        2. Displays a summary of object types before conversion.
        3. Removes objects with labels specified in `remove_labels` (case-sensitive).
        4. Displays a summary of object types after removal.
        5. Prints a separator line.

    Args:
        annotation_folders (list or iterable): List of paths to folders containing annotation files.

    Notes:
        - The function relies on `od_utils.collate_object_types` to summarize object types.
        - The function uses `od_utils.remove_objects_from_labels` to remove unwanted labels.
        - Only labels matching exactly 'UNKNOWN' or 'Question' (case-sensitive) are removed.
    """
    for folder in annotation_folders:
        annotation_folder = pathlib.Path(folder)
        print(f'Processing folder: {annotation_folder}')
        print(f'Object summary before conversion:')
        collate_object_types(annotation_folder)  # Collate and count object types in the annotations
        remove_objects_from_labels(annotation_folder, remove_labels=remove_labels)  # Case-sensitive labels
                                                                # Ensure case-sensitivity is handled, use exact label names
        print(f'Object summary after removing unwanted labels:')
        collate_object_types(annotation_folder)
        print(*'-'*50)

def convert_all_labels_to_cell(annotation_folders):
    """
    Converts all object labels in the specified annotation folders to 'cell', except for 'UNKNOWN' labels.

    For each folder in `annotation_folders`, this function:
        1. Prints the folder being processed.
        2. Displays a summary of object types before conversion.
        3. Converts all object labels (except 'UNKNOWN') to 'cell'.
        4. Displays a summary of object types after conversion.
        5. Prints a separator line.
    Args:  
        annotation_folders (list or iterable): List of paths to folders containing annotation files.
    Notes:
        - The function relies on `collate_object_types` to summarize object types.
        - The function uses `convert_labels_to_cell` to perform the label conversion.
    """
    for folder in annotation_folders:
        annotation_folder = pathlib.Path(folder)
        print(f'Processing folder: {annotation_folder}')
        print(f'Object summary before conversion:')
        collate_object_types(annotation_folder)  # Collate and count object types in the annotations
        convert_labels_to_cell(annotation_folder)  # Convert all labels to 'cell'
        print(f'Object summary after label conversion:')
        collate_object_types(annotation_folder)
        print(*'-'*50)

def convert_all_labels_to_species(annotation_folders):
    """
    Converts all object labels in the specified annotation folders to 'AA', 'AN' or 'A_morph' except for 'UNKNOWN' labels.

    For each folder in `annotation_folders`, this function:
        1. Prints the folder being processed.
        2. Displays a summary of object types before conversion.
        3. Converts all object labels (except 'UNKNOWN') to 'AA' or 'AN'.
        4. Displays a summary of object types after conversion.
        5. Prints a separator line.
    Args:
        annotation_folders (list or iterable): List of paths to folders containing annotation files.
    Notes: 
        - The function relies on `collate_object_types` to summarize object types.
        - The function uses `convert_labels_to_species` to perform the label conversion.            
    """
    for folder in annotation_folders:
        annotation_folder = pathlib.Path(folder)
        print(f'Processing folder: {annotation_folder}')
        print(f'Object summary before conversion:')
        collate_object_types(annotation_folder)  # Collate and count object types in the annotations
        convert_labels_to_species(annotation_folder)  # Convert all labels to 'AA' or 'AN'
        print(f'Object summary after label conversion:')
        collate_object_types(annotation_folder)
        print(*'-'*50)

def convert_all_labels_to_counts(annotation_folders):
    """
    Converts all object labels in the specified annotation folders to counts ('1', '2', '3' etc), except for 'UNKNOWN' labels.

    For each folder in `annotation_folders`, this function:
        1. Prints the folder being processed.
        2. Displays a summary of object types before conversion.
        3. Converts all object labels (except 'UNKNOWN') to counts based on the original label.
        4. Displays a summary of object types after conversion.
        5. Prints a separator line.         
    Args:
        annotation_folders (list or iterable): List of paths to folders containing annotation files.
    Notes:  
        - The function relies on `collate_object_types` to summarize object types.
        - The function uses `convert_labels_to_counts` to perform the label conversion.            
    """
    for folder in annotation_folders:
        annotation_folder = pathlib.Path(folder)
        print(f'Processing folder: {annotation_folder}')
        print(f'Object summary before conversion:')
        collate_object_types(annotation_folder)  # Collate and count object types in the annotations
        convert_labels_to_counts(annotation_folder)  # Convert all labels to counts
        print(f'Object summary after label conversion:')
        collate_object_types(annotation_folder)
        print(*'-'*50)


###########################################################################################################

if __name__ == "__main__":

    annotation_folder = args.annotation_folder


    proceed = input(f"Show object summary for annotations in '{annotation_folder}'? (y/n): ")
    if proceed.lower() != 'y':
        print("Operation aborted by user.")
        sys.exit()


    # object summmary -----------------------------------------------------------
    summary = collate_object_types(annotation_folder)
    for label, count in summary.items():
        print(f"Label: {label}, Count: {count}")


    # clean and/or convert labels ------------------------------------------------
    if args.clean_labels:
        remove_unwanted_labels(annotation_folder)
        print("Removed unwanted labels.")
        convert_uncertain_labels(annotation_folder)
        print("Converted uncertain labels.")

    cleaned_summary = collate_object_types(annotation_folder)
    print(f"Object summary after label handling: {cleaned_summary}")
    for label, count in cleaned_summary.items():
        print(f"Label: {label}, Count: {count}")


    if args.convert_labels:
        supported_types = [
            'cell_detection', # convert all labels to 'cell' except for 'UNKNOWN'
            'species_classification', # convert all labels to 'AA' or 'AN' except for 'UNKNOWN'
            'cell_counter', # convert all labels to counts ('1', '2', '3' etc) except for 'UNKNOWN'
            # 'detection_and_classification' # keep original labels (no conversion)
            ]
        
        if args.model_type is None:
            raise ValueError(
                f"`model_type` must be specified when `convert_labels` is True. Supported types are {supported_types}."
            )
        if args.model_type not in supported_types:
            raise ValueError(
                f"Unsupported model_type '{args.model_type}'. Supported types are {supported_types}."
            )

        if args.model_type == 'cell_detection':
            convert_all_labels_to_cell(annotation_folder)
            print(f"Labels in {annotation_folder} have been converted to 'cell' for all objects except 'UNKNOWN'.")
        elif args.model_type == 'species_classification':
            convert_all_labels_to_species(annotation_folder)
            print(f"Labels in {annotation_folder} have been converted to 'AA' or 'AN' based on the original label.")
        elif args.model_type == 'cell_counter':
            convert_all_labels_to_counts(annotation_folder)
            print(f"Labels in {annotation_folder} have been converted to counts based on the original label.")
        # For detection_and_classification, keep original labels (no conversion)
        else:
            print("No label conversion performed as model type not specified.")


        for folder in annotation_folder:
            annotation_folder = pathlib.Path(folder)
            print(f'Labels have been converted.')
            print(f'Final object summary for folder: {annotation_folder}')
            collate_object_types(annotation_folder)


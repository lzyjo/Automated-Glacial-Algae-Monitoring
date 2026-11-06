"""
Standalone script to create data.yaml for YOLO training.
Run this locally, then transfer the generated data.yaml to HPC.

Usage:
    python voc_to_yolo.py --data_folder "path/to/dataset" --label_map_type detector
"""

import argparse
import pathlib
import yaml
import sys
import os
from label_map import label_map_Detector, label_map_Classifier, label_map_Counter
from label_map import rev_label_map_OD, rev_label_map_Classifier, rev_label_map_Counter


argument_parser = argparse.ArgumentParser(description="Create data.yaml for YOLO training from VOC-style dataset structure.")
argument_parser.add_argument('--data_folder', type=str, required=True, help='Path to dataset folder (must contain images/train, images/val, labels/train, labels/val)')
# argument_parser.add_argument('--label_map_type', type=str, choices=['detector', 'classifier', 'counter'], default='detector', help='Type of label map to use (default: detector)')
argument_parser.add_argument('--model_type', type=str, choices=['cell_detection', 'species_classification', 'cell_counter'], help='Define what type of model you are training.')
argument_parser.add_argument('--split_dir', type=str, optional=True, default=None, help='Optional: If dataset is already split into train/val/test, specify the path to that directory. \\' \
    'Please ensure that the directory structure is correct (root/images/train, root/images/val, root/images/test, root/labels/train, root/labels/val, root/labels/test).')
args = argument_parser.parse_args()

"""
Usage:
  python voc_to_yolo.py --data_folder <dataset_path> \\
                            --model_type cell_detection \\
                            --split_dir <split_dataset_path>
"""

                            
if args.model_type == 'cell_detection':
    label_map = rev_label_map_OD
if args.model_type == 'species_classification':
    label_map = rev_label_map_Classifier
if args.model_type == 'cell_counter':
    label_map = rev_label_map_Counter

image_dir = os.path.join(args.data_folder, 'images')
annotation_dir = os.path.join(args.data_folder, 'labels')

if args.split_dir:
    # check if the provided split_dir exists and is a directory
    if not os.path.isdir(args.split_dir):
        print(f"Error: The provided split_dir '{args.split_dir}' does not exist or is not a directory.")
        sys.exit(1)
    # check if split_dir has the required subdirectories and is formatted correctly
        if not (os.path.isdir(os.path.join(args.split_dir, 'images', 'test')) and
            os.path.isdir(os.path.join(args.split_dir, 'labels', 'test'))):
            print(f"The provided split_dir '{args.split_dir}' does not have the required 'test' subdirectories.")
            if input("Do you want to continue without 'test' data? (y/n): ").lower() != 'y':
                print("Aborting.")
                sys.exit(0)

        if not (os.path.isdir(os.path.join(args.split_dir, 'images', 'train')) and
                os.path.isdir(os.path.join(args.split_dir, 'images', 'val')) and
                # os.path.isdir(os.path.join(args.split_dir, 'images', 'test')) and
                os.path.isdir(os.path.join(args.split_dir, 'labels', 'train')) and
                os.path.isdir(os.path.join(args.split_dir, 'labels', 'val')) #and
                # os.path.isdir(os.path.join(args.split_dir, 'labels', 'test'))
                ):
            print(f"Error: The provided split_dir '{args.split_dir}' does not have the required subdirectory structure.")
            print("Expected structure: root/images/train, root/images/val, root/images/test, root/labels/train, root/labels/val, root/labels/test")
            sys.exit(1)

        image_train_dir = os.path.join(args.split_dir, 'images', 'train')
        image_val_dir = os.path.join(args.split_dir, 'images', 'val')
        image_test_dir = os.path.join(args.split_dir, 'images', 'test')
        label_train_dir = os.path.join(args.split_dir, 'labels', 'train')
        image_val_dir = os.path.join(args.split_dir, 'labels', 'val')
        image_test_dir = os.path.join(args.split_dir, 'labels', 'test')
        all_dirs = [image_train_dir, image_val_dir, image_test_dir, label_train_dir, image_val_dir, image_test_dir]
        print(f"Using provided split_dir: {args.split_dir}")

        missing_dirs = [d for d in all_dirs if not os.path.exists(d)]
        if missing_dirs:
            print(f"✗ Missing required directories in split_dir:")
            for d in missing_dirs:
                print(f"  {d}")
            sys.exit(1)

    
        print(f"\n{'='*30}")
        print("Dataset structure:")
        print(f"{'='*30}\n")

        print(f"Train images: {image_train_dir}")
        print(f"Val images: {image_val_dir}")
        print(f"Test images: {image_test_dir}")
        print(f"Train labels: {label_train_dir}")
        print(f"Val labels: {image_val_dir}")
        print(f"Test labels: {image_test_dir}")





############################################ FUNCTIONS ############################################

def create_yolo_yaml(data_folder, 
                     label_map_type=label_map, 
                     split_dir=None,
                     output_name='data.yaml'):
    """
    Create a data.yaml file for YOLO training.
    
    Args:
        data_folder: Path to dataset folder (must contain images/train, images/val, labels/train, labels/val)
        label_map_type: Type of label map to use (detector, classifier, counter)
        split_dir: Path to the directory containing the train, val, and test splits
        output_name: Name of output YAML file
    
    Returns:
        yaml_path: Path to created data.yaml
    """
    
    data_folder = pathlib.Path(data_folder).resolve()

    image_dir = os.path.join(args.data_folder, 'images')
    annotation_dir = os.path.join(args.data_folder, 'labels')
    
    print(f"\n{'='*30}")
    print("Creating data.yaml for YOLO training")
    print(f"Folder YAML will be placed: {data_folder}")

    print(f"{'='*30}\n")
    print(f"Using dataset folder: {data_folder} and converting annotation dir: {annotation_dir}")

    if split_dir:
        image_train_dir = pathlib.Path(split_dir) / 'images' / 'train'
        image_val_dir = pathlib.Path(split_dir) / 'images' / 'val'
        image_test_dir = pathlib.Path(split_dir) / 'images' / 'test'
        label_train_dir = pathlib.Path(split_dir) / 'labels' / 'train'
        label_val_dir = pathlib.Path(split_dir) / 'labels' / 'val'
        label_test_dir = pathlib.Path(split_dir) / 'labels' / 'test'

        print(f"Using provided split_dir: {split_dir}")

        print(f"Train images: {image_train_dir}")
        print(f"Val images: {image_val_dir}")
        print(f"Test images: {image_test_dir}")
        print(f"Train labels: {label_train_dir}")
        print(f"Val labels: {label_val_dir}")
        print(f"Test labels: {label_test_dir}")
    
    # # Define label maps (same as in your label_map.py)
    # if label_map_type == 'detector':
    #     label_map = rev_label_map_OD
    # if label_map_type == 'classifier':
    #     label_map = rev_label_map_Classifier
    # if label_map_type == 'counter':
    #     label_map = rev_label_map_Counter

    print(f"Using label map type: {label_map} with {len(label_map)} classes")
    print(label_map)
    if input("Proceed? (y/n): ").lower() != 'y':
        print("Aborting.")
        sys.exit(0)


    # Build class names as a dictionary with integer keys indexed by class ID
    if isinstance(label_map, dict):
        names = {int(k): str(v) for k, v in label_map.items()}
    elif isinstance(label_map, (list, tuple)):
        names = {i: str(x) for i, x in enumerate(label_map) if x is not None}
    else:
        raise TypeError(f"Unsupported label_map type: {type(label_map)}")
    """# names = {i: str(x) for i, x in enumerate(label_map) if x is not None} 
    names = {i: x for i, x in enumerate(label_map) if x is not None} """


    # Count images
    if not split_dir:
        image_count = len(list(pathlib.Path(image_dir).glob('*.tif')))
        print(f"\nTotal images in dataset: {image_count}")


        data_config = { #create data.yaml config
            'path': str(data_folder),
            'train': str(pathlib.Path(image_dir) / 'train'), #train always required
            'nc': len(label_map),
            'names': names
            }

    if split_dir:
        train_count = len(list(image_train_dir.glob('*.tif')))
        val_count = len(list(image_val_dir.glob('*.tif')))
        test_count = len(list(image_test_dir.glob('*.tif')))    
    
        print(f"\nImage counts:")
        print(f"  Train: {train_count} images")
        print(f"  Val: {val_count} images")
        print(f"  Test: {test_count} images")

        data_config = {  # Create data.yaml config
            'path': str(data_folder),
            'train': str(image_train_dir), #train always required
            'val': str(image_val_dir), #optional
            'test': str(image_test_dir), #optional
            'nc': len(label_map),
            'names': names
        }
    
    # Save to file
    yaml_path = data_folder / output_name
    
    print(f"\n{'='*30}")
    print(f"Creating {output_name}...")
    print(f"{'='*30}\n")

    # Save the data.yaml file
    try:
        with open(yaml_path, 'w') as f:
            yaml.dump(data_config, f, default_flow_style=False, sort_keys=False)
        
        print(f"✓ Successfully created: {yaml_path}")
        print(f"\nFile contents:")
        print(f"{'-'*30}")
        with open(yaml_path, 'r') as f:
            print(f.read())
        print(f"{'-'*30}\n")

        print("YOLO data.yaml created successfully.")
        
        return yaml_path
        
    except Exception as e:
        print(f"✗ Error creating YAML: {e}")
        sys.exit(1)

############################################# MAIN ############################################

if __name__ == "__main__":
    yaml_path = create_yolo_yaml(data_folder=args.data_folder,
                label_map_type=label_map,
                split_dir=args.split_dir,
                output_name='data.yaml'
            )

    print(f"✓ data.yaml created at: {yaml_path}")

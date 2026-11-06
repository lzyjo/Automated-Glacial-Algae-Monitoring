import os
import pathlib
import argparse

# Example usage:
"""
This script creates a dataset folder structure for training data, including subfolders for annotations, images, and data splits. The folder structure is determined based on the specified date and whether the dataset is augmented.

Usage:
python create_formatted_folder.py \\
    --folder_name "MyDataset" \\
    --Split \\
    --Augmented \\
    --model_types detector cell_counter
"""

# Parse command-line arguments (add dynamic model/optimizer options)
parser = argparse.ArgumentParser(description='Data Augmentation for Cell Detection and Cell Counting')
parser.add_argument('--folder_name', type=str, required=True,
                    help='Name of the dataset folder to create')
parser.add_argument('--Split', action='store_false',
                    help='Whether to create train, test, and val subfolders for images and annotations')
parser.add_argument('--Augmented', action='store_false',
                    help='Whether the dataset is augmented. If True, "_Augmented" is appended to the folder name')
parser.add_argument('--model_types', type=str, nargs='+', required=True,
                    help='Type of model: detector, cell_counter')
args = parser.parse_args()

print(f"Folder Name: {args.folder_name}")
print(f"Split: {args.Split}")
print(f"Augmented: {args.Augmented}")
print(f"Model Types: {args.model_types}")
if input("Are you sure you want to proceed with the above settings? (y/n): ").strip().lower() != 'y':
    print("Dataset folder creation aborted by user.")
    SystemExit()


################################################ Functions ###################################################

def create_dataset_folder(folder_name,
                          Split=False,
                          Augmented=False,
                          model_type=None):
    """
    Creates a dataset folder structure for training data, including subfolders for annotations, images, 
    and data splits. The folder structure is determined based on the specified date and whether the dataset is augmented.
    Parameters:
        folder_date (str or None): The date string (in 'YYYYMMDD' format) for the folder. 
                                   If None, the current date is used.
        Augmented (bool): Whether the dataset is augmented. If True, '_Augmented' is appended to the folder name.
        Split (bool): Whether to create train, test, and val subfolders for images and annotations.
    Returns:
        tuple: A tuple containing the paths to the created folders:
               (annotations_folder, images_folder, split_folder, date_folder, train_image_folder, train_annotation_folder, 
               test_image_folder, test_annotation_folder, val_image_folder, val_annotation_folder).
    """

    # Append '_Augmented' to the folder name if the dataset is augmented
    folder_suffix = '_Augmented' if Augmented else ''
    folder_name = f'{folder_name}{folder_suffix}'
    if input(f"Creating dataset folder: {folder_name}. Press Enter to continue or Ctrl+C to abort.") != "":
        print("Dataset folder creation aborted by user.")
        SystemExit()


    # Append the model type to the folder name if provided
    if model_type:
        folder_name = f'{folder_name}_{model_type}'

    # Define the subfolders
    annotations_folder = os.path.join(folder_name, f'Annotations')
    images_folder = os.path.join(folder_name, f'Images')
    pathlib.Path(annotations_folder).mkdir(parents=True, exist_ok=True)
    pathlib.Path(images_folder).mkdir(parents=True, exist_ok=True)

    # Define train, test, and val subfolders if Split is True
    if Split:
        # Create the split folder
        split_folder = os.path.join(f'{folder_name}/Split')

        if os.path.exists(split_folder):
            print(f'Dataset folder already exists: {split_folder}')
            return

        split_folder = os.path.join(folder_name, f'Split')
        pathlib.Path(split_folder).mkdir(parents=True, exist_ok=True)

        train_image_folder = os.path.join(split_folder, 'train', 'images') 
        train_annotation_folder = os.path.join(split_folder, 'train', 'annotations') 
        test_image_folder = os.path.join(split_folder, 'test', 'images')
        test_annotation_folder = os.path.join(split_folder, 'test', 'annotations') 
        val_image_folder = os.path.join(split_folder, 'val', 'images')
        val_annotation_folder = os.path.join(split_folder, 'val', 'annotations')
        os.makedirs(train_image_folder, exist_ok=True)
        os.makedirs(train_annotation_folder, exist_ok=True)
        os.makedirs(test_image_folder, exist_ok=True)
        os.makedirs(test_annotation_folder, exist_ok=True)
        os.makedirs(val_image_folder, exist_ok=True)
        os.makedirs(val_annotation_folder, exist_ok=True)

        print(f'Dataset folder created: {folder_name}')
        print(f'Annotations Folder: {annotations_folder}')
        print(f'Images Folder: {images_folder}')
        print(f'Train Images Folder: {train_image_folder}')
        print(f'Train Annotations Folder: {train_annotation_folder}')
        print(f'Test Images Folder: {test_image_folder}')
        print(f'Test Annotations Folder: {test_annotation_folder}')
        print(f'Val Images Folder: {val_image_folder}')
        print(f'Val Annotations Folder: {val_annotation_folder}')

        return (folder_name, 
                annotations_folder,images_folder, 
                split_folder,
                train_image_folder, train_annotation_folder,
                test_image_folder, test_annotation_folder,
                val_image_folder, val_annotation_folder
        )

    else:
        print(f'Dataset folder created: {folder_name}')
        print(f'Annotations Folder: {annotations_folder}')
        print(f'Images Folder: {images_folder}')
        return (folder_name, annotations_folder, images_folder)


#########################################################################################################
if __name__ == "__main__":
    for model_type in args.model_types:
        print(f"Creating dataset folder for model type: {model_type}")
        create_dataset_folder(
            args.folder_name,
            Split=True,
            Augmented=True,
            model_type=model_type
        )


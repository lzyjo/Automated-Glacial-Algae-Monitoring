import argparse
from datetime import datetime
import os
import pathlib
import utils
import clean_dataset as od_utils
import sklearn.model_selection as sklearn_ms
import shutil

"""
Usage:
python dataset_formatting.py --split \\ #if you want to split the dataset into train/val/test, else omit this flag to skip splitting
                                --datasets_to_use "0_Completed_annotations/Mort_24,0_Completed_annotations/Greenland_24,0_Completed_annotations/Patagonia 24_25/1,0_Completed_annotations/Patagonia 24_25/2,0_Completed_annotations/Patagonia 24_25/3,0_Completed_annotations/Patagonia 24_25/GG_Glut"  \\
                                --output_folder "1_GA_Dataset/Mort24_Greenland24_Patagonia24_20251107" \\
                                --augmentation_root "2_DataAugmentation" \\
                                --model_types "detector, cell_counter"
"""

def _csv_to_list(value):
    return [item.strip() for item in value.split(',') if item.strip()]

parser = argparse.ArgumentParser(description='Dataset Pre-processing Script')
parser.add_argument('--split', action='store_true', help='Whether to split the dataset into train/val/test')
parser.add_argument('--datasets_to_use', type=str, default='0_Completed_annotations/Mort_24,0_Completed_annotations/Greenland_24,0_Completed_annotations/Patagonia 24_25/1,0_Completed_annotations/Patagonia 24_25/2,0_Completed_annotations/Patagonia 24_25/3,0_Completed_annotations/Patagonia 24_25/GG_Glut', help='Comma-separated list of dataset folders to use for initialization.')
parser.add_argument('--output_folder', type=str, default='1_GA_Dataset/Mort24_Greenland24_Patagonia24_20251107', help='Base output folder for the initialized dataset.')
parser.add_argument('--augmentation_root', type=str, default='2_DataAugmentation', help='Root folder for generated augmentations.')
parser.add_argument('--model_types', type=str, default='detector, cell_counter', help='Comma-separated list of model types to process.')
args = parser.parse_args()


""" 
For example, if you want to use the following datasets for initialization, 
you can specify them as a comma-separated list in the --datasets_to_use argument:
datasets_to_use = [r'0_Completed_annotations\Mort_24', 
                   r'0_Completed_annotations\Greenland_24',
                   r'0_Completed_annotations\Patagonia 24_25\1',
                   r'0_Completed_annotations\Patagonia 24_25\2',
                    r'0_Completed_annotations\Patagonia 24_25\3',
                    r'0_Completed_annotations\Patagonia 24_25\GG_Glut']
"""

datasets_to_use = _csv_to_list(args.datasets_to_use) 
print("Datasets to use for initialization:")
for dataset in datasets_to_use:
    print(f" - {dataset}")
if input("Are you sure you want to proceed with the above datasets? (y/n): ").strip().lower() != 'y':
    print("Dataset initialization aborted by user.")
    SystemExit()

output_folder = pathlib.Path(args.output_folder)
print(f"Output folder for initialized dataset: {output_folder}")
if input("Are you sure you want to proceed with the above output folder? (y/n): ").strip().lower() != 'y':
    print("Dataset initialization aborted by user.")
    SystemExit()

print(f"Augmentation root: {args.augmentation_root}")
if input("Are you sure you want to proceed with the above augmentation root? (y/n): ").strip().lower() != 'y':
    print("Dataset initialization aborted by user.")
    SystemExit()

model_types = _csv_to_list(args.model_types)
if model_types:
    print("Model types to process:")
    for model_type in model_types:
        print(f" - {model_type}")
    if input("Are you sure you want to proceed with the above model types? (y/n): ").strip().lower() != 'y':
        print("Dataset initialization aborted by user.")
        SystemExit()


# ------------------------------------------------------- DATASET INITIALIZATION ---------------------------------------------------------
# This script initializes dataset folders by organizing raw images and their corresponding annotations.


def initialize_dataset_folder(datasets_to_use, split=None):
    """
    Creates a new dataset folder for a given list of dataset paths, organizing raw images into a structured directory.
    For each dataset path provided, this function:
        - Extracts the stem (base name) of each dataset path, removing underscores.
        - Combines all dataset names and appends the current date (YYYYMMDD) to form a unique folder name.
        - Creates the folder inside '1_GA_Dataset' using `create_dataset_folder` with augmentation disabled and splitting enabled.
        - Moves raw images from each dataset path into the newly created dataset folder using `move_raw_images_to_dataset`.
        datasets_to_use (list or iterable of str): Paths to the datasets whose images should be organized.
        - Creates a new folder in '1_GA_Dataset' for the combined datasets.
        - Moves raw images from each dataset path into the new folder.
        initialize_dataset_folder([
            'path/to/Greenland24',
            'path/to/Mort24'
        ])
    Note:
    Requires the helper functions `create_dataset_folder` and `move_raw_images_to_dataset` to be defined elsewhere.
    """

    # Combine all dataset names (stems, no underscores) and append the current date to create the base folder name
    current_date = datetime.datetime.now().strftime('%Y%m%d')
    dataset_names = [pathlib.Path(ds).stem.replace('_', '') for ds in datasets_to_use]
    combined_name = '_'.join(dataset_names)
    proceed = input(f"Proceed with creating dataset folder: {combined_name}_{current_date}? (y/n): ")
    if proceed.lower() != 'y':
        custom_name = input("Please enter a custom name for the dataset folder (or press Enter to use the default or 'N' to cancel): ").strip()
        if custom_name:
            combined_name = custom_name
        if custom_name.lower() == "":
            print("Using default combined name.")
        if custom_name.lower() == "n":
            print("Dataset initialization aborted by user.")
            SystemExit()
        return
    base_folder = os.path.join('Dataset', f'{combined_name}_{current_date}')

    if split:
        (folder_name,
        annotations_folder,
        images_folder,
        split_folder,
        train_image_folder,
        train_annotation_folder,
        test_image_folder,
        test_annotation_folder,
        val_image_folder,
        val_annotation_folder
        ) = create_dataset_folder(folder_name=base_folder,
                                    Augmented=False,
                                    Split=True)
        
        traintestval_folders = [train_image_folder,
                                train_annotation_folder,
                                test_image_folder,
                                test_annotation_folder,
                                val_image_folder,
                                val_annotation_folder]

    if not split:
        (folder_name,
        annotations_folder,
        images_folder) = create_dataset_folder(folder_name=base_folder,
                                                Augmented=False,
                                                Split=False)

    # For each dataset, extract all raw image and annotation pairs
    image_annotation_pairs, unmatched_folders = find_and_extract_image_annotation_pairs(datasets_to_use, folder_name)

    print(len(os.listdir(images_folder)), " images found in the dataset folder.")
    print(len(os.listdir(annotations_folder)), " annotations found in the dataset folder.")

    if split:
        traintestval_split(images_folder, annotations_folder,
                        traintestval_folders,
                        test_size=None,# portion of dataset to use as test set
                        val_size=None, # portion of training set to use as validation set
                        random_state=None)

    print("Dataset initialization completed successfully.")
    print(f"Final dataset folder structure created at: {folder_name}")


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



def traintestval_split(images_folder, annotations_folder,
                         traintestval_folders,
                         split_type='stratified', # options: 'stratified', 'random'
                        test_size=None,# portion of dataset to use as test set
                        val_size=None, # portion of training set to use as validation set
                        random_state=None):
    
    """
    Splits image and annotation datasets into training, testing, and validation sets, and organizes them into specified folders.
    Parameters
    ----------
    images_folder : str
        Path to the folder containing image files (.tif).
    annotations_folder : str
        Path to the folder containing annotation files (.xml).
    traintestval_folders : list of str
        List of folder paths for train/test/val images and annotations in the following order:
        [train_image_folder, train_annotation_folder, test_image_folder, test_annotation_folder, val_image_folder, val_annotation_folder].
    test_size : float, optional
        Proportion of the dataset to use as the test set (default is 0.15).
    val_size : float, optional
        Proportion of the training set to use as the validation set (default is 0.15).
    random_state : int, optional
        Random seed for reproducibility (default is 42).
    Returns
    -------
    None
    Notes
    -----
    - Checks if the split folders already exist and contain the expected file types (.tif for images, .xml for annotations).
    - If the split already exists and is valid, the function prints relevant information and returns.
    - Otherwise, performs train/test/val split, copies files to their respective folders, and prints the number of files in each split.
    - Assumes that images and annotations are paired and have matching counts.
    """

    if test_size is None:
        test_size = 0.15
    if val_size is None:
        val_size = 0.15
    if random_state is None:
        random_state = 42 #default random seed for reproducibility

    # Check if split folders exist, contain expected file types, and are not empty
    image_folders = [folder for folder in traintestval_folders if 'image' in folder.lower()]
    annotation_folders = [folder for folder in traintestval_folders if 'annotation' in folder.lower()]

    folders_exist = all(os.path.exists(f) for f in traintestval_folders) # Check if all folders exist
    folders_not_empty = all(os.listdir(f) for f in traintestval_folders) # Check if all folders are not empty

    images_correct_filetype = all(file.endswith('.tif') for folder in image_folders for file in os.listdir(folder))
    annotations_correct_filetype = all(file.endswith('.xml') for folder in annotation_folders for file in os.listdir(folder))
    if not images_correct_filetype and not annotations_correct_filetype:
        print("Error: Some files in the image folders are not .tif or some files in the annotation folders are not .xml.")
 
    if folders_exist and folders_not_empty:
        print("Train/test/val split already exists and contains relevant files.")
        total_annotations = sum(len(os.listdir(f)) for f in annotation_folders)
        total_images = sum(len(os.listdir(f)) for f in image_folders)
        if total_images != total_annotations:
            print("Mismatch in image and annotation file count. Split may be outdated. Try re-extracting and splitting.")
        if total_annotations == len(total_images):
            print("Number of XML files matches total images.")
        return
    
    else:
        # format for sklearn train_test_split function requires a list of files
        # https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test

        image_files = [os.path.join(images_folder, f) for f in os.listdir(images_folder) if f.endswith('.tif')]
        annotation_files = [os.path.join(annotations_folder, f) for f in os.listdir(annotations_folder) if f.endswith('.xml')]        

        if split_type == 'random':
            print("Performing random train/test/val split.")

            # test train split
            (train_images, test_images, 
            train_annotations, test_annotations) = sklearn_ms.train_test_split(image_files, annotation_files, 
                                                                                          test_size=test_size, 
                                                                                          random_state=random_state)

            # train val split
            train_images, val_images, train_annotations, val_annotations = sklearn_ms.train_test_split(train_images, train_annotations,
                                                                                        test_size=val_size, 
                                                                                        random_state=random_state)

        if split_type == 'stratified':
            print("Performing stratified train/test/val split.")

            #train test split
            train_images, test_images, train_annotations, test_annotations = sklearn_ms.train_test_split(image_files, annotation_files,
                                                                                          test_size=test_size, 
                                                                                          random_state=random_state,
                                                                                          stratify=None)  # Stratified split based on labels

            # train val split
            train_images, val_images, train_annotations, val_annotations = sklearn_ms.train_test_split(train_images, train_annotations,
                                                                                        test_size=val_size, 
                                                                                        random_state=random_state,
                                                                                        stratify=None)  # Stratified split based on labels

        # Copy images to their respective folders
        copy_files(train_images, traintestval_folders[0], '.tif')  # train_image_folder
        copy_files(test_images, traintestval_folders[2], '.tif')   # test_image_folder
        copy_files(val_images, traintestval_folders[4], '.tif')    # val_image_folder

        # Copy annotations to their respective folders
        copy_files(train_annotations, traintestval_folders[1], '.xml')  # train_annotation_folder
        copy_files(test_annotations, traintestval_folders[3], '.xml')   # test_annotation_folder
        copy_files(val_annotations, traintestval_folders[5], '.xml')    # val_annotation_folder

        print(f"Number of train images: {len(train_images)} and annotations: {len(train_annotations)}")
        print(f"Number of test images: {len(test_images)} and annotations: {len(test_annotations)}")
        print(f"Number of val images: {len(val_images)} and annotations: {len(val_annotations)}")


def copy_files(files, destination_folder, file_extension):
    """
    Copy files to the destination folder.
    :param files: list of file paths to be copied
    :param destination_folder: folder where the files will be copied
    :param file_extension: extension of the files to be copied (e.g., '.tif' or '.xml')
    """
    
    if os.path.exists(destination_folder) and \
    any(file.endswith(file_extension) for file in os.listdir(destination_folder)) and \
    len(os.listdir(destination_folder)) > 0:
        print(f"Destination directory already exists and directory already contains {file_extension} files.")
            
    else:
        if not os.path.exists(destination_folder):
            os.makedirs(destination_folder)
        for index, file in enumerate(files):
            file_name = f"{index}{file_extension}"  # Derive the file name from an object in a list and append the file extension
            dest_file_path = os.path.join(destination_folder, file_name)  # Create the destination file path by joining the destination folder and the file name
            shutil.copy(file, dest_file_path) 

        print(f'Files copied to {destination_folder}') # Print a message indicating that the files have been copied to the destination folder


def find_and_extract_image_annotation_pairs(datasets_to_use, output_folder):
    """
    For each dataset in datasets_to_use, finds pairs of raw image folders and their corresponding annotation folders,
    then extracts matching .xml and .tif files to the output_folder with unique sequential filenames.

    Args:
        datasets_to_use (list of str): List of dataset paths.
        output_folder (str): Destination folder where 'Annotations' and 'Images' subfolders will be created.

    Returns:
        image_annotation_pairs: dict mapping dataset path to its {raw_image_folder: annotation_folder} dict
        unmatched_folders: dict mapping dataset path to its list of unmatched raw image folders
    """
    image_annotation_pairs = {}
    unmatched_folders = {}

    # Ensure output subfolders exist
    annotations_out = os.path.join(output_folder, "Annotations")
    images_out = os.path.join(output_folder, "Images")
    os.makedirs(annotations_out, exist_ok=True)
    os.makedirs(images_out, exist_ok=True)

    idx = 1
    for dataset in datasets_to_use:
        print(f"Processing dataset: {dataset}")
        dataset_pairs = {}
        dataset_unmatched = []

        # Find all raw image folders and their corresponding annotation folders
        dataset_path = pathlib.Path(dataset)
        found = False
        for root, dirs, files in os.walk(dataset_path):
                for dir_name in dirs:
                # Expect structure: .../Raw_Images/<subfolder>
                    if root.endswith('Raw_Images'):
                        raw_image_directory = os.path.join(root, dir_name)
                        found = True
                        # The annotation folder should be: .../<subfolder>/VOC
                        parent_folder = pathlib.Path(root).parent
                        mia_folder = os.path.join(parent_folder, dir_name, 'VOC')
                        #print(f"Raw image directory: {raw_image_directory}")
                        #print(f"Annotation folder: {mia_folder}")
                        #uncomment the above two lines if debugging is needed

                        if os.path.exists(mia_folder) and os.path.exists(raw_image_directory):
                            # Find all .xml files in annotation folder (recursively)
                            xml_files = []
                            for r, d, f in os.walk(mia_folder):
                                for file in f:
                                    if file.lower().endswith('.xml'):
                                        xml_files.append(os.path.join(r, file))
                            annotation_basenames = set(os.path.splitext(os.path.basename(f))[0] for f in xml_files)
    
                            # Find all .tif files in image folder (recursively)
                            tif_files = []
                            for r, d, f in os.walk(raw_image_directory):
                                for file in f:
                                    if file.lower().endswith('.tif'):
                                        tif_files.append(os.path.join(r, file))
                            image_dict = {os.path.splitext(os.path.basename(f))[0]: f for f in tif_files}
    
                            # Extract matching pairs
                            for base in sorted(annotation_basenames):
                                if base in image_dict:
                                    src_xml = None
                                    # Find the actual xml file path
                                    for xml_path in xml_files:
                                        if os.path.splitext(os.path.basename(xml_path))[0] == base:
                                            src_xml = xml_path
                                            break
                                    src_tif = image_dict[base]
                                    dst_xml = os.path.join(annotations_out, f"{idx}.xml")
                                    dst_tif = os.path.join(images_out, f"{idx}.tif")
                                    shutil.copy2(src_xml, dst_xml)
                                    shutil.copy2(src_tif, dst_tif)
                                    idx += 1
                                    #print(f"Copied annotation '{src_xml}' to '{dst_xml}' "
                                            #f"and image '{src_tif}' to '{dst_tif}'.")
                                    dataset_pairs[src_tif] = src_xml
                                else:
                                    print(f"Image for annotation '{base}' not found. Skipping.")
                        else:
                            if not os.path.exists(raw_image_directory):
                                dataset_unmatched.append(raw_image_directory)
                                print(f"Raw image folder missing: {raw_image_directory}")
                            if not os.path.exists(mia_folder):
                                dataset_unmatched.append(mia_folder)
                                print(f"Annotation folder missing: {mia_folder}")
    
        if not found:
            print(f"No Raw_Images folders found in {dataset}")

        image_annotation_pairs[dataset] = dataset_pairs
        unmatched_folders[dataset] = dataset_unmatched

    print(f"Copied {idx-1} annotation-image pairs to '{output_folder}'.")

    # uncomment the following lines if debugging is needed
    try:
        proceed = input("Look at debugging info for UNMATCHED folders? (yes/no): ").strip().lower()
    except Exception:
        proceed = 'no'
    
    if proceed == 'yes': # Print unmatched folders dict line by line for easy debugging
        if any(unmatched_folders.values()):
            print("Unmatched folders found:")
            for dataset, folders in unmatched_folders.items():
                print(f"Dataset: {dataset}")
                if folders:
                    for folder in folders:
                        print(f"  - {folder}")
                else:
                    print("  (no unmatched folders)")

    # uncomment the following lines if debugging if needed
    try:
        proceed = input("Look at debugging info for MATCHED folders? (yes/no): ").strip().lower()
    except Exception:
        proceed = 'no'

    if proceed == 'yes': #Print matched folders dict line by line for easy debugging
        for dataset, folders in image_annotation_pairs.items():
            print(f"Dataset: {dataset}")
            for img, ann in folders.items():
                print(f"  Image: {img}")
                print(f"    Annotation: {ann}")

    if not unmatched_folders:
        print("All images had annotation pairs.")

    return image_annotation_pairs, unmatched_folders




###################################################### FINAL TRAINING DATASET CREATION ##########################################################
# This script creates the final dataset folder structure for training, validation, and testing.
# Create final dataset folder structure


if __name__ == "__main__":
    if args.split:
        initialize_dataset_folder(datasets_to_use, split=True)
    else:
        initialize_dataset_folder(datasets_to_use, split=False)

    folder_name = pathlib.Path('3_TrainingData') / f'{output_folder.name}_Detector'
    utils.create_dataset_folder(str(folder_name), Split=True, Augmented=True, model_type='classifier')



    # Compile training data from multiple datasets and augmentations into a single folder
    datasets_to_use_dict = {
        "train": [],
        "val": [],
        "test": []
    }


    # Path to the main augmentation directory
    directory = os.path.join(args.augmentation_root, f'{output_folder.name}_Detector')
    augmentation_folders = [
        os.path.join(directory, folder)
        for folder in os.listdir(directory)
        if os.path.isdir(os.path.join(directory, folder))
    ]

    idx = 1
    for aug in augmentation_folders:
        print(f"Augmentation folder {idx}: {aug}")
        idx += 1

    # all augmentation folders to the training set
    for augmentations in augmentation_folders:
        datasets_to_use_dict["train"].append(augmentations)



    # original training set without augmentation
    datasets_to_use_dict["train"].append(str(output_folder / 'Split' / 'train'))
    datasets_to_use_dict["val"].append(str(output_folder / 'Split' / 'val'))
    datasets_to_use_dict["test"].append(str(output_folder / 'Split' / 'test'))


    # print the datasets to be used for compilation
    print("Datasets to use for compilation:")
    for split, paths in datasets_to_use_dict.items():
        print(f"{split}:")
        for path in paths:
            print(f"  - {path}")

    # compile the datasets into the final output folder
    folder_name = 'All_labels'  # Name of the final dataset folder
    compiled_output_folder = pathlib.Path('3_TrainingData', folder_name)
    print(f"Output folder for compiled dataset: {compiled_output_folder}")
    utils.compile_trainingdata(datasets_to_use_dict,
                            output_folder=compiled_output_folder,
                            include_augmentation_list=True)  # Set to True to include augmentation list in dataset creation 


    #object summary after all pre-processing -------------------------------------------------------
    #train set object summary
    annotations_folder = pathlib.Path('3_TrainingData') / f'{output_folder.name}_Detector_Augmented' / 'Split' / 'train' / 'annotations'
    train_object_summary = od_utils.collate_object_types(annotations_folder)  # Collate and count object types in the annotations

    #val set object summary
    annotations_folder = pathlib.Path('3_TrainingData') / f'{output_folder.name}_Detector_Augmented' / 'Split' / 'val' / 'annotations'
    val_object_summary = od_utils.collate_object_types(annotations_folder)  # Collate and count object types in the annotations

    #test set object summary
    annotations_folder = pathlib.Path('3_TrainingData') / f'{output_folder.name}_Detector_Augmented' / 'Split' / 'test' / 'annotations'
    test_object_summary = od_utils.collate_object_types(annotations_folder)  # Collate and count object types in the annotations    


    # ----------------------------------------------------- LABEL CONVERSION --------------------------------------------------------
    folders_for_conversion = [str(pathlib.Path('3_TrainingData') / f'{output_folder.name}_Detector_Augmented' / 'Split' / part / 'annotations') for part in ('train', 'val', 'test')]

    # Run label_handling for each annotation folder
    model_types = ['species_classification']  # Options: 'detector', 'classifier', 'cell_counter', 'detector_and_classifier'
    for model_type in model_types:
        print(f"Processing label handling for model type: {model_type}")
        for folder in folders_for_conversion:
            print(f"Converting labels in folder: {folder} to model type: {model_type}")
        od_utils.label_handling(folders_for_conversion,
                            model_type=model_type,  # options: 'object_detector', 'object_classifier', 'cell_counter', 'detector_and_classifier'
                            clean_labels=False,
                            convert_labels=True)
            

    folders_for_conversion = [str(pathlib.Path('3_TrainingData') / f'{output_folder.name}_Detector_Augmented' / 'Counter' / part / 'Annotations') for part in ('train', 'val', 'test')]

    # Run label_handling for each annotation folder
    model_types = ['count_classification']  # Options: 'detector', 'classifier', 'cell_counter', 'detector_and_classifier'
    for model_type in model_types:
        print(f"Processing label handling for model type: {model_type}")
        for folder in folders_for_conversion:
            print(f"Converting labels in folder: {folder} to model type: {model_type}")
        od_utils.label_handling(folders_for_conversion,
                            model_type=model_type,  # options: 'object_detector', 'object_classifier', 'cell_counter', 'detector_and_classifier'
                            clean_labels=True,
                            convert_labels=True)
            


    #object summary -------------------------------------------------------

    #train set object summary
    annotations_folder = pathlib.Path(r'3_TrainingData\Mort24_Greenland24_Patagonia24_20251107_Detector_Augmented\Split\train\annotations')
    train_object_summary = od_utils.collate_object_types(annotations_folder)  # Collate and count object types in the annotations

    #val set object summary
    annotations_folder = pathlib.Path(r'3_TrainingData\Mort24_Greenland24_Patagonia24_20251107_Detector_Augmented\Split\val\annotations')
    val_object_summary = od_utils.collate_object_types(annotations_folder)  # Collate and count object types in the annotations

    #test set object summary
    annotations_folder = pathlib.Path(r'3_TrainingData\Mort24_Greenland24_Patagonia24_20251107_Detector_Augmented\Split\test\annotations')
    test_object_summary = od_utils.collate_object_types(annotations_folder)  # Collate and count object types in the annotations  



    # misc formatting -----------------------------------------------
    folders_for_conversion = [str(pathlib.Path('3_TrainingData') / f'{output_folder.name}_Detector_Augmented' / 'Split' / part / 'annotations') for part in ('train', 'val', 'test')]


    for folder in folders_for_conversion:
        od_utils.collate_object_types(folder)  # Collate and count object types in the annotations
        od_utils.remove_objects_from_labels(folder, 
                                            remove_labels=['A_morph'])  # Case-sensitive labels
        od_utils.collate_object_types(folder)  # Collate and count object types in the annotations

    for folder in folders_for_conversion:
        od_utils.collate_object_types(folder)  # Collate and count object types in the annotations

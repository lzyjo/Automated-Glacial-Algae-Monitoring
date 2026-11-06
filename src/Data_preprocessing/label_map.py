import os
import pandas as pd

label_classes_path = os.path.abspath(r"label_classes.csv") # Load label classes from CSV
label_classes_df = pd.read_csv(label_classes_path) # read csv

# Define label maps for different model types

# Object Detection and Classification model (ODOC) - detects and classifies species and count---------------
label_classes_ODOC = tuple(label_classes_df.iloc[:, 0].tolist())  # Original labels from CSV (case preserved)
label_map_ODOC = {k.lower() : v + 1 for v, k in enumerate(label_classes_ODOC)}
                    # Convert keys to lowercase for case-insensitive matching
label_map_ODOC['background'] = 0  # Background is the first class
rev_label_map_ODOC = {v: k for k, v in label_map_ODOC.items()}  # Inverse mapping



# Detector model - detects (and classifies) species only (e.g., 'AA', 'AN', 'A_morph'), ignoring counts---------------------------------------
def extract_species(label):
    return str(label).split('_')[0] if '_' in str(label) else str(label)

label_classes_Detector = ['AA', 'AN', 'A_morph']  # Unique species labels
label_map_Detector = {k.lower(): v + 1 for v, k in enumerate(label_classes_Detector)}
label_map_Detector['background'] = 0  # Background is the first class
rev_label_map_Detector = {v: k for k, v in label_map_Detector.items()}  # Inverse mapping

# Detector_AAAN model - detects (and classifies) species only (e.g., 'AA', 'AN'), ignoring counts---------------------------------------
label_classes_Detector_AAAN = ['AA', 'AN']
label_map_Detector_AAAN = {k.lower(): v + 1 for v, k in enumerate(label_classes_Detector_AAAN)}
label_map_Detector_AAAN['background'] = 0  # Background is the first class
rev_label_map_Detector_AAAN = {v: k for k, v in label_map_Detector_AAAN.items()}  # Inverse mapping

# Counter model - counts number of cells in an image, labels are just numbers--------------------------------------
# Extract only the numbers from each label (e.g., 'AN_4' -> '4', 'AA_1' -> '1', 'AN_11' -> '11')
def extract_number(label):
    parts = str(label).split('_')
    return int(parts[1]) if len(parts) > 1 and parts[1].isdigit() else None

# Counter labels: keep only the numeric suffix, drop Nones, drop duplicates (first occurrence kept)
unique_nums = []
seen = set()
for lbl in label_classes_df.iloc[:, 0].tolist():
    n = extract_number(lbl)
    if n is None or n in seen:
        continue
    seen.add(n)
    unique_nums.append(n)

label_classes_counter = tuple(str(n) for n in unique_nums)  # e.g., ('1', '2', '3', ...) 
                        # convert to strings for label map keys
                        # to match with parse_annotation function which reads labels as strings
                        # so that label map keys are strings too
                        # reading {'1': 1, '2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7, '8': 8, '9': 9, '10': 10, '11': 11, '12': 12, '13': 13, '16': 14, '23': 15, 'background': 0}
                        # instead of e.g. {1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6, 7: 7, 8: 8, 9: 9, 10: 10, 11: 11, 12: 12, 13: 13, 16: 14, 23: 15, 'background': 0} 
label_map_Counter = {k: v + 1 for v, k in enumerate(label_classes_counter)}  # start at 1
label_map_Counter['background'] = 0
rev_label_map_Counter = {v: k for k, v in label_map_Counter.items()}

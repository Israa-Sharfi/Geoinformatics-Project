import os
import rasterio
from rasterio.enums import Resampling
from rasterio.warp import calculate_default_transform, reproject
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, Normalize
from sklearn.metrics import confusion_matrix
import seaborn as sns

# Input directories for HR and LR rasters
hr_dir = r"C:\Users\Israa\Desktop\Land cover maps\HR"
lr_dir = r"C:\Users\Israa\Desktop\Land cover maps\LR"

# Output directory for the processed rasters
output_dir = r"C:\Users\Israa\Desktop\Land cover maps\output"
output_processed_dir = os.path.join(output_dir, "processed")
output_agreement_dir = os.path.join(output_dir, "agreement_disagreement")
output_report_dir = os.path.join(output_dir, "reports")

# Ensure output directories exist
os.makedirs(output_processed_dir, exist_ok=True)
os.makedirs(output_agreement_dir, exist_ok=True)
os.makedirs(output_report_dir, exist_ok=True)

# Define the class mapping from LR to HR, excluding classes 120 and 150
class_mapping_lr_to_hr = {
    10: [4], 11: [4], 12: [4], 20: [4], 30: [4], 40: [4],  # LR Agriculture -> HR Agriculture
    60: [2], 61: [2], 70: [2], 80: [2], 90: [2], 100: [2], # LR Forest -> HR Forest
    110: [8], 130: [8],                                   # LR Grassland -> HR Rangeland
    180: [3],                                              # LR Wetland -> HR Wetland
    190: [5],                                              # LR Settlement -> HR Settlement
    200: [6], 201: [6], 202: [6],                          # LR Bare area -> HR Bare area
    210: [1],                                              # LR Water -> HR Water
    220: [7]                                               # LR Permanent snow and ice -> HR Snow/Ice
}

# Define the disagreement matrix based on the provided table
disagreement_matrix = np.array([
    [0, 12, 13, 14, 15, 16, 17, 18],
    [21, 0, 23, 24, 25, 26, 27, 28],
    [31, 32, 0, 34, 35, 36, 37, 38],
    [41, 42, 43, 0, 45, 46, 47, 48],
    [51, 52, 53, 54, 0, 56, 57, 58],
    [61, 62, 63, 64, 65, 0, 67, 68],
    [71, 72, 73, 74, 75, 76, 0, 78],
    [81, 82, 83, 84, 85, 86, 87, 0]
])

# Define class names and colors
class_names_and_colors = {
    1: ('Water', '#419bdf'),
    2: ('Forest', '#397d49'),
    3: ('Wetland', '#7a87c6'),
    4: ('Agriculture', '#e49635'),
    5: ('Settlement', '#c4281b'),
    6: ('Bare area', '#a59b8f'),
    7: ('Snow/Ice', '#a8ebff'),
    8: ('Rangeland', '#e3e2c3')
}

# Create the color map using hex codes directly
color_map = {k: v[1] for k, v in class_names_and_colors.items()}

# Function to convert hex to RGB
def hex_to_rgb(hex_color):
    return tuple(int(hex_color[i:i+2], 16) for i in (1, 3, 5))

# Create a ListedColormap for plotting
color_values = [hex_to_rgb(v[1]) for k, v in class_names_and_colors.items()]
cmap = ListedColormap([v[1] for k, v in class_names_and_colors.items()], name='landcover_cmap')

# Function to reproject HR image to 10m resolution
def reproject_hr_image_to_10m(hr_image_path):
    with rasterio.open(hr_image_path) as hr_image:
        transform, width, height = calculate_default_transform(
            hr_image.crs, hr_image.crs, hr_image.width, hr_image.height, *hr_image.bounds, resolution=(10, 10)
        )
        kwargs = hr_image.meta.copy()
        kwargs.update({
            'transform': transform,
            'width': width,
            'height': height,
            'dtype': rasterio.uint8,
            'nodata': 255  # Set NoData value to 255
        })
        hr_reprojected = np.empty((height, width), dtype=rasterio.uint8)
        reproject(
            source=rasterio.band(hr_image, 1),
            destination=hr_reprojected,
            src_transform=hr_image.transform,
            src_crs=hr_image.crs,
            dst_transform=transform,
            dst_crs=hr_image.crs,
            resampling=Resampling.bilinear
        )
        return hr_reprojected, kwargs

# Function to apply class mapping
def apply_class_mapping_lr_to_hr(lr_image, class_mapping):
    mapped_image = np.full_like(lr_image, fill_value=255, dtype=np.uint8)  # Set initial value to 255 (nodata for uint8)
    for lr_class, hr_classes in class_mapping.items():
        if hr_classes:
            hr_class_value = hr_classes[0]  # Assume one-to-one mapping for simplicity
            mapped_image[lr_image == lr_class] = hr_class_value
    return mapped_image

# Function to compute confusion matrix
def compute_confusion_matrix(hr_image, lr_image, class_names):
    # Flatten the arrays to compare pixel by pixel
    hr_flat = hr_image.flatten()
    lr_flat = lr_image.flatten()
    
    # Compute confusion matrix
    cm = confusion_matrix(hr_flat, lr_flat, labels=range(1, len(class_names) + 1))
    return cm

# Function to` plot confusion matrix
def plot_confusion_matrix(cm, class_names, output_path):
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted Class (LR)')
    plt.ylabel('True Class (HR)')
    plt.title('Confusion Matrix')
    plt.savefig(output_path)
    plt.show()

# Function to generate agreement-disagreement map and calculate statistics
def generate_agreement_disagreement_map(hr_image, lr_image, agreement_path, hr_meta, report_path):
    agreement_disagreement = np.zeros_like(hr_image, dtype=np.uint8)
    disagreement_counts = np.zeros_like(disagreement_matrix, dtype=np.int32)
    total_pixels = hr_image.size

    # Calculate agreement and disagreement
    for i in range(8):  # Class indices start from 0
        for j in range(8):
            mask = (hr_image == i + 1) & (lr_image == j + 1)
            disagreement_value = disagreement_matrix[i, j]
            agreement_disagreement[mask] = disagreement_value
            disagreement_counts[i, j] = np.sum(mask)

    # Calculate total agreement and disagreement
    total_agreement = disagreement_counts[disagreement_matrix == 0].sum()
    total_disagreement = total_pixels - total_agreement
    percentage_agreement = (total_agreement / total_pixels) * 100
    percentage_disagreement = 100 - percentage_agreement

    # Prepare data for the Excel report
    report_data = {
        'HR Class': [],
        'LR Class': [],
        'Disagreement Value': [],
        'Pixel Count': []
    }

    for i in range(8):
        for j in range(8):
            report_data['HR Class'].append(i + 1)
            report_data['LR Class'].append(j + 1)
            report_data['Disagreement Value'].append(disagreement_matrix[i, j])
            report_data['Pixel Count'].append(disagreement_counts[i, j])

    # Create a DataFrame and write to Excel
    df = pd.DataFrame(report_data)
    summary_data = {
        'Total Pixels': [total_pixels],
        'Total Agreement': [total_agreement],
        'Percentage Agreement': [percentage_agreement],
        'Total Disagreement': [total_disagreement],
        'Percentage Disagreement': [percentage_disagreement]
    }
    summary_df = pd.DataFrame(summary_data)

    with pd.ExcelWriter(report_path) as writer:
        summary_df.to_excel(writer, sheet_name='Summary', index=False)
        df.to_excel(writer, sheet_name='Disagreement Details', index=False)

    # Save the agreement-disagreement map
    with rasterio.open(agreement_path, 'w', **hr_meta) as dst:
        dst.write(agreement_disagreement.astype(rasterio.uint8), 1)
    
    # Create a color ramp for visualization
    cmap = plt.cm.get_cmap('viridis', np.max(disagreement_matrix) - np.min(disagreement_matrix) + 1)
    norm = Normalize(vmin=np.min(disagreement_matrix), vmax=np.max(disagreement_matrix))

    # Plot the agreement-disagreement map
    plt.figure(figsize=(10, 10))
    plt.imshow(agreement_disagreement, cmap=cmap, norm=norm, interpolation='none')
    plt.colorbar(label='Disagreement Level')
    plt.title(f'Agreement-Disagreement Map: {os.path.basename(agreement_path)}')
    plt.show()

# Function to process each pair of HR and LR rasters
def process_rasters(hr_path, lr_path, processed_path, agreement_path, report_path, class_names):
    print(f"Processing HR: {hr_path}, LR: {lr_path}")

    # Reproject HR image to 10m resolution
    hr_reprojected, hr_meta = reproject_hr_image_to_10m(hr_path)

    # Apply class mapping to the LR data
    with rasterio.open(lr_path) as low_res_raster:
        # Reproject LR image to match HR raster's CRS, resolution, and extent
        lr_reprojected = np.empty((hr_reprojected.shape[0], hr_reprojected.shape[1]), dtype=rasterio.uint8)
        reproject(
            source=rasterio.band(low_res_raster, 1),
            destination=lr_reprojected,
            src_transform=low_res_raster.transform,
            src_crs=low_res_raster.crs,
            dst_transform=hr_meta['transform'],
            dst_crs=hr_meta['crs'],
            resampling=Resampling.nearest
        )

        # Map LR classes to HR classes
        mapped_lr_data = apply_class_mapping_lr_to_hr(lr_reprojected, class_mapping_lr_to_hr)

    # Save the reprojected and aligned LR raster
    with rasterio.open(processed_path, 'w', **hr_meta) as dst:
        dst.write(mapped_lr_data.astype(rasterio.uint8), 1)

    # Compute confusion matrix
    cm = compute_confusion_matrix(hr_reprojected, mapped_lr_data, class_names)

    # Plot and save the confusion matrix
    confusion_matrix_path = os.path.join(output_report_dir, f'{os.path.basename(lr_path).replace(".tif", "")}_confusion_matrix.png')
    plot_confusion_matrix(cm, class_names, confusion_matrix_path)

    # Generate and save the agreement-disagreement map and write statistics to Excel
    generate_agreement_disagreement_map(hr_reprojected, mapped_lr_data, agreement_path, hr_meta, report_path)

# Get all TIFF files from the HR and LR directories
hr_files = [os.path.join(hr_dir, f) for f in os.listdir(hr_dir) if f.endswith('.tif')]
lr_files = [os.path.join(lr_dir, f) for f in os.listdir(lr_dir) if f.endswith('.tif')]

# Print HR and LR files for debugging
print("HR Files:")
for f in hr_files:
    print(f)
print("LR Files:")
for f in lr_files:
    print(f)

# Ensure the files are sorted to match HR and LR pairs (if necessary, based on your naming convention)
hr_files.sort()
lr_files.sort()

# Matching LR files to HR files based on the common part of the filename
def normalize_filename(filename):
    base = os.path.basename(filename)
    base = base.replace(' HR', '').replace(' LR', '').replace('_reprojected', '')
    return base.split('.')[0]

def match_files(hr_files, lr_files):
    hr_dict = {normalize_filename(f): f for f in hr_files}
    lr_dict = {normalize_filename(f): f for f in lr_files}
    matched_pairs = [(hr_dict[key], lr_dict[key]) for key in hr_dict if key in lr_dict]
    return matched_pairs

matched_pairs = match_files(hr_files, lr_files)
 

# Print matched pairs for debugging
print("Matched Pairs:")
for hr_path, lr_path in matched_pairs:
    print(f"HR: {hr_path}, LR: {lr_path}")

# Loop through each pair of HR and LR rasters
for hr_path, lr_path in matched_pairs:
    base_name = os.path.basename(lr_path).replace('.tif', '')
    
    processed_path = os.path.join(output_processed_dir, f'{base_name}_processed.tif')
    agreement_path = os.path.join(output_agreement_dir, f'{base_name}_agreement_disagreement.tif')
    report_path = os.path.join(output_report_dir, f'{base_name}_report.xlsx')
    
    print(f'Processing:\n  HR: {hr_path}\n  LR: {lr_path}\n  Processed: {processed_path}\n  Agreement: {agreement_path}\n  Report: {report_path}')
    
    process_rasters(hr_path, lr_path, processed_path, agreement_path, report_path, [v[0] for k, v in class_names_and_colors.items()])

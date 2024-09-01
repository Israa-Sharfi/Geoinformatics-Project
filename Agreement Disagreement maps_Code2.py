import os
import numpy as np
import pandas as pd
import rasterio
from rasterio.warp import reproject
from rasterio.enums import Resampling
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from sklearn.metrics import confusion_matrix

# Define mappings
class_mapping_lr_to_hr = {
    10: [4], 11: [4], 12: [4], 20: [4], 30: [4], 40: [4],
    60: [2], 61: [2], 70: [2], 80: [2], 90: [2], 100: [2],
    110: [8], 130: [8],
    180: [3],
    190: [5],
    200: [6], 201: [6], 202: [6],
    210: [1],
    220: [7]
}

class_mapping_hr_to_hr_aligned = {
    1: [1], 2: [2], 4: [3], 5: [4], 7: [5], 8: [6], 9: [7], 11: [8]
}

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

def reclassify_image(src_image, class_mapping, excluded_classes=None):
    if excluded_classes is None:
        excluded_classes = []
    
    with rasterio.open(src_image) as src:
        data = src.read(1)
        
        reclassified_data = np.zeros_like(data)
        for old_class, new_classes in class_mapping.items():
            reclassified_data[np.isin(data, old_class)] = new_classes[0]
        
        for ex_class in excluded_classes:
            reclassified_data[data == ex_class] = 0
        
        metadata = src.meta.copy()
        metadata.update(dtype=rasterio.uint8, count=1)
    
    return reclassified_data, metadata

def align_hr_to_lr(lr_image, hr_image, output_image):
    with rasterio.open(lr_image) as lr_src, rasterio.open(hr_image) as hr_src:
        lr_meta = lr_src.meta.copy()
        lr_transform = lr_src.transform
        
        hr_data = hr_src.read(1)
        hr_meta = hr_src.meta.copy()
        hr_meta.update({
            'height': lr_meta['height'],
            'width': lr_meta['width'],
            'transform': lr_transform,
            'crs': lr_meta['crs']
        })
        
        resampled_hr_data = np.empty((lr_meta['height'], lr_meta['width']), dtype=np.uint8)
        reproject(
            source=hr_data,
            destination=resampled_hr_data,
            src_transform=hr_src.transform,
            src_crs=hr_src.crs,
            dst_transform=lr_transform,
            dst_crs=lr_meta['crs'],
            resampling=Resampling.nearest
        )
        
        with rasterio.open(output_image, 'w', **hr_meta) as dst:
            dst.write(resampled_hr_data, 1)

def create_agreement_disagreement_map(lr_data, hr_data, disagreement_matrix, output_image):
    """
    Create an agreement/disagreement map between LR and HR images based on the provided disagreement matrix.
    """
    error_map = np.full(lr_data.shape, -1)  # -1 indicates undefined
    
    # Populate the error map based on the disagreement matrix
    for hr_class in range(1, 9):  # HR classes range from 1 to 8
        for lr_class in range(1, 9):  # LR classes range from 1 to 8
            disagreement_value = disagreement_matrix[hr_class-1, lr_class-1]
            if disagreement_value != 0:
                error_map[(lr_data == lr_class) & (hr_data == hr_class)] = disagreement_value
    
    # Define colors for visualization
    cmap = ListedColormap(plt.cm.viridis(np.linspace(0, 1, 256)))
    unique_vals = np.unique(error_map[error_map != -1])
    norm = BoundaryNorm([0] + list(unique_vals), cmap.N, clip=True)
    
    plt.figure(figsize=(10, 10))
    img = plt.imshow(error_map, cmap=cmap, norm=norm)
    plt.title('Agreement/Disagreement Map')
    
    # Create custom legend
    legend_labels = []
    for val in unique_vals:
        hr_class = (val // 10) % 10
        lr_class = val % 10
        hr_class_name = class_names_and_colors[hr_class][0]
        lr_class_name = class_names_and_colors[lr_class][0]
        legend_labels.append(f"{hr_class_name} vs {lr_class_name}")
    
    cbar = plt.colorbar(img, ticks=list(unique_vals), format='%d')
    cbar.ax.set_yticklabels(legend_labels)
    
    plt.savefig(output_image)
    plt.show()

def compute_error_matrix(lr_data, hr_data):
    """
    Compute the error matrix and related metrics.
    """
    # Flatten arrays and compute confusion matrix
    lr_flat = lr_data.flatten()
    hr_flat = hr_data.flatten()
    conf_matrix = confusion_matrix(hr_flat, lr_flat, labels=range(1, 9))

    # Calculate metrics
    total_samples = np.sum(conf_matrix)
    correct_classifications = np.trace(conf_matrix)
    overall_accuracy = correct_classifications / total_samples

    # User's and Producer's Accuracy
    user_accuracy = np.diag(conf_matrix) / np.sum(conf_matrix, axis=0)
    prod_accuracy = np.diag(conf_matrix) / np.sum(conf_matrix, axis=1)

    # Kappa coefficient
    expected_agreement = np.sum(np.sum(conf_matrix, axis=0) * np.sum(conf_matrix, axis=1)) / (total_samples ** 2)
    kappa = (overall_accuracy - expected_agreement) / (1 - expected_agreement)

    return conf_matrix, overall_accuracy, user_accuracy, prod_accuracy, kappa

def generate_report(conf_matrix, overall_accuracy, user_accuracy, prod_accuracy, kappa, output_file):
    """
    Generate a report from the confusion matrix and metrics and save it to an Excel file.
    """
    # Define class labels for the confusion matrix
    class_labels = [class_names_and_colors[i][0] for i in range(1, 9)]  # ['Water', 'Forest', 'Wetland', 'Agriculture', 'Settlement', 'Bare area', 'Snow/Ice', 'Rangeland']
    
    # Create confusion matrix DataFrame with class names
    conf_matrix_df = pd.DataFrame(conf_matrix, columns=[f"LR_{label}" for label in class_labels],
                                  index=[f"HR_{label}" for label in class_labels])
    
    # Create a DataFrame for metrics
    metrics_df = pd.DataFrame({
        'Metric': ['Overall Accuracy', 'Kappa Coefficient'],
        'Value': [overall_accuracy, kappa]
    })
    
    # User's Accuracy DataFrame
    user_accuracy_df = pd.DataFrame({
        'Class': class_labels,
        'User\'s Accuracy': user_accuracy
    })
    
    # Producer's Accuracy DataFrame
    prod_accuracy_df = pd.DataFrame({
        'Class': class_labels,
        'Producer\'s Accuracy': prod_accuracy
    })

    # Save to Excel
    with pd.ExcelWriter(output_file) as writer:
        conf_matrix_df.to_excel(writer, sheet_name='Confusion Matrix')
        metrics_df.to_excel(writer, sheet_name='Metrics', index=False)
        user_accuracy_df.to_excel(writer, sheet_name='User\'s Accuracy', index=False)
        prod_accuracy_df.to_excel(writer, sheet_name='Producer\'s Accuracy', index=False)

def load_image(image_path):
    with rasterio.open(image_path) as src:
        data = src.read(1)
        meta = src.meta
    return data, meta

def process_images(lr_folder, hr_folder, lr_output_folder, hr_output_folder, comparison_output_folder):
    for year in range(2017, 2023):
        lr_filename = f"{year} LR.tif"
        hr_filename = f"{year} HR.tif"
        lr_path = os.path.join(lr_folder, lr_filename)
        hr_path = os.path.join(hr_folder, hr_filename)
        aligned_lr_path = os.path.join(lr_output_folder, f"{year}LR_aligned.tif")
        aligned_hr_path = os.path.join(hr_output_folder, f"{year}HR_aligned.tif")
        agreement_disagreement_map_path = os.path.join(comparison_output_folder, f"{year}_agreement_disagreement_map.png")
        report_file = os.path.join(comparison_output_folder, f"{year}_report.xlsx")
        
        reclassified_lr, lr_meta = reclassify_image(lr_path, class_mapping_lr_to_hr, excluded_classes=[120, 150])
        with rasterio.open(aligned_lr_path, 'w', **lr_meta) as dst:
            dst.write(reclassified_lr, 1)
        
        align_hr_to_lr(aligned_lr_path, hr_path, aligned_hr_path)
        
        reclassified_hr, hr_meta = reclassify_image(aligned_hr_path, class_mapping_hr_to_hr_aligned, excluded_classes=[10])
        with rasterio.open(aligned_hr_path, 'w', **hr_meta) as dst:
            dst.write(reclassified_hr, 1)
        
        lr_data, _ = load_image(aligned_lr_path)
        hr_data, _ = load_image(aligned_hr_path)
        
        create_agreement_disagreement_map(lr_data, hr_data, disagreement_matrix, agreement_disagreement_map_path)
        
        conf_matrix, overall_accuracy, user_accuracy, prod_accuracy, kappa = compute_error_matrix(lr_data, hr_data)
        
        generate_report(conf_matrix, overall_accuracy, user_accuracy, prod_accuracy, kappa, report_file)

# Define paths
lr_folder = r"C:\Users\Israa\Desktop\Land cover maps\LR"
hr_folder = r"C:\Users\Israa\Desktop\Land cover maps\HR"
lr_output_folder = r"C:\Users\Israa\Desktop\Land cover maps\Code2 outputs\Aligned_LR2"
hr_output_folder = r"C:\Users\Israa\Desktop\Land cover maps\Code2 outputs\Aligned_HR2"
comparison_output_folder = r"C:\Users\Israa\Desktop\Land cover maps\Code2 outputs\Comparison_Results"

os.makedirs(lr_output_folder, exist_ok=True)
os.makedirs(hr_output_folder, exist_ok=True)
os.makedirs(comparison_output_folder, exist_ok=True)

process_images(lr_folder, hr_folder, lr_output_folder, hr_output_folder, comparison_output_folder)

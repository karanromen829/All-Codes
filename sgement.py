import os
import pandas as pd

# Input and output directories
input_folder = 'D:\\PLL project\\data\\typical'
output_base_folder = 'D:\PLL project\\segmentation\\segmentation_on_time_without_normalization\\Segmenting typical'

# Time segments
segments = [
    (0, 1e-6, 'segment_0_to_1e-6'),
    (1e-6, 2.5e-6, 'segment_1e-6_to_2.5e-6'),
    (2.5e-6, 5e-6, 'segment_2.5e-6_to_5e-6')
]

# Create output directories for each segment
for _, _, folder_name in segments:
    segment_folder = os.path.join(output_base_folder, folder_name)
    os.makedirs(segment_folder, exist_ok=True)

# Process each file in the input folder
for file_name in os.listdir(input_folder):
    if file_name.endswith('.csv'):
        file_path = os.path.join(input_folder, file_name)
        data = pd.read_csv(file_path)

        time_column = 'Sim_time'  # Replace 'Sim_time' with the actual column name if different

        for start, end, folder_name in segments:
            segment_folder = os.path.join(output_base_folder, folder_name)
            segment_data = data[(data[time_column] >= start) & (data[time_column] < end)]
            output_file_path = os.path.join(segment_folder, f"{os.path.splitext(file_name)[0]}_{folder_name}.csv")
            segment_data.to_csv(output_file_path, index=False)

print("Data segmentation completed. CSV files saved in respective folders.")

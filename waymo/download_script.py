import os


def split_file(file_path, output_folder):
    # Read the content of the text file
    with open(file_path, 'r', newline='') as file:
        lines = file.readlines()

    total_lines = len(lines)
    lines_per_portion = total_lines // 10  # Calculate the number of lines per portion

    # Create 10 equal portions
    for i in range(10):
        start_index = i * lines_per_portion
        end_index = (i + 1) * lines_per_portion if i < 9 else total_lines

        portion_content = lines[start_index:end_index]

        # Write portion to a separate file
        portion_file_path = f"{output_folder}/portion_{i + 1}.txt"
        with open(portion_file_path, 'w', newline='') as portion_file:
            portion_file.writelines(portion_content)
            
# Example usage:
# split_file("training_data.txt", "training_file_names")

def download_files(batch_number, dest_path, file_path):
    fname = 'portion_' + str(batch_number) + ".txt"
    with open(file_path + fname, 'r', newline='') as file:
        lines = file.readlines()
        
    for l in lines:
        os.system(f"gsutil -m cp \"{l}\"")
        
download_files(1, "/pvcvolume/waymo-motion/lidar_and_camera/training/", "/pvcvolume/cruz-control/waymo/")
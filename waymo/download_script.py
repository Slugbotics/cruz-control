import os


def split_file(file_path, output_folder):
    # Read the content of the text file
    with open(file_path, 'r', newline='') as file:
        lines = file.readlines()
    print(lines[2])
    total_lines = len(lines)
    lines_per_portion = total_lines // 10  # Calculate the number of lines per portion

    # Create 10 equal portions
    for i in range(10):
        start_index = i * lines_per_portion
        end_index = (i + 1) * lines_per_portion if i < 9 else total_lines

        portion_content = lines[start_index:end_index]

        # Write portion to a separate file
        portion_file_path = f"{output_folder}/portion_{i + 1}.txt"
        with open(portion_file_path, 'w', newline='\n') as portion_file:
            for x in range(start_index, end_index):
                portion_file.write(lines[x])
            
# Example usage:
# split_file("training_data.txt", "training_file_names")

def download_files(batch_number, dest_path, file_path):
    fname = 'portion_' + str(batch_number) + ".txt"
    with open(file_path + fname, 'r') as file:
        lines = file.readlines()
    os.system("gsutil -m cp \"{lines[1]}\" .")
    # for l in lines:
    #     os.system(f"gsutil -m cp \"{l}\" .")
    
download_files(1, "/pvcvolume/waymo-motion/lidar_and_camera/training/", "/pvcvolume/cruz-control/waymo/training_file_names/")
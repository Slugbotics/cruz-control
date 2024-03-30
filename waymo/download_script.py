import os
import threading
from datetime import datetime

def split_file(file_path, output_folder):
    # Read the content of the text file
    with open(file_path, 'r') as file:
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
        with open(portion_file_path, 'w', newline='') as portion_file:
            for x in range(start_index, end_index):
                portion_file.write(lines[x])
            
# Example usage:
# split_file("training_data.txt", "sync_from_list")

# file_path: path to the directory containing
# dest_path: directory for the unsynced files to be stored
def build_sync_list(sync_from, sync_to, dest_path, bucket_directory, batch_number):
    bucket_directory_str_size = len(bucket_directory)
    # creates list of files in the sync_from file
    fname = 'portion_' + str(batch_number) + ".txt"
    with open(sync_from + fname, 'r') as file:
        lines = file.readlines()
    
    # creates a set of the files present on the sync_to directory
    present = []
    for (dirpath, dirnames, filenames) in os.walk(sync_to):
        present.extend(filenames)
    print(len(present))
    print(len(lines))
    present = set(present)
    
    print(lines[1][bucket_directory_str_size::])
    unsynced = [x for x in lines if x[bucket_directory_str_size::].strip() not in present]
    
    portion_file_path = f"{dest_path}/unsync_batch_{batch_number}.txt"
    with open(portion_file_path, 'w', newline='') as portion_file:
        for x in unsynced:
            portion_file.write(x)

# build_sync_list("/pvcvolume/cruz-control/waymo/sync_from_list/", "/pvcvolume/waymo-motion/lidar_and_camera/training/", "/pvcvolume/cruz-control/waymo/unsynced/", "gs://waymo_open_dataset_motion_v_1_2_1/uncompressed/lidar_and_camera/training/", 1)    

def download_files(start_point, offset, batch_number, dest_path, file_path):
    fname = 'unsync_batch_' + str(batch_number) + ".txt"
    batch_size = 8
    with open(file_path + fname, 'r') as file:
        lines = file.readlines()
    end_point = min(start_point + 20000, len(lines))
    for l in range(spoint+offset*batch_size, end_point, 64):
        # os.system("clear")
        line_list = lines[l].strip() + " " + lines[l+1].strip() + " " + lines[l+2].strip() + " " + lines[l+3].strip() + " " + lines[l+4].strip() + " " + lines[l+5].strip() + " " + lines[l+6].strip() + " " + lines[l+7].strip()
        # print(line_list)
        os.system(f"gsutil -q cp -n {line_list} \"{dest_path}\"")
        if(l%100 < batch_size):
            time = datetime.now().strftime("%H:%M:%S")
            print(f"Currently at {l} , time: {time}")
            
def batch_download(batch_number, dest_path, file_path):
    # split_file("training_data.txt", "training_file_names")
    fname = 'unsync_batch_' + str(batch_number) + ".txt"
    os.system(f"cat {file_path + fname} | gsutil -m cp -n -I \"{dest_path}\"")

build_sync_list("/pvcvolume/cruz-control/waymo/sync_from_list/", "/pvcvolume/waymo-motion/lidar_and_camera/training/", "/pvcvolume/cruz-control/waymo/unsynced/", "gs://waymo_open_dataset_motion_v_1_2_1/uncompressed/lidar_and_camera/training/", 3)    

batch_download(3, "/pvcvolume/waymo-motion/lidar_and_camera/training/", "/pvcvolume/cruz-control/waymo/unsynced/")

# spoint = 0
# batch_num = 2
# target_dir = "/pvcvolume/waymo-motion/lidar_and_camera/training/" # this dir is for batch 3 and 4, will be merged later
# file_name_dir = "/pvcvolume/cruz-control/waymo/unsynced/" 
# t1 = threading.Thread(target=download_files, args=(spoint, 0, batch_num, target_dir, file_name_dir))
# t2 = threading.Thread(target=download_files, args=(spoint, 1, batch_num, target_dir, file_name_dir))
# t3 = threading.Thread(target=download_files, args=(spoint, 2, batch_num, target_dir, file_name_dir))
# t4 = threading.Thread(target=download_files, args=(spoint, 3, batch_num, target_dir, file_name_dir))
# t5 = threading.Thread(target=download_files, args=(spoint, 4, batch_num, target_dir, file_name_dir))
# t6 = threading.Thread(target=download_files, args=(spoint, 5, batch_num, target_dir, file_name_dir))
# t7 = threading.Thread(target=download_files, args=(spoint, 6, batch_num, target_dir, file_name_dir))
# t8 = threading.Thread(target=download_files, args=(spoint, 7, batch_num, target_dir, file_name_dir))

# t1.start()
# t2.start()
# t3.start()
# t4.start()
# t5.start()
# t6.start()
# t7.start()
# t8.start()

# t1.join()
# t2.join()
# t3.join()
# t4.join()
# t5.join()
# t6.join()
# t7.join()
# t8.join()
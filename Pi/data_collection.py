import BluetoothControl
from BMX160 import BMX160
from CAMERA import Camera
from HOKUYOLX import hokuyolx
import os
import csv
from datetime import datetime

# record timestamp and get date-time in YYYY-MM-DD HH:MM:SS:MS
def getTime():
    timestamp = datetime.now().timestamp()
    time = str(datetime.fromtimestamp(timestamp))

    # format the time string, replacing periods with colons
    time = time.replace(".",":")
    return str(time)


csvfields = ["timestamp: ", "thrust-steering: ", "imu: ", "hokuyo: "]
file_directory = str(os.path.dirname(__file__))
output_path = file_directory + "/data_out/"
fileName = getTime()

bmx = BMX160(1)
camera = Camera(0, fileName, output_path)
hokuyo = hokuyolx()

csv_out = open(output_path + str(fileName) + ".csv", "w")
csvwriter = csv.DictWriter(csv_out, fieldnames=csvfields)
csvwriter.writeheader()

while True:
    camera.read()
    
    # write to csv file using csvWriter
    csvwriter.writerow({"timestamp: ": getTime(),
               "thrust-steering: ": BluetoothControl.bluetoothControl(),
               "imu: ": bmx.get_all_data(),
               "hokuyo: ": hokuyo.read()})

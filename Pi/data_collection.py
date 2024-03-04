import BluetoothControl
from BMX160 import BMX160
from CAMERA import Camera
import os
import csv
from datetime import datetime


def record():
    # get data from bmx as array
    # ARRAY ORDER
    # magn: x, y, z, gyro: x, y, z, accel: x, y, z
    imuData= [0,1,2,3,4,5,6,7,8]# bmx.get_all_data()

    # create dictionary to write
    csvData = {"timestamp: ": getTime(),
               "thrust-steering: ": targetString,
               "imu: ": imuData}
    
    # write to csv file using csvWriter
    csvwriter.writerow(csvData)

# record timestamp and get date-time in YYYY-MM-DD HH:MM:SS:MS
def getTime():
    timestamp = datetime.now().timestamp()
    time = str(datetime.fromtimestamp(timestamp))

    # format the time string, replacing periods with colons
    time = time.replace(".",":")
    return str(time)


csvfields = ["timestamp: ", "thrust-steering: ", "imu: "]
file_directory = str(os.path.dirname(__file__))
output_path = file_directory + "/data_out/"
fileName = getTime()

targetString = "0,0"
bmx160 = BMX160(1)
camera = Camera(0, fileName, output_path)

csv_out = open(output_path + str(fileName) + ".csv", "w")
csvwriter = csv.DictWriter(csv_out, fieldnames=csvfields)
csvwriter.writeheader()

while True:
    targetString = BluetoothControl.bluetoothControl()
    camera.read()
    record()


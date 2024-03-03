# import BluetoothControl
# from BMX160 import BMX160
from CAMERA import Camera
import os
import csv
from datetime import datetime

csvfields = ["timestamp: ", "thrust-steering: ", "imu: "]
file_directory = str(os.path.dirname(__file__))
output_path = file_directory + "/csv_out/"
fileName = "a"

output = 0
# bmx160 = BMX160(1)
camera = Camera(0, fileName, output_path)

csv_out = open(output_path + str(fileName) + ".csv", "w")
csvwriter = csv.DictWriter(csv_out, fieldnames=csvfields)
csvwriter.writeheader()

def record():
    # get data from bmx as array
    # ARRAY ORDER
    # magn: x, y, z, gyro: x, y, z, accel: x, y, z
    imuData= [0,1,2,3,4,5,6,7,8]# bmx.get_all_data()
    
    # print data
    # print("magn: x: {0:.2f} uT, y: {1:.2f} uT, z: {2:.2f} uT".format(data[0],data[1],data[2]))
    # print("gyro  x: {0:.2f} g, y: {1:.2f} g, z: {2:.2f} g".format(data[3],data[4],data[5]))
    # print("accel x: {0:.2f} m/s^2, y: {1:.2f} m/s^2, z: {2:.2f} m/s^2".format(data[6],data[7],data[8]))


    # create dictionary to write
    csvData = {"timestamp: ": getTime(),
               "thrust-steering: ": 9,#targetString,
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

while True:
    # output = BluetoothControl.bluetoothControl()
    camera.read()
    record()


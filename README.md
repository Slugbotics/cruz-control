# f1tenth-2023
Added TTC node under package lab2pkg, which will brake the car if smallest_distance/velocity is under a given threshold. I believe it works as the lab describes, but is not perfect as it has high false-positive rates if the car is close to the edge of a wall. From what I understood from the lab, the algorithm is indiscriminate of the direction of velocity the car compared angle of the smallest distance and can trigger a false positive if the wall is within the scan radius but the car is not on a path to collide with it. 

I've only tested it in the ros2_gym (local install on wsl Ubuntu 20.04) with teleop controls.

Run by: 
1) adding lab2_ws as a directory in home/[USERNAME]/
2) sourcing lab2_ws/install/setup.bash 
3) using the command ros2 run lab2_pkg TTC
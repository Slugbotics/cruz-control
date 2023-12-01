# f1tenth-2023
PiD controller for lab 3, used to minimize error between target distance from wall

The PiD controller is not tuned, but seems to work fine in the simulator.

The derivative calculation might work, but it would probably be best to update it using the last value the controller outputted. (a little bit of editing could fix it)

Tested in f1tenth gym (Ubuntu 22.04)
Run by:

    1) adding lab3_ws as a directory in home/[USERNAME]/
    2) sourcing lab3_ws/install/setup.bash
    3) using the command ros2 run lab3_pkg WF

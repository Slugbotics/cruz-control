# Gazebo Models

## Moving the model

Model uses the `gazebo-ackermann-steering-system` which looks at the topic `\cmd_vel`, which needs type `geometry_msgs::Twist`. 

### Moving Message

Putting this on a new command line tab publihses the `cmd_vel` message.

```bash
 gz topic -t "/cmd_vel" -m gz.msgs.Twist -p "linear: {x: 5}, angular: {z: 5}"
```

Links:
<https://gazebosim.org/docs/garden/moving_robot>
<https://app.gazebosim.org/RudisLaboratories/fuel/models/MR-Buggy3>
<http://docs.ros.org/en/melodic/api/geometry_msgs/html/msg/Twist.html>
<https://gazebosim.org/api/msgs/6.2/classignition_1_1msgs_1_1Twist.html>
# Lane Following

## Model v1

This model had the following inputs and outputs:

- Input: Image
- Output: [steering, throttle, breaking]

### Update v1.1

The labels were not perfectly aligned from 0-1 due to the sensor being looked at (zoe), v1.1 looks at a different set of sensors that now should be mapped form 0-1.

## Model v2

This model has the following inputs and outputs:

- Input: Image, [x_acceleration, y_acceleration, z_acceleration]
- Output: [steering, throttle, breaking]

IMU data is included in the inputs for the model because it help the model be smoother as it is more aware of the forces acting on the car. An expert driver, especially race car drivers, is constantly aware of the forces applied on the car in order to ensure that the car can take corners smoothly, without loosing traction.

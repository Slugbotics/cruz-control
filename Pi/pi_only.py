import machine
import utime
# Set the GPIO pin number (based on the connection in step 2)
SERVO_PIN = 25
# Set the PWM frequency (50Hz is common for servos)
PWM_FREQUENCY = 50
# Create a PWM object with the specified pin and frequency
pwm = machine.PWM(machine.Pin(SERVO_PIN))
pwm.freq(PWM_FREQUENCY)
# Function to set the servo angle
def set_angle(angle):
# Convert the angle to a duty cycle value (0.5 ms to 2.5 ms pulse width)
duty_cycle = int((angle / 180) * (1000000 / PWM_FREQUENCY) + 2500)
# Set the duty cycle to control the servo
pwm.duty_ns(duty_cycle)
# Main loop
while True:
    # Move the servo from 0 to 180 degrees
    for angle in range(0, 180, 5):
    set_angle(angle)
    utime.sleep_ms(50)
    # Move the servo back from 180 to 0 degrees
    for angle in range(180, 0, -5):
    set_angle(angle)
    utime.sleep_ms(50) 
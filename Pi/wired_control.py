from smbus import SMBus
# from smbus2 import SMBus
import inputs

gp = inputs.devices.gamepads
if len(gp) == 0:
	raise Exception("No gamepads detected!")

def translate(value, leftMin, leftMax, rightMin, rightMax):
    # Figure out how 'wide' each range is
    leftSpan = leftMax - leftMin
    rightSpan = rightMax - rightMin

    # Convert the left range into a 0-1 range (float)
    valueScaled = float(value - leftMin) / float(leftSpan)

    # Convert the 0-1 range into a value in the right range.
    return int(rightMin + (valueScaled * rightSpan))

addr = 0x8 # Pico communication bus on address 4


def connect(port):
	global bus
	bus = SMBus(port)

def string2Bytes(val):
	retVal = []
	for c in val:
		retVal.append(ord(c))
	return retVal

def send(data):
	byteValue = string2Bytes(data)
	print(byteValue)
	bus.write_i2c_block_data(addr, 0x00, byteValue) 
	return -1
def recv():
	pass

# thrust -> -100, 100 steering -> -15, 15

if __name__ == '__main__':
	connect(1)
	thrust=0
	steering = 0
	while True:
		events = inputs.get_gamepad()
		for event in events:
			if event.code == 'ABS_X':
				steering = event.state
			if event.code == 'ABS_RZ':
				thrust = event.state
			elif event.code == 'ABS_Z':
				thrust = -event.statecombinedTriggerValue
		inp = str(translate(thrust,0, 1023, 0, 10))+","+str(translate(steering,-32700,32700,-20, 20))
#		print(inp)
#		if inp == "-1":
#			exit()
		print(inp)
		send(inp)



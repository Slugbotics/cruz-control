from smbus import SMBus

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
	bus.write_i2c_block_data(addr, 0x00, byteValue) 
	return -1
def recv():
	pass

if __name__ == '__main__':
	connect(1)
	while True:
		inp = input("Input (thrust, steering): ")
		if inp == "-1":
			exit()
		send(inp)



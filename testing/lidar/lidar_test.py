from hokuyolx import HokuyoLX
laser = HokuyoLX()

timestamp, scan = laser.get_dist() # Single measurment mode

while True:
	timestamp, scan = laser.get_dist()
	print(scan)
	
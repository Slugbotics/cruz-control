from hokyuolx import HokuyoLX

# hokuyolx laser documentation https://hokuyolx.readthedocs.io/en/latest/

class hokuyolx:
    def __init__(self):
        self.laser = HokuyoLX()
        self.laser.activate()

    def read(self):
        timestamp, scan = self.laser.get_dist()
        return scan

    def __del__(self):
        self.laser.sleep()
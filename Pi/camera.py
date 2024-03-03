import cv2, queue, threading
class Camera:
    def __init__(self, id, fileName, output_path):
        self.camera = cv2.VideoCapture(id)
        # check if camera is open
        if (self.camera.isOpened() == False):  
            raise Exception("Camera is opened by another program") 
        
        # get camera resolution
        frame_width = int(self.camera.get(3)) 
        frame_height = int(self.camera.get(4))  
        self.size = (frame_width, frame_height) 
        self.video_file = cv2.VideoWriter(output_path+str(fileName)+'.avi', cv2.VideoWriter_fourcc(*'MJPG'), 10, self.size)
    
    def read(self):
        ret, frame = self.camera.read() 
        if ret == True:
            self.video_file.write(frame)

    def __del__(self):
        self.camera.release() 
        cv2.destroyAllWindows() 

from picamera import PiCamera

from time import sleep

camera = piCamera()
camera.start_preview(alpha=192)
sleep(1)
camera.capture("/home/pi/Desktop/pic.jpg")
camera.stop_preview()

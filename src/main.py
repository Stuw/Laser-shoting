#!/usr/bin/env python3

from argparse import ArgumentParser
import cv2
import logging
import math
import numpy as np
import subprocess
import sys
import time
import traceback


logging.basicConfig(
	level=logging.INFO,
	handlers=[]
)

logFormatter = logging.Formatter("%(asctime)s [%(threadName)-12.12s] [%(levelname)-5.5s]  %(message)s")
log = logging.getLogger(__name__)

consoleHandler = logging.StreamHandler()
consoleHandler.setFormatter(logFormatter)
log.addHandler(consoleHandler)


X = 0
Y = 1

SHOT_STATE_FIRE = True
SHOT_STATE_NONE = False


def mouse_callback(action, x, y, flags, *userdata):
	self = userdata[0]

	if action == cv2.EVENT_LBUTTONDOWN:
		shot_x = (x - self.target_x) / self.target_radius
		shot_y = (y - self.target_y) / self.target_radius

		shot = (x, y)
		dist = math.sqrt(shot_x * shot_x + shot_y * shot_y)

		self.sound_shot()
		self.draw_shot(shot)
		self.update_score(shot, dist)


class Shooting:
	def __init__(self, args):
		self.args = args

		# Camera
		self.camera = None
		self.img_width = args.width
		self.img_height = args.height

		# Frame
		self.center = (int(args.centerx), int(args.centery))
		self.radius = int(args.radius)
		self.area_scale = args.area_scale
		self.top = self.get_top(args.area_scale)
		self.bottom = self.get_bottom(args.area_scale)

		# Target
		self.target_original = None
		self.target_image = None
		self.target_x = 0
		self.target_y = 0
		self.target_radius = 0

		# Shots
		self.shots_count = 0
		self.score = 0.0
		self.max_shots = args.shots

		# Shot detection
		self.shot_timeout_ms = int(args.timeout * 1000)
		self.last_shot_time = None
		self.last_shot_state = SHOT_STATE_FIRE
		self.round_timeout_ms = int(args.round_timeout * 1000)

		# Constants
		self.line_thickness = 1
		self.text_thickness = 2
		self.draw_color = (255, 0, 0)
		self.font_scale = 1
		self.font = cv2.FONT_HERSHEY_SIMPLEX


	#####################################

	def sound_shot(self):
		self.sound('Shot.wav')


	def sound_done(self):
		self.sound('924.wav')


	def sound(self, fname):
		subprocess.Popen(f'aplay "{fname}"', shell=True)


	#####################################

	def get_top(self, scale = 1.0):
		return (self.center[X] - int(self.radius * scale), self.center[Y] - int(self.radius * scale))


	def get_bottom(self, scale = 1.0):
		return (self.center[X] + int(self.radius * scale), self.center[Y] + int(self.radius * scale))

	#####################################

	def init_camera(self):
		if self.camera:
			return

		self.camera = cv2.VideoCapture(self.args.video)
		self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, self.img_width)
		self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, self.img_height)

		# we capture the first frame for the camera to adjust itself to the exposure
		#ret_val , cap_for_exposure = self.camera.read()
		#self.camera.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.75)
		#self.camera.set(cv2.CAP_PROP_EXPOSURE , -1)


	def init_target(self):
		if self.target_original:
			return

		self.target_original = cv2.imread("target.jpg", cv2.IMREAD_COLOR)
		self.target_x = float(self.target_original.shape[0])*0.5
		self.target_y = float(self.target_original.shape[1])*0.5
		self.target_radius = min(self.target_x, self.target_y)


	def reset(self):
		self.shots_count = 0
		self.score = 0
		self.target_image = self.target_original.copy()

	#####################################

	def draw_shot(self, shot):
		cv2.circle(self.target_image, shot, 5, (60,60,255),10)
		cv2.circle(self.target_image, shot, 10, (120,120,120),1)
		cv2.imshow("Result", self.target_image)


	def show_score(self):
		result = self.score * 100.0 / self.max_shots
		text = f"Your score: {result:.2f}"

		org = (215, 30)

		cv2.putText(self.target_image, text, org, self.font, 
			self.font_scale, self.draw_color, self.text_thickness, cv2.LINE_AA)

		cv2.imshow("Result", self.target_image)


	def draw_borders(self, frame):
		cv2.rectangle(frame, self.top, self.bottom, self.draw_color, self.line_thickness)
		cv2.circle(frame, self.center, self.radius, self.draw_color, self.line_thickness)

	#####################################

	def get_greyscale_image(self, frame):
		grey_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
		ret, grey_image = cv2.threshold(grey_image, 245, 255, cv2.THRESH_BINARY)

		return grey_image


	def get_shooting_target(self, frame):
		#cv2.normalize(frame, frame, 0, 255, cv2.NORM_MINMAX)
		# normalize the frame
		#cv2.normalize(frame, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)

		self.draw_borders(frame)

		if self.args.debug:
			cv2.imshow("Frame", frame)

		# Crop
		frame = frame[self.top[Y]:self.bottom[Y], self.top[X]:self.bottom[X]]

		return self.get_greyscale_image(frame)


	def get_laser_mark(self, image):
		(laser_mark, _) = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

		return laser_mark


	def get_shot_point(self, laser_mark):
		center = sorted(laser_mark, key = cv2.contourArea, reverse=True)[0]

		(x, y), radius = cv2.minEnclosingCircle(center)

		wh = self.radius * self.area_scale

		shot_x = (float(x) - wh) / self.radius
		shot_y = (float(y) - wh) / self.radius

		dist = math.sqrt(shot_x * shot_x + shot_y * shot_y)

		shot_x = self.target_x + shot_x * self.target_radius
		shot_y = self.target_y + shot_y * self.target_radius

		return (int(shot_x), int(shot_y)), dist


	def update_score(self, shot, dist):
		self.shots_count += 1

		if dist < 1.0:
			self.score += 1 - dist

		if self.shots_count >= self.max_shots:
			self.show_score()
			self.sound_done()
			self.reset()


	def is_timout_expired(self, check_time, timeout_ms):
		end_time = time.time()
		exec_time = (end_time - check_time) * 1000

		if timeout_ms and exec_time < timeout_ms:
			return False

		return True


	def is_shot_allowed(self):
		if not self.last_shot_time:
			return self.last_shot_state != SHOT_STATE_FIRE

		if not self.is_timout_expired(self.last_shot_time, self.shot_timeout_ms):
			return False

		if self.shots_count == 0:
			if not self.is_timout_expired(self.last_shot_time, self.round_timeout_ms):
				return False

		return self.last_shot_state != SHOT_STATE_FIRE


	def check_shot(self, laser_mark):
		if not laser_mark:
			self.last_shot_state = SHOT_STATE_NONE
			return

		if not self.is_shot_allowed():
			return

		self.last_shot_time = time.time()
		self.last_shot_state = SHOT_STATE_FIRE

		self.sound_shot()

		shot, dist = self.get_shot_point(laser_mark)
		log.info(f"Shot {self.shots_count}: {shot}")

		self.draw_shot(shot)

		self.update_score(shot, dist)

		self.skip_next = True


	def capture_frame(self):
		ret,frame = self.camera.read()

		if frame is None:
			cv2.waitKey(100)
			log.warning("Can't read frame")
			return

		img = self.get_shooting_target(frame)
		laser_mark = self.get_laser_mark(img)
		self.check_shot(laser_mark)


	def run(self):
		self.init_camera()
		self.init_target()

		self.reset()

		cv2.namedWindow("Result", 1)
		#cv2.setMouseCallback("Result", mouse_callback, self)

		cv2.imshow("Result", self.target_image)

		while True:
			if cv2.waitKey(1) >= 0:
				break

			try:
				self.capture_frame()
			except KeyboardInterrupt:
				break
			except:
				raise

		self.camera.release()
		cv2.destroyAllWindows()


def main(args):
	s = Shooting(args)

	s.run()

	return 0


if __name__ == '__main__':
	try:
		parser = ArgumentParser()
		parser.add_argument("-d", "--debug", default=False, action="store_true")

		parser.add_argument("-w", "--width",      default=800, type=int)
		parser.add_argument("-g", "--height",     default=640, type=int)
		parser.add_argument("-x", "--centerx",    default=288, type=int)
		parser.add_argument("-y", "--centery",    default=231, type=int)
		parser.add_argument("-r", "--radius",     default=17,  type=int)
		parser.add_argument("-s", "--shots",      default=7,   type=int)
		parser.add_argument("-a", "--area-scale", default=2,   type=float)
		parser.add_argument("-t", "--timeout",    default=0,   type=float, help="Timeout between shots in seconds")
		parser.add_argument("--round_timeout",    default=3,   type=float, help="Timeout after last shot before new round")
		parser.add_argument("-v", "--video",      default=0)

		args = parser.parse_args()

		if args.debug:
			log.setLevel(logging.DEBUG)

		log.info("{}".format(__file__))

		res = main(args)
	except SystemExit:
		raise
	except:
		traceback.print_exc()
		sys.exit(-1)

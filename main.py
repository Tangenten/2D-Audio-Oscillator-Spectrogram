import math
import multiprocessing
import queue
import sys

import librosa
import librosa as _librosa
import numpy
import pyaudio
import pygame
import soundfile
from presets import Preset

def samplesToMono(samples):
	stereoList = [0.0] * len(samples)
	for i in range(0, len(samples) - 1, 1):
		sample = (samples[i] + samples[i + 1] / 2)
		stereoList[i] = sample
		stereoList[i + 1] = sample

	return stereoList


def samplesToMid(samples):
	midsList = [0.0] * len(samples)
	for i in range(0, len(samples) - 1, 1):
		sample = (samples[i] + samples[i + 1]) / math.sqrt(2)
		midsList[i] = sample
		midsList[i + 1] = sample

	return midsList


def samplesToSide(samples):
	sidesList = [0.0] * len(samples)
	for i in range(0, len(samples) - 1, 1):
		sample = (samples[i] - samples[i + 1]) / math.sqrt(2)
		sidesList[i] = sample
		sidesList[i + 1] = sample

	return sidesList


def sampleToYPixel(sample, height):
	return int(((sample + 1) * (height / 2)))


def samplesToAverage(samples):
	sum = 0
	for i in range(0, len(samples), 1):
		sum += samples[i]

	return sum / len(samples)


def samplesToAverage2(samples):
	sum = 0
	for i in range(0, len(samples), 1):
		if samples[i] > 0:
			sum += samples[i]
		else:
			sum += -samples[i]

	return (sum * 2) / len(samples)


def samplesToRootMeanSquare(samples):
	sum = 0
	for i in range(0, len(samples), 1):
		sum += (samples[i] * samples[i])

	mean = sum / len(samples)
	return math.sqrt(mean)


def samplesToMinMax(samples):
	return (min(samples), max(samples))


def samplesToMax(samples):
	return max(samples)


def samplesToMin(samples):
	return min(samples)


def linearInterpolation(y1, y2, frac):
	return (y1 * (1.0 - frac) + y2 * frac)


def cosineInterpolation(y1, y2, frac):
	frac2 = (1.0 - math.cos(frac * 3.14)) / 2
	return (y1 * (1.0 - frac2) + y2 * frac2)


def cubicInterpolation(y0, y1, y2, y3, mu):
	mu2 = mu * mu
	a0 = y3 - y2 - y0 + y1
	a1 = y0 - y1 - a0
	a2 = y2 - y0
	a3 = y1

	return (a0 * mu * mu2 + a1 * mu2 + a2 * mu + a3)


def hermiteInterpolation(y0, y1, y2, y3, mu, tension, bias):
	mu2 = mu * mu
	mu3 = mu2 * mu

	m0 = (y1 - y0) * (1 + bias) * (1 - tension) / 2
	m0 += (y2 - y1) * (1 - bias) * (1 - tension) / 2
	m1 = (y2 - y1) * (1 + bias) * (1 - tension) / 2
	m1 += (y3 - y2) * (1 - bias) * (1 - tension) / 2
	a0 = 2 * mu3 - 3 * mu2 + 1
	a1 = mu3 - 2 * mu2 + mu
	a2 = mu3 - mu2
	a3 = -2 * mu3 + 3 * mu2

	return (a0 * y1 + a1 * m0 + a2 * m1 + a3 * y2);


def logarithmicInterpolation(min, max, current, destination):
	return min * math.pow(max / min, current / (destination - 1))


class audioHandler:

	def __init__(self, pipe):
		self.outputQueue = pipe
		self.renderedQueue = queue.Queue()

		self.pyAudio = pyaudio.PyAudio()
		self.latency = 0
		self.deviceInfo = self.pyAudio.get_default_output_device_info()
		self.bitDepth = 32
		# self.sampleRate = int(self.deviceInfo['defaultSampleRate'])
		self.sampleRate = 44100
		self.channels = 2
		self.frameCount = self.sampleRate // 32
		self.bufferSize = self.frameCount * self.channels
		self.running = False
		self.stream = self.pyAudio.open(format = pyaudio.paFloat32, channels = self.channels, rate = self.sampleRate, frames_per_buffer = self.frameCount, stream_callback = self.audioCallback, output = True)

		self.librosa = Preset(_librosa)
		self.librosa['sr'] = self.sampleRate

		self.latency = self.stream.get_output_latency()

		self.stream.start_stream()

	def audioCallback(self, in_data, frame_count, time_info, status):
		if status == pyaudio.paOutputOverflow or status == pyaudio.paOutputUnderflow:
			print("Underflow / Overflow")

		samples = self.outputQueue.recv()

		self.renderedQueue.put_nowait(samples)

		return (numpy.array(samples, dtype = numpy.float32), pyaudio.paContinue)

	def start(self):
		self.running = True

	def stop(self):
		self.running = False


class graphicHandler:

	def __init__(self, audioHandler):
		pygame.init()
		pygame.display.set_caption("Audio Graphics")

		self.clock = pygame.time.Clock()
		self.width = 1200
		self.height = 600
		self.screen = pygame.display.set_mode((self.width, self.height), pygame.DOUBLEBUF | pygame.HWSURFACE)
		self.running = False
		self.deltaTime = 0

		self.audioHandler = audioHandler
		self.oscilloscope = oscilloscope(self.audioHandler, 600, 600, 0, 0)
		self.spectrogram = spectrogram(self.audioHandler, 600, 600, 600, 0)

	def start(self):
		self.running = True
		self.run()

	def stop(self):
		self.running = False

	def run(self):
		while self.running:
			self.screen.fill((0, 0, 0))

			font = pygame.font.SysFont('Times New Roman', 24)
			textsurface = font.render('FPS: ' + str(self.clock.get_fps())[:3], False, (255, 255, 255))
			self.screen.blit(textsurface, (self.width - 120, self.height - 28))

			self.deltaTime = self.clock.get_time() / 1000

			self.input()

			samples = []
			while not self.audioHandler.renderedQueue.empty():
				samples += self.audioHandler.renderedQueue.get_nowait()

			lines = []
			lines += self.oscilloscope.output(self.screen, self.deltaTime, samples)
			lines += self.spectrogram.output(self.screen, self.deltaTime, samples)

			for line in lines:
				pygame.draw.line(self.screen, (line[0][0], line[0][1], line[0][2]), (line[1][0], line[1][1]), (line[2][0], line[2][1]))

			pygame.display.flip()
			self.clock.tick(60)

	def input(self):
		events = pygame.event.get()
		for event in events:
			if event.type == pygame.QUIT:
				pygame.quit()
				sys.exit()

		self.oscilloscope.input(events)
		self.spectrogram.input(events)


class oscilloscope:

	def __init__(self, audioHandler, width, height, xOffset, yOffset):
		self.samplesToRender = audioHandler.sampleRate
		self.samplesPerPixel = self.samplesToRender / width
		self.pixelsPerSample = 1.0 / (self.samplesToRender / width)

		self.sampleRate = audioHandler.sampleRate
		self.bufferSize = audioHandler.bufferSize
		self.width = width
		self.height = height
		self.xOffset = xOffset
		self.yOffset = yOffset
		self.font = pygame.font.SysFont('Times New Roman', 24)

		self.ringBufferSize = self.sampleRate * 64
		self.ringBuffer = [0.0] * self.ringBufferSize
		self.writePointer = (self.samplesToRender + self.bufferSize) // self.samplesPerPixel
		self.readPointer = 0

		self.mode = 0

	def input(self, events):
		for event in events:
			if event.type == pygame.KEYDOWN:
				if event.key == pygame.K_LEFT:
					if self.samplesToRender > 64:
						self.samplesToRender //= 2
						self.samplesPerPixel = self.samplesToRender / self.width
						self.pixelsPerSample = 1.0 / (self.samplesToRender / self.width)

						self.update()
				if event.key == pygame.K_RIGHT:
					if self.samplesToRender < 100000:
						self.samplesToRender = self.samplesToRender * 2
						self.samplesPerPixel = self.samplesToRender / self.width
						self.pixelsPerSample = 1.0 / (self.samplesToRender / self.width)

						self.update()

				if event.key == pygame.K_x:
					self.mode += 1
					self.mode %= 4

	def update(self):
		if self.samplesToRender <= self.width:
			self.writePointer = (self.samplesToRender + self.bufferSize)
		else:
			self.writePointer = (self.samplesToRender + self.bufferSize) // self.samplesPerPixel

	def output(self, screen, deltaTime, samples):
		textsurface = self.font.render('Sample Range: ' + str(self.samplesToRender), False, (180, 180, 180))
		screen.blit(textsurface, (0 + self.xOffset, 0 + self.yOffset))

		lineList = []

		samplesPerFrame = self.sampleRate * deltaTime

		if self.samplesToRender <= self.width:
			if samples != []:
				# samples = samplesToMono(samples)
				for i in range(0, len(samples), 1):
					self.ringBuffer[int(i + self.readPointer + self.writePointer) % self.ringBufferSize] = samples[i]

			peaks = []
			for i in range(0, int(self.samplesToRender), 1):
				for j in range(0, int(self.pixelsPerSample), 1):
					peaks.append(cosineInterpolation(self.ringBuffer[((i + int(self.readPointer)) % self.ringBufferSize)], self.ringBuffer[((i + 1 + int(self.readPointer)) % self.ringBufferSize)], (j / self.pixelsPerSample)))

			for i in range(0, self.width - 1, 1):
				y1 = sampleToYPixel(peaks[i], self.height)
				y2 = sampleToYPixel(peaks[i + 1], self.height)

				distance = abs((self.height / 2) - y1)
				colorR = numpy.interp(distance, [1, self.height / 2], [255, 0])
				colorG = 180
				colorB = 255

				lineList.append([[colorR, colorG, colorB], [i + self.xOffset, y1 + self.yOffset], [i + 1 + self.xOffset, y2 + self.yOffset]])

			self.readPointer = self.readPointer + ((samplesPerFrame)) * 2

		else:
			if samples != []:
				# samples = samplesToMono(samples)

				offset = 0
				for i in range(0, len(samples), int(self.samplesPerPixel)):
					if self.mode == 0:
						# self.ringBuffer[int(offset + self.readPointer + self.writePointer) % self.ringBufferSize] = samplesToAverage(samples[int(i): int(i) + int(self.samplesPerPixel)])
						self.ringBuffer[int(offset + self.readPointer + self.writePointer) % self.ringBufferSize] = samples[int(i)]

					elif self.mode == 1:
						self.ringBuffer[int(offset + self.readPointer + self.writePointer) % self.ringBufferSize] = samplesToAverage2(samples[int(i): int(i) + int(self.samplesPerPixel)])

					elif self.mode == 2:
						self.ringBuffer[int(offset + self.readPointer + self.writePointer) % self.ringBufferSize] = samplesToRootMeanSquare(samples[int(i): int(i) + int(self.samplesPerPixel)])

					elif self.mode == 3:
						self.ringBuffer[int(offset + self.readPointer + self.writePointer) % self.ringBufferSize] = samplesToMin(samples[int(i): int(i) + int(self.samplesPerPixel)])

					offset += 1

			for i in range(0, self.width - 1, 1):
				y1 = sampleToYPixel(self.ringBuffer[(i + int(self.readPointer)) % self.ringBufferSize], self.height)
				y2 = sampleToYPixel(self.ringBuffer[(i + 1 + int(self.readPointer)) % self.ringBufferSize], self.height)

				distance = abs((self.height / 2) - y1)
				colorR = numpy.interp(distance, [1, self.height / 2], [255, 0])
				colorG = 180
				colorB = 255

				lineList.append([[colorR, colorG, colorB], [i + self.xOffset, y1 + self.yOffset], [i + 1 + self.xOffset, y2 + self.yOffset]])

			self.readPointer = self.readPointer + ((samplesPerFrame // self.samplesPerPixel)) * 2
			return lineList


class spectrogram:

	def __init__(self, audioHandler, width, height, xOffset, yOffset):
		self.sampleRate = audioHandler.sampleRate
		self.bufferSize = audioHandler.bufferSize
		self.width = width
		self.height = height
		self.xOffset = xOffset
		self.yOffset = yOffset
		self.font = pygame.font.SysFont('Times New Roman', 24)

		self.frequencyResolution = 4096
		self.timeResolution = 1024
		self.windowResolution = self.frequencyResolution

		self.frequencyBins = int(self.frequencyResolution / 2)
		self.timeBinsPerSecond = audioHandler.sampleRate / self.timeResolution
		self.timeBinsPerBuffer = audioHandler.bufferSize / self.timeResolution
		self.pixelsPerFrequencyBin = self.width / self.frequencyBins

		self.readPointer = 0
		self.writePointer = self.timeBinsPerBuffer * 2

		self.ringBufferSize = int(self.timeBinsPerBuffer * 4)
		self.ringBuffer = [[0] * self.ringBufferSize for i in range(int(self.frequencyBins))]

		self.pixelList = [self.height] * self.width
		self.pixelFallingRate = int(self.height / (60 / 3))

		self.mode = 0

		self.logScale = [0.0] * self.width
		for i in range(0, self.width, 1):
			self.logScale[i] = logarithmicInterpolation(1, self.frequencyBins, i, self.width) - 1

	def input(self, events):
		for event in events:
			if event.type == pygame.KEYDOWN:
				if event.key == pygame.K_DOWN:
					if self.frequencyResolution > 128:
						self.frequencyResolution //= 2
						self.frequencyBins = int(self.frequencyResolution / 2)
						self.pixelsPerFrequencyBin = self.width / self.frequencyBins

						self.ringBufferSize = int(self.timeBinsPerBuffer * 4)
						self.ringBuffer = [[0] * self.ringBufferSize for i in range(int(self.frequencyBins))]
						self.windowResolution = self.frequencyResolution

						self.logScale = [0.0] * self.width
						for i in range(0, self.width, 1):
							self.logScale[i] = logarithmicInterpolation(1, self.frequencyBins, i, self.width) - 1

				if event.key == pygame.K_UP:
					if self.frequencyResolution < 15000:
						self.frequencyResolution *= 2
						self.frequencyBins = int(self.frequencyResolution / 2)
						self.pixelsPerFrequencyBin = self.width / self.frequencyBins

						self.ringBufferSize = int(self.timeBinsPerBuffer * 4)
						self.ringBuffer = [[0] * self.ringBufferSize for i in range(int(self.frequencyBins))]

						self.windowResolution = self.frequencyResolution

						self.logScale = [0.0] * self.width
						for i in range(0, self.width, 1):
							self.logScale[i] = logarithmicInterpolation(1, self.frequencyBins, i, self.width) - 1
				if event.key == pygame.K_z:
					self.mode += 1
					self.mode %= 2

	def output(self, screen, deltaTime, samples):
		textsurface = self.font.render('Frequency Bins: ' + str(self.frequencyBins), False, (255, 255, 255))
		screen.blit(textsurface, (0 + self.xOffset, 0 + self.yOffset))

		lineList = []

		timeBinsPerFrame = self.timeBinsPerSecond * deltaTime

		if samples != []:
			# samples = samplesToMono(samples)

			stft = numpy.abs(librosa.stft(numpy.asarray(samples), n_fft = self.frequencyResolution, hop_length = self.timeResolution, win_length = self.windowResolution))
			stft_magnitude, stft_phase = librosa.magphase(stft)
			stft_magnitude_db = librosa.amplitude_to_db(stft_magnitude, amin = 1, top_db = 100)

			samples = stft_magnitude_db.tolist()
			for i in range(0, int(self.frequencyBins), 1):
				for j in range(0, int(self.timeBinsPerBuffer), 1):
					self.ringBuffer[i][int(self.writePointer + j) % self.ringBufferSize] = samples[i][j]
			self.writePointer += self.timeBinsPerBuffer

		self.pixelList = [x + self.pixelFallingRate for x in self.pixelList]

		if self.mode == 0:
			for i in range(0, self.width - 2, 1):
				frac, whole = math.modf(self.logScale[i])

				y0 = self.ringBuffer[int(whole - 1)][int(self.readPointer) % self.ringBufferSize]
				y1 = self.ringBuffer[int(whole)][int(self.readPointer) % self.ringBufferSize]
				y2 = self.ringBuffer[int(whole + 1)][int(self.readPointer) % self.ringBufferSize]
				y3 = self.ringBuffer[int(whole + 2)][int(self.readPointer) % self.ringBufferSize]

				y1 = hermiteInterpolation(y0, y1, y2, y3, frac, 1.5, 0)

				y1 = numpy.interp(y1, [1, 70], [self.height, 0])

				distance = abs((self.height / 4) - y1)
				colorR = numpy.interp(distance, [1, self.height / 1], [0, 255])
				colorG = numpy.interp(distance, [1, self.height / 1], [162, 255])
				colorB = numpy.interp(distance, [1, self.height / 1], [255, 255])

				if y1 < self.pixelList[i]:
					self.pixelList[i] = y1
					lineList.append([[colorR, colorG, colorB], [i + self.xOffset, y1 + self.yOffset], [i + self.xOffset, self.height + self.yOffset]])
				else:
					lineList.append([[colorR, colorG, colorB], [i + self.xOffset, self.pixelList[i] + self.yOffset], [i + self.xOffset, self.height + self.yOffset]])
		elif self.mode == 1:
			for i in range(0, self.width, 1):
				value = self.ringBuffer[int(self.logScale[i])][int(self.readPointer) % self.ringBufferSize]

				y1 = numpy.interp(value, [1, 70], [self.height, 0])

				distance = abs((self.height / 8) - y1)
				colorR = numpy.interp(distance, [1, self.height / 1], [0, 255])
				colorG = numpy.interp(distance, [1, self.height / 1], [162, 255])
				colorB = numpy.interp(distance, [1, self.height / 1], [255, 255])

				if y1 < self.pixelList[i]:
					self.pixelList[i] = y1
					lineList.append([[colorR, colorG, colorB], [i + self.xOffset, y1 + self.yOffset], [i + self.xOffset, self.height + self.yOffset]])
				else:
					lineList.append([[colorR, colorG, colorB], [i + self.xOffset, self.pixelList[i] + self.yOffset], [i + self.xOffset, self.height + self.yOffset]])
		self.readPointer += timeBinsPerFrame * 2
		return lineList


def run(pipe, bufferSize):
	data, samplerate = soundfile.read('C:\\Users\\Tangent\\Dropbox\\Programming\\PycharmProjects\\Oscillator 2D Spectrogram\\02 - Carefree.flac')

	dataLength = len(data)
	samples = [0.0] * bufferSize
	offset = 0

	while True:
		for i in range(0, bufferSize, 2):
			samples[i] = data[offset][0]
			samples[i + 1] = data[offset][1]

			offset += 1

			if offset >= dataLength:
				offset = 0

		pipe.send(samples)  # Blocking if size == 1


if __name__ == '__main__':
	child_conn, parent_conn = multiprocessing.Pipe(duplex = False)

	a = audioHandler(child_conn)
	g = graphicHandler(a)

	a.start()
	audioThread = multiprocessing.Process(target = run, args = (parent_conn, a.bufferSize))
	audioThread.daemon = True
	audioThread.start()
	g.start()

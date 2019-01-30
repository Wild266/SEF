import librosa
print("Loaded librosa")

import numpy as np
print("Loaded numpy")

from math import log10, log2

def product(l):
	p = 1
	for x in l:
		if x > 0 and x < 1e4:
			p *= x
	return p

def calcDB(amplitude):
	return max(-100, 20 * log10(abs(amplitude) + 0.0001))

SAMPLE_CHUNK = int(0.01 * 16000)
I_MIN = -100
I_MAX = 100
I_COUNT = 100
I_SIZE = (I_MAX - I_MIN) / I_COUNT

def weightedSum(p1):
	p1 = [x * I_SIZE * p1[x] for x in range(len(p1))]
	return sum(p1)

def calcAmps(filename):
	amps, sr = librosa.load(filename, sr=16000) # y is a numpy array of the wav file, sr = sample rate
	print("Loaded", filename)
	# amps = height of the wave
	# how much the sound changes
	dbDifs = [(calcDB(amps[x]) - calcDB(amps[x - 1])) for x in range(1, len(amps))]
	#dbDifs = [abs(amps[x] - amps[x - 1]) for x in range(len(amps) - 1)]
	dbDifChunk = []
	for x in range(len(dbDifs)):
		mx = 0
		for dif in dbDifs[x * SAMPLE_CHUNK : x * SAMPLE_CHUNK + SAMPLE_CHUNK]:
			if abs(dif) > abs(mx):
				mx = dif # preserves negativity
		dbDifChunk.append(mx)
	dbInt = [0 for _ in range(I_COUNT + 1)]
	#dbInt = [[] for _ in range(101)] # range 0 .. 0.1, step = 0.001
	# db intervals
	for db in dbDifChunk:
		iv = max(0, min(I_COUNT, int((db - I_MIN) / I_SIZE))) # interval
		dbInt[iv] += 1#.append(db) # keep track of how many are in this range
	return amps, dbDifs, dbInt
	
def display(dbDifs, dbInt, removeProp = False):
	for x in range(len(dbInt)):
		print("INTERVAL ", I_MIN + x * I_SIZE, " -> ", I_MIN + (x + 1) * I_SIZE, ": ", dbInt[x], (dbInt[x] / (len(dbDif) // SAMPLE_CHUNK)) if not removeProp else "")
	print("MAX/MIN Articulation: ", max(dbDifs), min(dbDifs))
	print("AVERAGE Articulation: ", (sum(dbDifs) / len(dbDifs)))

# each list contains totals for the number of amplitude differences within each interval
# we should make it proportional
# dbInt = proportion of values
# convert list to proportions
def prop(l, size):
	return [_ / size for _ in l]

def frequencies(y, sr = 16000, frame = 0, low = 100, high = 1000):
	map = {}
	pitch, mag = librosa.piptrack(y = y[160 * frame : 160 * frame + 160], sr = sr, threshold = 0.05, fmin = low, fmax = high)
	for x in range(len(pitch)):
		if pitch[x] > 0:
			map[pitch[x][0]] = mag[x][0]
	return map

def maxFrequency(frequencies):
	maxMagn = 0
	maxPitch = 0
	for x in frequencies:
		if frequencies[x] > maxMagn:
			maxMagn = frequencies[x]
			maxPitch = x
	return maxPitch, maxMagn

def maxFrequencies(y, sr = 16000, low = 100, high = 1000):
	pitch, mag = librosa.piptrack(y = y, sr = sr, fmin = low, fmax = high, threshold = 0.1)
	timeArray = [(0, 1.5) for x in range(len(pitch[0]))]
	for time in range(len(pitch[0])): # loop through frames
		for p in range(len(pitch)): # loop through pitches at time
			if pitch[p][time] > 0: # check pitch at point
				if mag[p][time] > timeArray[time][1]: # check if volume is higher than others
					timeArray[time] = (pitch[p][time], mag[p][time])
	return timeArray

# first two are raw data
# next are sum, product, std, mean
def getStats(file1):
	y1, dbDif, dbInt = calcAmps(file1)

	mf1 = maxFrequencies(y1, sr = 16000)
	filtered_mf1 = []
	for x in range(len(mf1)):
		if mf1[x][0] > 0:
			filtered_mf1.append(mf1[x][0])
	p1 = prop(dbInt, len(dbDif) // SAMPLE_CHUNK)	# proportions of intervals in File1
	# higher jumps should be weighted differently!
	# [Decibels above silence] * Number in interval -> New Number in Interval
	# x * I_SIZE = number of decibels above silence
	# silence = -100db
	p1 = [x * I_SIZE * p1[x] for x in range(len(p1))]
	return sum(p1), product(p1), np.std(filtered_mf1), np.mean(filtered_mf1)

def displayStats(file1, sm, prod, std, mean):
	print("ARTICULATION [DECIBEL DIFFERENCES]")
	print("\tWeighted sum:", sm)
	print("\tWeighted product:", prod)
	print("PITCHES")
	print("\tFile", file1)
	print("\t\tStd. Dev:", std)
	print("\t\tMean:", mean)
	
file1 = input("File: ")

data = getStats(file1)
print("Raw stats =", data)
displayStats(file1, *data)
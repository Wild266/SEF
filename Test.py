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
I_MIN = 0
I_MAX = 100
I_COUNT = 100
I_SIZE = (I_MAX - I_MIN) / I_COUNT

def weightedSum(p1):
	p1 = [x * I_SIZE * p1[x] for x in range(len(p1))]
	return sum(p1)

def calcAmps(filename):
	amps, sr = librosa.load(filename, sr=16000) # y is a numpy array of the wav file, sr = sample rate
	print("Loaded file")
	# amps = height of the wave
	print(calcDB(max(amps)))
	print(calcDB(min(amps)))
	# how much the sound changes
	dbDifs = [abs(calcDB(amps[x]) - calcDB(amps[x - 1])) for x in range(1, len(amps))]
	#dbDifs = [abs(amps[x] - amps[x - 1]) for x in range(len(amps) - 1)]
	dbDifChunk = [
	max(dbDifs[x * SAMPLE_CHUNK : x * SAMPLE_CHUNK + SAMPLE_CHUNK])
	for x in range(len(dbDifs) // 160)]
	dbInt = [0 for _ in range(I_COUNT + 1)]
	#dbInt = [[] for _ in range(101)] # range 0 .. 0.1, step = 0.001
	# db intervals
	for db in dbDifChunk:
		iv = min(I_COUNT, int((db - I_MIN) / I_SIZE)) # interval
		dbInt[iv] += 1#.append(db) # keep track of how many are in this range
	return dbDifs, dbInt
	
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
	
file1 = input("File 1: ")
file2 = input("File 2: ")
dbDif, dbInt = calcAmps(file1)
dbDif2, dbInt2 = calcAmps(file2)
p1 = prop(dbInt, len(dbDif) // SAMPLE_CHUNK)	# proportions of intervals in File1
p2 = prop(dbInt2, len(dbDif2) // SAMPLE_CHUNK)	# proportions of intervals in File2

# higher jumps should be weighted differently!
# [Decibels above silence] * Number in interval -> New Number in Interval
# x * I_SIZE = number of decibels above silence
# silence = -100db
p1 = [x * I_SIZE * p1[x] for x in range(len(p1))]
p2 = [x * I_SIZE * p2[x] for x in range(len(p2))]

print(file1)
#display(dbDif, dbInt)
#display(dbDif, p1, removeProp = True)
print(file2)
#display(dbDif2, dbInt2)
#display(dbDif, p2, removeProp = True)

fileSubtract = [p1[_] - p2[_] for _ in range(len(dbInt))]
fileDivide = [p1[_] / (p2[_] + 1e-99) for _ in range(len(dbInt))]
fileDivideProduct = product(fileDivide)

print("Differences [file1 - file2]: ")
display(dbDif, fileSubtract, removeProp = True)

print("Differences [file1 / file2]: ")
display(dbDif, fileDivide, removeProp = True)

print("Weighted sum file1:", sum(p1))
print("Weighted sum file2:", sum(p2))
print("Weighted product file1:", product(p1))
print("Weighted product file2:", product(p2))
print("Sum of file1 - file2: ", sum(fileSubtract))
print("Product of file1 / file2: ", fileDivideProduct)
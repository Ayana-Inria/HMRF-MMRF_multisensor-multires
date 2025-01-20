import numpy as np
import struct

'''
How pixel are stored: pixel(b, h, w)

	b num bands (4)
	h num height (400)
	w num width (800)

	------------------------------------
	|   first band                      |
	|   [0,0,0][0,0,1]  .  .  [0,0,w]   | 
	|   [0,1,0]                 .       |
	|   .          .            .       | 
	|   .               .               |
	|   [0,h,0]   .     .     [0,h,w]   |
	------------------------------------

How file represent pixels: bsq
	band per band separately, then images follows raster scan

Map images are standardized with values:
	0: as background
	1,...,N: label of class

'''
class Image:
	def __init__(self, width, height, bands = 1, depth = 255, pixelType = 'float32'):
		self.width = width
		self.height = height
		self.bands = bands
		self.depth = depth

		pixel = np.zeros((self.bands,self.height, self.width), dtype = pixelType)
		
		self.pixel = pixel
		#self.setType(pixelType)

	def getWidth(self):
		return self.width

	def getHeight(self):
		return self.height

	def getBands(self):
		return self.bands

	def setType(self, typeWanted):
		#TO DO: insert checking
		self.pixel=self.pixel.astype(typeWanted) #type conversion

	def zeros(self):
		self.pixel = np.zeros((self.bands,self.height, self.width))

	def invertGraylevel(self):
		print('Inverting graylevels')
		#do like p[...] = self.depth - p
		self.pixel *= -1
		self.pixel += self.depth
			
	#Load from file in image numByte byte for graylevel, mode:bsq
	def load(self, filePath, numByte):
		print('loading image from -> "' , filePath, '"')
		with open(filePath, "rb") as inputFile:
			val = inputFile.read(numByte) #read first value
			for p in np.nditer(self.pixel, op_flags=['readwrite']): # loop every element of array in readwrite mode
				valInt = int.from_bytes(val, byteorder='little') #from hex string to int, little-endian
				p[...] = valInt
				val = inputFile.read(numByte) #read next byte from input
	
	#Store graylevel in a file, numByte byte for graylevel, mode:bsq
	def store(self, filePath, numByte):
		print('storing image in -> "' , filePath, '"')
		if numByte == 1:
			with open(filePath, 'wb') as outputFile: #open as write string
				for p in np.nditer(self.pixel): # loop every element of array in read only mode
					outputFile.write(int(p).to_bytes(1, byteorder='little')) #write value
		elif numByte == 2 or numByte == 4 or numByte == 8: #expect that pixel are float (16, 32, 64bit)
			if numByte == 2:
				fmt = 'h'
			if numByte == 4:
				fmt = 'f'
			if numByte == 8:
				fmt = 'd'
			with open(filePath, 'wb') as outputFile: #open as write string
				for p in np.nditer(self.pixel): # loop every element of array in read only mode
					bArray = bytearray(struct.pack(fmt, p)) #gives array of fmt byte
					for bArrayElement in bArray:
						outputFile.write(bArrayElement.to_bytes(1, byteorder='little')) #write value
		else:
			print('image.store(), unsupported option')
			
		'''
		if numByte == 8: #expect that pixel are float (64bit)
			with open(filePath, 'wb') as outputFile: #open as write string
				for p in np.nditer(self.pixel): # loop every element of array in read only mode
					bArray = bytearray(struct.pack("d", p)) #gives array of 8 byte
					for bArrayElement in bArray:
						outputFile.write(bArrayElement.to_bytes(1, byteorder='little')) #write value
		elif numByte == 4: #expect that pixel are float (32bit)
			with open(filePath, 'wb') as outputFile: #open as write string
				for p in np.nditer(self.pixel): # loop every element of array in read only mode
					bArray = bytearray(struct.pack("f", p)) #gives array of 4 byte
					for bArrayElement in bArray:
						outputFile.write(bArrayElement.to_bytes(1, byteorder='little')) #write value
		elif numByte == 2: #expect that pixel are float (16bit)
			with open(filePath, 'wb') as outputFile: #open as write string
				for p in np.nditer(self.pixel): # loop every element of array in read only mode
					bArray = bytearray(struct.pack("h", p)) #gives array of 4 byte
					for bArrayElement in bArray:
						outputFile.write(bArrayElement.to_bytes(1, byteorder='little')) #write value
		'''

	#Map all the labels with 0,1,...,n
	#n+1 -> different values in map image (number of classes)
	def stdMaker(self):
		unique, counts = np.unique(self.pixel, return_counts=True) #get unique elements

		#print('unique:')
		#print(unique)

		#map every pixel
		for p in np.nditer(self.pixel, op_flags=['readwrite']): #iter every element in writ mode
			p[...] = sorted(unique).index(p) #map to new values

		#print('self.height ', self.height)
		#print('self.width ', self.width)

	''' DUMMY WORKING stdMaker
	def stdMaker(self):
		unique, counts = np.unique(self.pixel, return_counts=True) #get unique elements
		#map every pixel
		for p in np.nditer(self.pixel, op_flags=['readwrite']): #iter every element in writ mode
			if(sorted(unique).index(p) > 2):
				p[...] = 0
			else:
				p[...] = sorted(unique).index(p) #map to new values
	'''

	#Equilibrate all the classes to have similar numbers of occurrencies in dataset.
	#all the classes will have no more than the maxFraction times number of samples 
	#with respect to the weakest class.
	#the samples lost will be discarted in raster order
	def equilibrateClasses(self, maxFraction):
		unique, counts = np.unique(self.pixel, return_counts=True) #get unique elements

		#remove unlabeled samples
		unique = np.delete(unique, 0)
		counts = np.delete(counts, 0)

		C = unique.shape[0]

		#identify class with less samples
		lessClass = np.argmin(counts)

		for c in range(C):
			if counts[c] > maxFraction*counts[lessClass]:
				#reduce sample vector for class c
				pixToBeDiscarded = counts[c] - maxFraction*counts[lessClass]

				for h in range(0, self.height):
					for w in range(0, self.width):
						if self.pixel[0][h][w] == c+1:
							if pixToBeDiscarded > 0:
								self.pixel[0][h][w] = 0
								pixToBeDiscarded -= 1

	#delete bands indexed by bandsToBeCutted list
	def cutBand(self, bandsToBeCutted):
		if all(cutIndex > -1 and cutIndex < self.bands for cutIndex in bandsToBeCutted):
			self.pixel = np.delete(self.pixel, bandsToBeCutted, axis=0)
			self.bands -= len(bandsToBeCutted)
		else:
			print('bad argument for cutBand function')
		
	def getNumOfClasses(self):
		if self.bands > 1:
			print('Image.getNumOfClasses(), getNumOfClasses undefined on multichannel images.')
		return int(self.pixel.max()) #return max value o self.pixel
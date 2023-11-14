import numpy as np
import cv2
import tifffile
import os
from argparse import ArgumentParser
import matplotlib.pyplot as plt
from matplotlib import colors

refArray = [[]]

def getInputs():
        # sets input variables taken from (i) terminal command line or (ii) 'default's

        parser = ArgumentParser()
        # parser.add_argument("--input_path", type=str, dest="input", help="Path to input directory", default='/Users/aholub/Documents/LLSvsNFINDR_04252022/synthetic-datasets/')
        # parser.add_argument("--output_path", type=str, dest="output", help="Path to output directory", default='/Users/aholub/Downloads/LSS-norm-testing')
        # parser.add_argument("--reference_wells", nargs="+", type=str, dest="references", help="Wells wih (1) bioreport/dye", default=['A1', 'B1', 'C1', 'D1'])
        # parser.add_argument("--data_wells", nargs="+", type=str, dest="data", help="Wells with multiple/all bioreports/dyes", default=['E1'])
        # parser.add_argument("--blank_well", dest="blank", type=str, help="Well with (0) bioreports/dyes", default='H12')
        parser.add_argument("--input_path", type=str, dest="input", help="Path to input directory", default='/Users/aholub/Downloads/AH_CP-390-650-5-refs-and-data_06082022-copy')
        parser.add_argument("--output_path", type=str, dest="output", help="Path to output directory",default='/Users/aholub/Downloads/getDECONrunnning')
        parser.add_argument("--reference_wells", nargs="+", type=str, dest="references",help="Wells wih (1) bioreport/dye", default=['B1', 'C1', 'D1', 'E1'])
        parser.add_argument("--data_wells", nargs="+", type=str, dest="data", help="Wells with multiple/all bioreports/dyes", default=['F1'])
        parser.add_argument("--blank_well", dest="blank", type=str, help="Well with (0) bioreports/dyes", default='Z1')

        args = parser.parse_args()

        print("\ninput file path:", args.input)
        print("\noutput file path:", args.output)
        print("\nreference wells:", args.references)
        print("\ndata wells:", args.data)
        print("\nblank well:", args.blank)

        return args

def createLists(data):
        FL = []
        for well in data:
                for file in os.listdir('.'):
                        if well in file and file.endswith(".tif"):
                                FL.append(file)
        FL.sort()

        # Create lists of dataset timepoints and number of tiles using tokens from the file names
        timepointsList = []
        tilelocationList = []
        spectrum = []
        # token example: PID########_ExperimentName_Timepoints_0-0_Tile_Transmission-Automated-Wavelength#_0.0_0_1
        for file in FL:
                tokens = str(file).split('_')
                wavelength = str(tokens[6]).split('-')
                if tokens[2] not in timepointsList:
                        timepointsList.append(tokens[2])
                if tokens[5] not in tilelocationList:
                        tilelocationList.append(tokens[5])
                if wavelength not in spectrum:
                        spectrum.append(wavelength)
                # if wavelength[2] not in spectrum:
                #         spectrum.append(wavelength[2])
        timepointsList.sort()
        tilelocationList.sort()
        spectrum.sort()
        spectrum = list(np.concatenate(spectrum).flat) #flatten list, 2D > 1D ex. [[1],[2],[3],...] > [1,2,3,...]
        return timepointsList, tilelocationList, spectrum, FL

def transferFunction(blank):
        # DARK: image stack of calibration well, no illumination - exposure = 0 ms @ 650 nm (=wavelength[max])
        # BRIGHT: image stack of calibration well, illumination > 50% dynamic range @ 650 nm (=wavelength[max])
        # LAMP: image stack of unobstructed light path, illumination = BRIGHT
        # CC (correction coeff.): inverse of the transfer function (TF) calculated from DARK/BRIGHT/LAMP calibration images

        darkMeans, brightMeans, lampMeans = [], [], []
        meanStacks = {'DARK': darkMeans, 'BRIGHT': brightMeans, 'LAMP': lampMeans}

        for stack in meanStacks:
                tiles = [i for i in os.listdir() if (blank in i and stack in i)]
                images = createTransferStack(tiles)
                meanStacks[stack] = np.mean(images, axis=(1, 2), dtype=float)  # average in XY per tile

        TF = meanStacks['BRIGHT'] - meanStacks['DARK'] / meanStacks['LAMP']
        CC = 1.0 / TF

        print("correction coefficient CC is", CC)
        print("darkMeans is", meanStacks['DARK'])
        return CC, meanStacks

def createTransferStack(tiles):
        # translates filename stacks to pixel-value arrays
        row, col = tifffile.TiffFile(tiles[0]).asarray().shape
        images = np.zeros((len(tiles), row, col))
        tiles.sort()

        for i in tiles:
                print("Processing file: ", i)
                image = tifffile.TiffFile(i).asarray()
                images[tiles.index(i),:,:] = np.array(image,dtype=float)

        return images

def normalize_refs(references, spectrum):

        FL = os.listdir(".")
        FL.sort()

        refMatrix = np.zeros((len(references), len(spectrum)))

        for ref in references:
                stackArray = np.zeros((53, 4194304))
                for wavelength in spectrum:
                        indexREF = references.index(ref)
                        indexWAVE = spectrum.index(wavelength)

                        files = [file for file in FL if ('_' + ref + '_' in file and wavelength + '_' in file)]

                        image = tifffile.TiffFile(files[0]).asarray()
                        stackArray[indexWAVE,:] = np.ndarray.flatten(image).T

                # Reshape into a # pixels X # channels matrix
                mu = np.mean(stackArray, axis=0)
                sd = np.std(stackArray,  axis=0)
                normalized = (stackArray - mu) / sd
                normalized = np.clip(normalized, 0, None)
                normalized = np.nan_to_num(normalized, nan=0)
                normalized = np.mean(normalized, axis=1)
                refMatrix[indexREF] = normalized

        return refMatrix

def createRefMatrix(references, spectrum):
        # Create an array of the reference spectrum (mean intensity value per wavelength per pixel) per image for deconvolution
        refMatrix = np.zeros((len(references), len(spectrum)))
        FL = os.listdir(".")
        FL.sort()

        for ref in references:
                for wavelength in spectrum:
                        indexWAVE = spectrum.index(wavelength)
                        indexREF = references.index(ref)
                        tiles = [file for file in FL if ('_' + ref + '_' in file and wavelength + '_' in file)]
                        waveArray = []
                        for file in tiles:
                                image = tifffile.TiffFile(file).asarray()
                                waveArray = np.append(waveArray, image)

                        # waveArray = waveArray - 104.7
                        # for i in waveArray:
                        #         if i <= 104.7:
                        #                 i = 0
                        # waveArray = np.ptp(waveArray)

                        waveMean = np.mean(waveArray)

                        # norm = np.linalg.norm(waveArray)
                        # waveArray_NORM = waveArray/norm
                        # waveMean_NORM = np.mean(waveArray_NORM)

                        # wave_NORM = normalize(waveArray)
                        # wave_NORM_MEAN = np.mean(wave_NORM)

                        refMatrix[indexREF, indexWAVE] = waveMean

        # refMatrix = normalize(refMatrix)

        return refMatrix

def createDataStack(tileStackList):
        # Creates a stack of images (= # of wavelengths images) per data well tile
        # (np.mean(tempTif))*1.25)) will only include values in the array that are siginificantly different from the background, being if the intensity value is larger than the mean pixel intensity * 1.25.
        tileStackList.sort()
        print("tileStackList is", tileStackList)
        tempTif = tifffile.TiffFile(tileStackList[0]).asarray()
        row,col = tempTif.shape
        imgStack = np.zeros((len(tileStackList),row,col))
        wavelength = 0
        for i in tileStackList:
            print("Processing file: ", i)
            tempTif = tifffile.TiffFile(i).asarray()
            tempTif -= np.mean(tempTif, dtype=np.uint16)
            imgStack[tileStackList.index(i),:,:] = np.array(tempTif*(tempTif > (np.mean(tempTif))*1)) #1.5 OG
            wavelength+=1
        print("imgstack is", imgStack)
        return imgStack

def processLLS(CC, meanStacks, refMatrix, imgStack):
        # coefficientOutput = np.abs(np.linalg.inv(hypercube.T.dot(hypercube)).dot(hypercube.T).dot(X))  #Ordinary least squares fit from wikipedia
        # y = X^T * B + eps  # linear regression, where * is matrix mult, reduces to inneer product
        # pixel = (mean channel)^T * B + eps, y needs to be same size as beta
        # beta = X^T * X * X^T * y, where * is matrix multiplication
        # np.max(X, axis=0))) = y intercept
        # X.T.dot(X)
        # X^2 * X * Y * b0

        coefficientOutput = []
        hypercube = imgStack.T.reshape(imgStack.shape[1]*imgStack.shape[2], imgStack.shape[0])
        dimZ,dimY,dimX = imgStack.shape
        progress=0
        for pixel in hypercube:
                Y = np.array(pixel)
                # Y = (pixel - meanStacks['DARK']) * CC  # transfer function based correction / this was deleted a one point and that version was copied. placing back, needs work
                X = np.array(refMatrix).T
                coefficientOutput.append(np.sqrt(np.linalg.inv(X.T.dot(X)).dot(X.T).dot(Y) * np.max(X, axis=0)))
                if progress%10000==0:
                        print(progress)
                progress+=1
        # Reshape the coefficient array to create images of each channel image, ouput an array of which columns are channels
        chan = np.array(coefficientOutput).shape[1]
        image = np.array(coefficientOutput, dtype = np.uint16).T.reshape(chan,dimY,dimX)
        return image

def preview(fullstack, spectrum, references, data):
        fig, ax = plt.subplots(figsize = (15,5))
        ax.imshow(fullstack)

        ax.grid(which='minor', axis='both', linestyle='-', color='k', linewidth=2)
        ax.set_xticks(np.arange(0, len(spectrum), 1))
        ax.set_xticklabels(spectrum)
        ax.set_yticks(np.arange(0, len(references + data), 1))
        ax.set_yticklabels(references + data)
        plt.show()

def saveFile(output, image):
        # Save the data as a tif file in the output path
        tifffile.imsave(output, image)

def Run(input, output, references, data, blank):
        # move to the input data directory
        os.chdir("/Users/aholub/Downloads/AH_CP-390-650-5-refs-and-data_06082022-copy")
        # create a file list with all files from the data wells, including all wells and tiles
        timepoints, tiles, spectrum, FL = createLists(data)
        # create correction coefficent for the lightsource using blank (calibration) well
        CC, meanStacks = transferFunction(blank)
        # create a stack for the reference wells
        # refMatrix = createRefMatrix(references, spectrum)

        refMatrix = normalize_refs(references, spectrum)

        # create lists of corresponding timepoints/wells/tiles, and pass the list to create image stack and process via LinearLeastSquare
        for timepoint in timepoints:
                for well in data:
                        for tile in tiles:
                                tileStackList = []
                                for file in FL:
                                        tokens = str(file).split('_')
                                        if timepoint == tokens[2] and well == tokens[4] and tile == tokens[5]:
                                               tileStackList.append(file)
                                tileStackList.sort()
                                if len(tileStackList) != 0:
                                        imgStack = createDataStack(tileStackList)
                                        avgStack = np.mean(imgStack, axis=(1,2))
                                        avgStack = np.expand_dims(avgStack, axis = 0)
                                        fullstack = np.append(refMatrix, avgStack, axis = 0)
                                        preview(fullstack, spectrum, references, data)
                                        image = processLLS(CC, meanStacks, refMatrix, imgStack)
                                        # image = np.invert(image)
                                        output = output + '/' + well + '_' + tile + '_' + timepoint + '.tif'
                                        saveFile(output, image)
                                        print("output file name is:", output)

if __name__ == '__main__':
        args = getInputs()
        Run(**vars(args))
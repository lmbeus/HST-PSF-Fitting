import numpy as np
import sys
import math
import json
from astropy.io import fits
from photutils import centroid_2dg
from photutils import CircularAperture
from photutils import aperture_photometry as phot
from astropy.io import fits
import sys, os, getopt

#To call the program on the command line: python3 modelfilename imagefilename
#Load in camera data and PSF model and store it into numpy array
cameras = json.loads(open('Cameras.json').read())

infile = sys.argv[1]
psf_model = fits.open(infile)
if len(psf_model) != 1:
	frame = 'SCI'
else:
	frame = 0
	
psf_model_data = np.array(psf_model[frame].data, dtype = '<f8')
psf_model.close()

#Load in the HST image into a numpy array, and store the header info
infile = sys.argv[2]
psf_image = fits.open(infile)
if len(psf_image) != 1:
	frame = 'SCI'
else:
	frame = 0
	
psf_image_data = np.array(psf_image[frame].data, dtype = '<f8')
image_header = psf_image[0].header
psf_image.close()

telescope = image_header['TELESCOP']
instrument = image_header['INSTRUME'] + str(image_header['CAMERA'])
filtername = image_header['FILTER']
header_array = [telescope, instrument, filtername]

#Create the output file
#Initilaized with the file rootname, filter, and date of observsation
out_file = image_header['ROOTNAME'] + '_output.txt'
out = open(out_file, 'w')
out.write(image_header['ROOTNAME'] + ' - ' + filtername + '\n')
out.write('UT ' + image_header['DATE-OBS'] + ' ' + image_header['TIME-OBS'] + '\n\n')

#Prompt for x and ycoordinates, store coordinates as tuple 'center'
print('Please enter the coordinates of the center of the object')
center = input('in the format "XXX YYY" with a space between each number\n')

#Prepare the image data for calculations
bkgd = cameras[instrument]['Background']
center = [int(center.split()[1]), int(center.split()[0])]
image_psf = psf_image_data[center[0]-3:center[0]+2, center[1]-3:center[1]+2]
image_psf = image_psf - bkgd

#save the two psf arrays to files to be read into C++ 
#np.savetxt('psf_model_data.csv', psf_model_data, delimiter = ',') 
np.save('psf_model_data.npy', psf_model_data) 
np.save('image_psf.npy', image_psf) 
#np.savetxt('image_psf.csv', psf_image_data, delimiter = ',')

#prepare additional parameters for binary fit calculations
coords = []
coords.append((center[0] - 1, center[1] - 1))
coords.append((center[0] - 1, center[1] + 0))
coords.append((center[0] - 1, center[1] + 1))
coords.append((center[0] + 0, center[1] - 1))
coords.append((center[0] + 0, center[1] + 0))
coords.append((center[0] + 0, center[1] + 1))
coords.append((center[0] + 1, center[1] - 1))
coords.append((center[0] + 1, center[1] + 0))
coords.append((center[0] + 1, center[1] + 1))
temp = []
for secondary in coords:
	if not any(i < 0 for i in secondary):
		if secondary[0] < psf_image_data.shape[0]:
	        	if secondary[1] < psf_image_data.shape[1]:
	        		temp.append(secondary)
	        
coords = np.asarray(temp, dtype = '<f8')
center_np = np.asarray(center, dtype = '<f8')
#save coords and center to csv files to be used in binary fitting in C++
np.save('coords.npy', coords)
np.save('center.npy', center_np)

#Run the binning program
os.system('./PSF_prep')

#Run the single fit Program
os.system('./Single_Fitting')

#Load the single fit results
iteration_array = np.load('iteration_array.npy')
PSF_number_array = np.load('PSF_number_array.npy')
flux_array = np.load('flux_array.npy')
minchi_value_array = np.load('minchi_value_array.npy')
residual_error_array = np.load('residual_error_array.npy')

#Write the results to the outfile
out.write('Single Fit:\n')
for i in range(len(iteration_array)):
	iteration = str(iteration_array[i])
	PSF = str(PSF_number_array[i])
	flux = str(flux_array[i])
	minchi = str(minchi_value_array[i])
	residual_error = str(residual_error_array[i])
	
	out.write('Iteration: '+ iteration + ' PSF: '+ PSF + ' Flux: '+ flux + ' Min Chi: ' + minchi + ' Residual Error: ' + residual_error + '\n')

#Run the binary fit program
print('- - -')
os.system('./Binary_Fitting')

residual_error_array = np.load('residual_error_array.npy')
center_array = np.load('center_array.npy')
secondary_array = np.load('secondary_array.npy')
best_primary_array = np.load('best_primary_array.npy')
flux_primary_array = np.load('flux_primary_array.npy')
best_secondary_array = np.load('best_secondary_array.npy')
flux_secondary_array = np.load('flux_secondary_array.npy')
flux_sum_array = np.load('flux_sum_array.npy')

outputs = []
for i in range(len(residual_error_array)):
	residual_error = residual_error_array[i]
	center = center_array[i]
	secondary = secondary_array[i]
	best_primary = best_primary_array[i]
	flux_primary = flux_primary_array[i]
	best_secondary = best_secondary_array[i]
	flux_secondary = flux_secondary_array[i]
	flux_sum = flux_sum_array[i]
	
	outputs.append((residual_error, center, secondary, best_primary, flux_primary, best_secondary, flux_secondary, flux_sum))
	
PlateScaleX = cameras[instrument]['PlateScaleX']
PlateScaleY = cameras[instrument]['PlateScaleY']
	
print('- - -\nBegin calculation of angles and separation')	
orient=image_header['orientat']%360

array_of_PSFs = [];
for i in range(100):
	filename = " PSFmodel_" + str(i + 1) + ".npy"
	array_of_PSFs.append(np.load(filename))
array_of_PSFs = np.asarray(array_of_PSFs)    	

bestprim=[]
bestsec=[]
tempout=[]
cx=int((array_of_PSFs[0].shape[1]-1)/2)
cy=int((array_of_PSFs[0].shape[0]-1)/2)
for itr,output in enumerate(outputs):
    
    xp=output[1][1]+centroid_2dg(array_of_PSFs[output[3]])[0]-cx
    yp=output[1][0]+centroid_2dg(array_of_PSFs[output[3]])[1]-cy
    xs=output[2][1]+centroid_2dg(array_of_PSFs[output[5]])[0]-cx
    ys=output[2][0]+centroid_2dg(array_of_PSFs[output[5]])[1]-cy

    x1=cx+centroid_2dg(array_of_PSFs[output[3]])[0]-cx
    y1=cy+centroid_2dg(array_of_PSFs[output[3]])[1]-cy
    x2=cx+centroid_2dg(array_of_PSFs[output[5]])[0]-cx
    y2=cy+centroid_2dg(array_of_PSFs[output[5]])[1]-cy

    angle=(orient-90+math.atan2(ys-yp,xs-xp)*180/math.pi)%180
    if (ys-yp,xs-xp)==(0,0):
        angle = 0.0    
    sep=(((xp-xs)*PlateScaleX)**2+((yp-ys)*PlateScaleY)**2)**(1/2)
    tempout.append((output[0],)+((yp,xp),(ys,xs))+output[3:8]+(angle,sep))

    bestprim.append((array_of_PSFs[output[3]]*output[4],(y1,x1)))
    bestsec.append((array_of_PSFs[output[5]]*output[6],(y2,x2)))
outputs=tempout

print('- - -\nBegin calculation of magnitudes')

FNU=image_header['PHOTFNU']
Fv=cameras[instrument]['Filters'][filtername]['Fv']
apcorr=cameras[instrument]['Filters'][filtername]['apcorr']
magsout=[]
ap_rad = cameras[instrument]['Aperture']
for prim,sec in zip(bestprim,bestsec):
    # Primary
    apertures = CircularAperture(prim[1],r=ap_rad)
    ap_table=phot(prim[0],apertures)
    counts=ap_table['aperture_sum'][0]
    if FNU*counts*apcorr/Fv>0:
        Pmag=-2.5*math.log(FNU*counts*apcorr/Fv,10)
    else:
        Pmag=float('NaN')
    # Secondary
    apertures = CircularAperture(sec[1],r=ap_rad)
    ap_table=phot(sec[0],apertures)
    counts=ap_table['aperture_sum'][0]
    if FNU*counts*apcorr/Fv>0:
        Smag=-2.5*math.log(FNU*counts*apcorr/Fv,10)
    else:
        Smag=float('NaN')
    
    magsout.append((Pmag,Smag))

tempout=[]
for output,mags in zip(outputs,magsout):
    tempout.append(output+mags)
outputs=tempout

out.write('\t---\nBinary Fit:')
outputs.sort(key=lambda x: x[0])
for itr,output in enumerate(outputs):
    out.write('\n'+str(itr+1)+'   error:%.5f\n'
        %output[0])
    out.write(' Primary: (%6.2f,%6.2f)  Secondary: (%6.2f,%6.2f)\n'
        %(output[1][1],output[1][0],output[2][1],output[2][0]))
    out.write(' Primary PSF:   %2i f: %9.4f (%5.2f%%)\n'
        %(output[3],output[4],output[4]/output[7]*100))
    out.write(' Secondary PSF: %2i f: %9.4f (%5.2f%%)\n'
        %(output[5],output[6],output[6]/output[7]*100))
    out.write(' Position Angle: %6.2f Separation: %6.4f\n'
        %(output[8],output[9]))
    out.write(' Primary Magnitude: %6.3f Secondary Magnitude: %6.3f\n'
        %(output[10],output[11]))
out.close()
print('end')
	


#Call 1st C++ program - this program handles the binning of the model and the single fit

#Open the csv files saved from the single fit
#print the necessary data and write it to the outfile

##out.write('iter: %i  psf:%2i  flux:%9.4f  minchi:%9.4f\n'
	 # % (iteration, single_fit, flux, minchi_val))

#print('- - -\nBegin selection of best binary fit')
#Call 2nd C++ program - this program handles the binary fitting


#This is for calcualtion of angles and separation
#Plate_Scale_X = cameras[instrument]['PlateScaleX']
#Plate_Scale_Y = cameras[instrument]['PlateScaleY']




# HST-PSF-Fitting
PSF fitting code for HST NICMOS data.

cameras.json provides information on the HST cameras that is needed for the fits

Each of the code files work as follows:

PSFfitting.py is the main file that reads in the image and model and does some prepatory work on the data for the fit. It also adds the random noise to the HST image and generates X number of these images*. It then calls each C++ file to complete a different part of the fitting process

PSF_prep bins the model image and prepares it to match the HST image so it can properly conduct the fit

single_fitting conducts the single fit and outputs the results to files that can be read back in by the main script

binary_fitting conducts the binary fit and outputs the results to files that can be read back in by the main script

The main script then uses the results to calculate the angles, separations, and primary and secondary magnitudes of each of the fits. After that is complete the main script conducts the monte carl simulation 

The rn_single conducts the single fits on the X number of images with the added random noise.

The rn_binary conducts the binary fits on the X number of images with the added random noise. It only runs the fits on the top 3 fits from the binary_fitting

After the fits are done on the images with random noise, the main script calculates the angles and separations for each. It then plots the distribution of the single fluxes, primary fluxes, secondary fluxes, angles, and separations for the single fits and each of the top 3 fits with the random noise added images

*The number of images can be changed depending on how many you want. I have them set at 100. To change it, go into the PSFfitting.py, rn_single.cpp, and rn_binary.cpp files and change the variable "size" to whatever number of random noise images you want to run. Keep in mind that the larger the number the longer the code will take to run.

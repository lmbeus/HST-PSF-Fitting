#include <iostream>
#include <istream>
#include <fstream>
#include <list>
#include <vector>
#include <cmath>
#include <string>

#include "xtensor/xtensor.hpp"
#include "xtensor/xarray.hpp"
#include "xtensor/xcsv.hpp"
#include "xtensor/xnpy.hpp"
#include "xtensor/xview.hpp"
#include "xtensor/xio.hpp"
#include "xtensor/xfixed.hpp"
#include "xtensor/xmath.hpp" 
#include "xsimd/xsimd.hpp"
using namespace std;

xt::xarray<xt::xarray<double>> BinPSFModel(xt::xarray<double> model_psf) 
{
	int nbin = 10;
	int nrows = model_psf.shape()[0];
	int ncols = model_psf.shape()[1];
	xt::xarray<xt::xarray<double>> array_of_PSFs = xt::zeros<xt::xarray<double>>({nbin * nbin});

	for (int i = 1; i <= nbin; ++i) {
		for (int j = 1; j <= nbin; ++j) {
			xt::xarray<double> temp_array = xt::view(model_psf, xt::range(j, nrows - (nbin - j)), xt::range(i, ncols - (nbin - i)));
			int a = temp_array.shape()[0] / nbin;
			int b = temp_array.shape()[1] / nbin;
			temp_array.reshape({a, nbin, b, nbin});
	//"append" the reshaped array (there is no append function for xarrays)
			if (i == 1) {
				array_of_PSFs(j - 1) = xt::sum(temp_array, {1,3}, xt::evaluation_strategy::immediate);
			}
			else if (i == 2) {
				array_of_PSFs(10 + j - 1) = xt::sum(temp_array, {1,3}, xt::evaluation_strategy::immediate);
			}
			else if (i == 3) {
				array_of_PSFs(20 + j - 1) = xt::sum(temp_array, {1,3}, xt::evaluation_strategy::immediate);
			}
			else if (i == 4) {
				array_of_PSFs(30 + j - 1) = xt::sum(temp_array, {1,3}, xt::evaluation_strategy::immediate);
			}
			else if (i == 5) {
				array_of_PSFs(40 + j - 1) = xt::sum(temp_array, {1,3}, xt::evaluation_strategy::immediate);
			}
			else if (i == 6) {
				array_of_PSFs(50 + j - 1) = xt::sum(temp_array, {1,3}, xt::evaluation_strategy::immediate);
			}
			else if (i == 7) {
				array_of_PSFs(60 + j - 1) = xt::sum(temp_array, {1,3}, xt::evaluation_strategy::immediate);
			}
			else if (i == 8) {
				array_of_PSFs(70 + j - 1) = xt::sum(temp_array, {1,3}, xt::evaluation_strategy::immediate);
			}
			else if (i == 9) {
				array_of_PSFs(80 + j - 1) = xt::sum(temp_array, {1,3}, xt::evaluation_strategy::immediate);
			}
			else if (i == 10) {
				array_of_PSFs(90 + j - 1) = xt::sum(temp_array, {1,3}, xt::evaluation_strategy::immediate);
			}
		}
	}
	return array_of_PSFs;
}

int main() {

//read in the npy files to xtensor arrays

	auto model_psf = xt::load_npy<double>("psf_model_data.npy");
	auto image_psf = xt::load_npy<double>("image_psf.npy");

	//bin the model array and get back the 100 binned variations
	xt::xarray<xt::xarray<double>> array_of_PSFs = BinPSFModel(model_psf);

	//prep the data for single and binary fit
	double sigma = 0.0; //for NICMOS
	int nrows = image_psf.shape()[0];
	int ncols = image_psf.shape()[1]; 
	
	for (int i = 0; i < nrows; ++i) {
		for (int j = 0; j < ncols; ++j) {
			if (image_psf(i,j) < 0 || image_psf(i,j) <= sigma) {
				image_psf(i,j) = 0;
			}
		}
	}
	//output image_psf and array_of_PSFs to npy files
	xt::dump_npy("image_psf.npy", image_psf);
 	int length = 100;
    	for (int i = 1; i <= length; ++i) {
    		string filename = "PSFmodel_" + to_string(i) + ".npy";
    		xt::dump_npy(filename, array_of_PSFs(i - 1));
    	}
	


	return 0;
}

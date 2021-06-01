#include <iostream>
#include <istream>
#include <fstream>
#include <list>
#include <vector>
#include <cmath>

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

//Get the index of the model with the smallest chi-squared value
int GetMinChiSquared(xt::xarray<xt::xarray<double>> array_of_PSFs, xt::xarray<double> image_psf, double flux, xt::xarray<double> weights)
{
	xt::xarray<double> chi_array = xt::zeros<xt::xarray<double>>({array_of_PSFs.shape()[0]});
	int cx = (int) ((array_of_PSFs(0).shape()[1] - 1) * .5);
	int cy = (int) ((array_of_PSFs(0).shape()[0] - 1) * .5);
	
	//iterate through the 100 models
	for (int i = 0; i < array_of_PSFs.shape()[0]; ++i) {
		xt::xarray<double> model_PSF = array_of_PSFs(i);
		xt::xarray<double> model_fluxed = (xt::view(model_PSF, xt::range(cy - 2, cy + 3 ), xt::range(cx - 2, cx + 3))) * flux;
		xt::xarray<double> chi_PSF = weights * (image_psf - model_fluxed);
		chi_array(i) = xt::sum(chi_PSF * chi_PSF, xt::evaluation_strategy::immediate)(0);
	}
	
	double min_chi_squared = xt::amin(chi_array)(0);
	
	//get the index (there appears to be no built-in xtensor function for this)
	int index = 0;
	for (int i = 0; i < chi_array.shape()[0] ; ++i) {
		if (chi_array(i) == min_chi_squared) {
			index = i;
			break;
		}
	}	
	return index;
}

//refine the flux and the chi squared values
xt::xarray<double> FitSinglePSF(xt::xarray<double> binned_model_psf,xt::xarray<double> image_psf, double base, xt::xarray<double> weights) 
{
	int cx = (int) ((binned_model_psf.shape()[1] - 1) * .5);
	int cy = (int) ((binned_model_psf.shape()[0] - 1) * .5);
	xt::xarray<double> return_values = xt::zeros<double>({2});
	xt::xarray<double> scale_array = xt::arange<double>(0, 10.0001, 0.0001);
	xt::xarray<double> model_psf_mid = xt::view(binned_model_psf, xt::range(cy - 2, cy + 3 ), xt::range(cx - 2, cx + 3));
	xt::xarray<double> fluxes = xt::zeros<double>({scale_array.shape()[0]});
	xt::xarray<double> chi_squareds = xt::zeros<double>({scale_array.shape()[0]});

	for (int i = 0; i < scale_array.shape()[0]; ++i) {
		double scale = scale_array(i);
		xt::xarray<double> temp_array = model_psf_mid * scale * base;
		fluxes(i) = scale * base;
		xt::xarray<double> temp = image_psf - temp_array;
		xt::xarray<double> chi = xt::sum(weights * temp * temp, xt::evaluation_strategy::immediate);
		chi_squareds(i) = chi(0);
	}
	
	double min_chi = xt::amin(chi_squareds)(0);
	int min_chi_index = 0;
	for (int i = 0; i < chi_squareds.shape()[0] ; ++i) {
		if (chi_squareds(i) == min_chi) {
			min_chi_index = i;
			break;
		}
	}	 
	return_values(0) = min_chi;
	return_values(1) = fluxes(min_chi_index);
	return return_values;
}	

int main() 
{	
	//read in model PSF's
	int array_length = 100;
	xt::xarray<xt::xarray<double>> array_of_PSFs = xt::zeros<xt::xarray<double>>({array_length});
	for (int i = 1; i <= array_length; ++i) {
		string filename = " PSFmodel_" + to_string(i) + ".npy";
		array_of_PSFs(i - 1) = xt::load_npy<double>(filename);
	}
	
	vector<double> flux_distribution;
	int size = 100;
	//loop through the 100 noise-added images
	for (int n = 0; n < size; ++n) { 
		string filename = "rn_image" + to_string(n) + ".npy";
		auto image_psf = xt::load_npy<double> (filename);

		double base = xt::sum(image_psf,{0,1}, xt::evaluation_strategy::immediate)(0);
		xt::xarray<double> weights
		{{0,0,0,0,0},
		 {0,1,1,1,0},
		 {0,1,1,1,0},
		 {0,1,1,1,0},
		 {0,0,0,0,0}}; 
		auto sqrt_weights = xt::sqrt(xt::abs(image_psf));
		double flux = base * 1;
		int iteration = 0;
		double comp = 0.0;
		double minchi_value = 0.0;
		double tol = .000001;
		int best_PSF = -9999;
		xt::xarray<double> single_fit = xt::zeros<double>({2});

		//fit the image to the model
		while (abs(comp - flux) > .000001 && iteration < 10) {
			comp = flux;
			best_PSF = GetMinChiSquared(array_of_PSFs, image_psf, flux, sqrt_weights);
			single_fit = FitSinglePSF(array_of_PSFs(best_PSF), image_psf, base, weights);
			minchi_value = single_fit(0);
			flux = single_fit(1);
			++iteration;
		}
		flux_distribution.push_back(flux);
	}
	xt::xarray<double> fluxes = xt::adapt(flux_distribution,{size});
	xt::dump_npy("flux_distributions.npy", fluxes);
	
	return 0;
}

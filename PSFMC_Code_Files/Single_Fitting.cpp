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
	//read in image_psf
	//read in array_of_PSFs
	auto image_psf = xt::load_npy<double> ("image_psf.npy");

	int array_length = 100;
	xt::xarray<xt::xarray<double>> array_of_PSFs = xt::zeros<xt::xarray<double>>({array_length});
    	for (int i = 1; i <= array_length; ++i) {
    		string filename = " PSFmodel_" + to_string(i) + ".npy";
    		array_of_PSFs(i - 1) = xt::load_npy<double>(filename);
    	}
	
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
	
	//Do the single fit
	cout << "Beginning Single Fit" << endl;
	cout << "iteration: " << iteration << " psf: ? flux: " << flux << endl;
	
	vector<int> iteration_vector;
	vector<int> PSF_number_vector;
	vector<double> flux_vector;
	vector<double> minchi_value_vector;
	vector<double> residual_error_vector;
	
	while (abs(comp - flux) > .000001 && iteration < 10) {
		comp = flux;
		best_PSF = GetMinChiSquared(array_of_PSFs, image_psf, flux, sqrt_weights);
		single_fit = FitSinglePSF(array_of_PSFs(best_PSF), image_psf, base, weights);
		minchi_value = single_fit(0);
		flux = single_fit(1);
		++iteration;
		
		cout << "iteration: " << iteration << " psf: " << best_PSF << " flux: " << 
			flux << " chi-squared: " << minchi_value << endl;
	
		//Try a residual error to compare to binary
		xt::xarray<double> the_model = array_of_PSFs(best_PSF);
		xt::xarray<int> model_coordinates = {2,2};

		int cx = (the_model.shape()[1] - 1) * .5;
		int cy = (the_model.shape()[0] - 1) * .5;
		
		int rangeY_1 = cy - model_coordinates(0);
		int rangeY_2 = cy - model_coordinates(0) + 5;
		int rangeX_1 = cx - model_coordinates(1);
		int rangeX_2 = cx - model_coordinates(1) + 5;
			
		xt::xarray<double> best_model = (xt::view(the_model, xt::range(rangeY_1, rangeY_2), xt::range(rangeX_1, rangeX_2)));
		auto PSF_sum = best_model * flux;
		auto pre_residual = xt::abs(image_psf - PSF_sum);
		auto residual = xt::sum(weights * pre_residual, xt::evaluation_strategy::immediate)(0);
		auto residual_error = residual / xt::sum(weights, xt::evaluation_strategy::immediate)(0);
		
		cout << "residual error: " << residual_error << endl;
		//Need: ITeration, PSF #, flux, min_chi_val
		iteration_vector.push_back(iteration);
		PSF_number_vector.push_back(best_PSF);
		flux_vector.push_back(flux);
		minchi_value_vector.push_back(minchi_value);
		residual_error_vector.push_back(residual_error);
	}
	
	cout << endl;
	xt::xarray<int> iteration_array = xt::adapt(iteration_vector, {iteration});
	xt::xarray<int> PSF_number_array = xt::adapt(PSF_number_vector, {iteration});
	xt::xarray<double> flux_array = xt::adapt(flux_vector, {iteration});
	xt::xarray<double> minchi_value_array = xt::adapt(minchi_value_vector, {iteration});
	xt::xarray<double> residual_error_array = xt::adapt(residual_error_vector, {iteration});
	
	//write the single fit results to npy files
	xt::dump_npy("iteration_array.npy", iteration_array);
	xt::dump_npy("PSF_number_array.npy", PSF_number_array);
	xt::dump_npy("flux_array.npy", flux_array);
	xt::dump_npy("minchi_value_array.npy", minchi_value_array);
	xt::dump_npy("residual_error_array.npy", residual_error_array);
	
	return 0;
}

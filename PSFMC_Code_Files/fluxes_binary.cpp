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
#include "xtensor/xstrides.hpp"
#include "xtensor/xsort.hpp"
#include "xtensor/xadapt.hpp"
using namespace std;

xt::xarray<xt::xarray<double>> BinaryBase(xt::xarray<int> primary_coordinates, xt::xarray<int> secondary_coordinates) 
{
	xt::xarray<int> w1 = xt::zeros<int>({5,5});
	xt::xarray<int> w2 = xt::zeros<int>({5,5});
	
	w1(primary_coordinates(0) - 1, primary_coordinates(1) - 1) = 1;
	w1(primary_coordinates(0) - 1, primary_coordinates(1)) = 1;
	w1(primary_coordinates(0) - 1, primary_coordinates(1) + 1) = 1;
	w1(primary_coordinates(0), primary_coordinates(1) - 1) = 1;
	w1(primary_coordinates(0), primary_coordinates(1)) = 1;
	w1(primary_coordinates(0), primary_coordinates(1) + 1) = 1;
	w1(primary_coordinates(0) + 1, primary_coordinates(1) - 1) = 1;
	w1(primary_coordinates(0) + 1, primary_coordinates(1)) = 1;
	w1(primary_coordinates(0) + 1, primary_coordinates(1) + 1) = 1;

	w2(secondary_coordinates(0) - 1, secondary_coordinates(1) - 1) = 1;
	w2(secondary_coordinates(0) - 1, secondary_coordinates(1)) = 1;
	w2(secondary_coordinates(0) - 1, secondary_coordinates(1) + 1) = 1;
	w2(secondary_coordinates(0), secondary_coordinates(1) - 1) = 1;
	w2(secondary_coordinates(0), secondary_coordinates(1)) = 1;
	w2(secondary_coordinates(0), secondary_coordinates(1) + 1) = 1;
	w2(secondary_coordinates(0) + 1, secondary_coordinates(1) - 1) = 1;
	w2(secondary_coordinates(0) + 1, secondary_coordinates(1)) = 1;
	w2(secondary_coordinates(0) + 1, secondary_coordinates(1) + 1) = 1;
	
	auto w = w1 || w2;
	xt::xarray<xt::xarray<int>> w_array = {w, w1, w2};
	
	return w_array;
}

xt::xarray<int> BinaryPSF(xt::xarray<xt::xarray<double>> array_of_PSFs, xt::xarray<int> secondary_coordinates, xt::xarray<double> image_psf, double flux, xt::xarray<int> w)
{
	xt::xarray<double> scale_array = xt::arange<double> (0.95, 0, -.05);
	int models_length = array_of_PSFs.shape()[0];
	int scale_length = scale_array.shape()[0];
	int chi_array_length = models_length * models_length * scale_length;
	
	vector<double> chi_array;
	chi_array.reserve(chi_array_length);
	
	int cx = (array_of_PSFs(0).shape()[1] - 1) * .5;
	int cy = (array_of_PSFs(0).shape()[0] - 1) * .5;
	
	for (int i = 0; i < models_length; ++i) { 
		xt::xarray<double> model_PSF = array_of_PSFs(i);
		xt::xarray<double> PSF1 = (xt::view(model_PSF, xt::range(cy - 2, cy + 3 ), xt::range(cx - 2, cx + 3)));
		for (int j = 0; j < models_length; ++j) {
			xt::xarray<double> model_PSF = array_of_PSFs(j);
			int rangeY_1 = cy - secondary_coordinates(0);
			int rangeY_2 = cy - secondary_coordinates(0) + 5;
			int rangeX_1 = cx - secondary_coordinates(1);
			int rangeX_2 = cx - secondary_coordinates(1) + 5;
			
			xt::xarray<double> PSF2 = (xt::view(model_PSF, xt::range(rangeY_1, rangeY_2), xt::range(rangeX_1, rangeX_2)));	
			for (int k = 0; k < scale_length; ++k) {
				double relative_flux = scale_array(k);
				xt::xarray<double> a = PSF1 * relative_flux;
				xt::xarray<double> b = PSF2 * (1 - relative_flux);
				xt::xarray<double> temp_array = (a + b) * flux;
				xt::xarray<double> chi_PSF = w * (image_psf - temp_array);
				double chi_value = xt::sum(chi_PSF * chi_PSF, xt::evaluation_strategy::immediate)(0);
				chi_array.push_back(chi_value);	
			}
		}	
	} 

	auto chi = xt::adapt(chi_array, {100, 100, 19});
	xt::xarray<int> best_PSFs = xt::adapt(xt::unravel_index(xt::argmin(chi)(0), chi.shape(), xt::layout_type::row_major), {3});
	
	return best_PSFs;
}

xt::xarray<double> BinaryRelativeFlux(xt::xarray<xt::xarray<double>> array_of_PSFs, xt::xarray<int> secondary_coordinates, xt::xarray<double> image_psf, xt::xarray<double> flux, xt::xarray<int> w, int best_primary, int best_secondary) 
{
	//ref = relative flux
	xt::xarray<int> primary_ref = xt::zeros<int>({9});
	xt::xarray<int> secondary_ref = xt::zeros<int>({9});
	
	if (best_primary < 10) {
		best_primary += 10;
	}
    	if (best_primary >= 90) {
        	best_primary -= 10;
        }
    	if (best_primary % 10 == 0) {
        	best_primary += 1;
	}
   	if (best_primary % 10 == 9) {
        	best_primary-= 1;
	}
	
	if (best_secondary < 10) {
		best_secondary  += 10;
	}
    	if (best_secondary >= 90) {
        	best_secondary -= 10;
        }
    	if (best_secondary % 10 == 0) {
        	best_secondary  += 1;
	}
   	if (best_secondary % 10 == 9) {
        	best_secondary  -= 1;
	}
	
	primary_ref(0) = best_primary - 11;
	primary_ref(1) = best_primary - 10;
	primary_ref(2) = best_primary - 9;
	primary_ref(3) = best_primary - 1;
	primary_ref(4) = best_primary;
	primary_ref(5) = best_primary + 1;
	primary_ref(6) = best_primary + 9;
	primary_ref(7) = best_primary + 10;
	primary_ref(8) = best_primary + 11;
	
	secondary_ref(0) = best_secondary - 11;
	secondary_ref(1) = best_secondary - 10;
	secondary_ref(2) = best_secondary - 9;
	secondary_ref(3) = best_secondary - 1;
	secondary_ref(4) = best_secondary;
	secondary_ref(5) = best_secondary + 1;
	secondary_ref(6) = best_secondary + 9;
	secondary_ref(7) = best_secondary + 10;
	secondary_ref(8) = best_secondary + 11;
	
	xt::xarray<double> scale_array = xt::arange<double>(.99, 0, -.01);
	
	int primary_length = primary_ref.shape()[0];
	int secondary_length = secondary_ref.shape()[0];
	int scale_length = scale_array.shape()[0];
	int chi_array_length = primary_length * secondary_length * scale_length;
	vector<double> chi_array;
	chi_array.reserve(chi_array_length);
	
	for (int i = 0; i < primary_length; ++i ) {
		int primary = primary_ref(i);
		xt::xarray<double> model_PSF = array_of_PSFs(primary);
		int cx = (model_PSF.shape()[1] - 1) * .5;
		int cy = (model_PSF.shape()[0] - 1) * .5;
		xt::xarray<double> PSF1 = (xt::view(model_PSF, xt::range(cy - 2, cy + 3 ), xt::range(cx - 2, cx + 3)));	
		for (int j = 0; j < secondary_length; ++j) {
			int secondary = secondary_ref(j);
			xt::xarray<double> model_PSF = array_of_PSFs(secondary);
			int rangeY_1 = cy - secondary_coordinates(0);
			int rangeY_2 = cy - secondary_coordinates(0) + 5;
			int rangeX_1 = cx - secondary_coordinates(1);
			int rangeX_2 = cx - secondary_coordinates(1) + 5;
			
			xt::xarray<double> PSF2 = (xt::view(model_PSF, xt::range(rangeY_1, rangeY_2), xt::range(rangeX_1, rangeX_2)));	
			for (int k = 0; k < scale_length; ++k) {
				double relative_flux = scale_array(k);
				xt::xarray<double> a = PSF1 * relative_flux;
				xt::xarray<double> b = PSF2 * (1 - relative_flux);
				xt::xarray<double> temp_array = (a + b) * flux;
				xt::xarray<double> chi_PSF = w * (image_psf - temp_array);
				double chi_value = xt::sum(chi_PSF * chi_PSF, xt::evaluation_strategy::immediate)(0);
				chi_array.push_back(chi_value);
			}
		}
	}
	
	xt::xarray<double> chi = xt::adapt(chi_array, {9, 9, 99});
	xt::xarray<int> best_PSFs = xt::adapt(xt::unravel_index(xt::argmin(chi)(0), chi.shape(), xt::layout_type::row_major), {3});
	double best_primary_ref = primary_ref(best_PSFs(0));
	double best_secondary_ref = secondary_ref(best_PSFs(1));
	double best_ref = scale_array(best_PSFs(2));
	xt::xarray<double> final_best_PSF = {best_primary_ref, best_secondary_ref, best_ref};
	
	return final_best_PSF;
}

double BinaryFit(xt::xarray<double> model_primary, xt::xarray<double> model_secondary, xt::xarray<int> secondary_coordinates, xt::xarray<double> image_psf, double base, xt::xarray<int> w, double relative_flux) 
{
	int cx = (model_primary.shape()[1] - 1) * .5;
	int cy = (model_primary.shape()[0] - 1) * .5;
	int rangeY_1 = cy - secondary_coordinates(0);
	int rangeY_2 = cy - secondary_coordinates(0) + 5;
	int rangeX_1 = cx - secondary_coordinates(1);
	int rangeX_2 = cx - secondary_coordinates(1) + 5;
	
	xt::xarray<double> PSF1 = (xt::view(model_primary, xt::range(cy - 2, cy + 3 ), xt::range(cx - 2, cx + 3)));		
	xt::xarray<double> PSF2 = (xt::view(model_secondary, xt::range(rangeY_1, rangeY_2), xt::range(rangeX_1, rangeX_2)));
	vector<double> flux_array;
	vector<double> chi_array;
	
	//change "range" to something like "Scale"
	xt::xarray<double> scale_array = xt::arange<double>(0, 10.01, 0.01);
	int scale_length = scale_array.shape()[0];
	flux_array.reserve(scale_length);
	chi_array.reserve(scale_length);
	xt::xarray<double> PSF_sum = PSF1 * relative_flux + PSF2 * (1 - relative_flux);
	double scale;
	for (int i = 0; i < scale_length; ++i) {
		scale = scale_array(i);
		xt::xarray<double> scale_PSF = PSF_sum * scale * base;
		flux_array.push_back(scale * base);
		xt::xarray<double> chi_PSF = w * (image_psf - scale_PSF);
		double chi_value = xt::sum(chi_PSF * chi_PSF, xt::evaluation_strategy::immediate)(0);
		chi_array.push_back(chi_value);
	}
		
	auto chi_squareds = xt::adapt(chi_array, {scale_length});
	auto fluxes = xt::adapt(flux_array, {scale_length});
	double min_chi = xt::amin(chi_squareds)(0);
	int min_chi_index = 0;
	int chi_length = chi_squareds.shape()[0];
	//USE A VARIABLE FOR CHI_SQUAREDS.SHAPE, CHECK IN THE OTHER FILES TOO
	for (int i = 0; i < chi_length; ++i) {
		if (chi_squareds(i) == min_chi) {
			min_chi_index = i;
			break;
		}
	}	
	
	double best_flux = fluxes(min_chi_index);
	
	return best_flux;	 
}

double SecondaryBinaryFit(xt::xarray<double> primary, xt::xarray<double> secondary, xt::xarray<int> secondary_coordinates, xt::xarray<int >w2, xt::xarray<double> image_psf, double flux, double relative_flux)
{
	int cx = (primary.shape()[1] - 1) * .5;
	int cy = (primary.shape()[0] - 1) * .5;
	int rangeY_1 = cy - secondary_coordinates(0);
	int rangeY_2 = cy - secondary_coordinates(0) + 5;
	int rangeX_1 = cx - secondary_coordinates(1);
	int rangeX_2 = cx - secondary_coordinates(1) + 5;
	
	xt::xarray<double> PSF1 = (xt::view(primary, xt::range(cy - 2, cy + 3), xt::range(cx - 2, cx + 3)));
	xt::xarray<double> PSF2 = (xt::view(secondary, xt::range(rangeY_1, rangeY_2), xt::range(rangeX_1, rangeX_2)));
	xt::xarray<double> scale_array = xt::arange<double>(0, 10.0001, .0001);
	int scale_length = scale_array.shape()[0];
	vector<double> flux_array;
	vector<double> chi_array;
	flux_array.reserve(scale_length);
	chi_array.reserve(scale_length);
	
	xt::xarray<double> relative_PSF2 = w2 * (image_psf - PSF1 * flux * relative_flux);
	for (int i = 0; i < scale_length; ++i)
	{
		double scale = scale_array(i);
		xt::xarray<double> scale_PSF2 =  w2 * PSF2 * scale * flux;
		flux_array.push_back(scale * flux);
		xt::xarray<double> chi_PSF = w2 * (relative_PSF2 - scale_PSF2);
		double chi_value = xt::sum(chi_PSF * chi_PSF, xt::evaluation_strategy::immediate)(0);
		chi_array.push_back(chi_value);
	}

	auto chi_squareds = xt::adapt(chi_array, {scale_length});
	auto fluxes = xt::adapt(flux_array, {scale_length});
	double min_chi = xt::amin(chi_squareds)(0);
	int min_chi_index = 0;
	int chi_length = chi_squareds.shape()[0];
	for (int i = 0; i < chi_length ; ++i) {
		if (chi_squareds(i) == min_chi) {
			min_chi_index = i;
			break;
		}
	}	
	
	double best_flux = fluxes(min_chi_index);
	
	return best_flux;
}

double RelativeFluxes(xt::xarray<double> PSF_1, xt::xarray<int> coordinates_1, xt::xarray<double> PSF_2, xt::xarray<int> coordinates_2, xt::xarray<int> w, xt::xarray<double> image_psf, double base, int flux) 
{
	int cx = (PSF_1.shape()[1] - 1) * .5;
	int cy = (PSF_2.shape()[0] - 1) * .5;
	
	int P_rangeY_1 = cy - coordinates_1(0);
	int P_rangeY_2 = cy - coordinates_1(0) + 5;
	int P_rangeX_1 = cx - coordinates_1(1);
	int P_rangeX_2 = cx - coordinates_1(1) + 5;
	
	int S_rangeY_1 = cy - coordinates_2(0);
	int S_rangeY_2 = cy - coordinates_2(0) + 5;
	int S_rangeX_1 = cx - coordinates_2(1);
	int S_rangeX_2 = cx - coordinates_2(1) + 5;

	xt::xarray<double> PSF1 = (xt::view(PSF_1, xt::range(P_rangeY_1, P_rangeY_2), xt::range(P_rangeX_1, P_rangeX_2)));
	xt::xarray<double> PSF2 = (xt::view(PSF_2, xt::range(S_rangeY_1, S_rangeY_2), xt::range(S_rangeX_1, S_rangeX_2)));
	
	xt::xarray<double> scale_array = xt::arange<double>(0, 10.0001, .0001);
	int scale_length = scale_array.shape()[0];
	vector<double> flux_array;
	vector<double> chi_array;
	flux_array.reserve(scale_length);
	chi_array.reserve(scale_length);
	
	auto rPSF = w * (image_psf - PSF2 * flux);
	for (int i = 0; i < scale_length; ++i) {
		double scale = scale_array(i);
		xt::xarray<double> scale_PSF = w * PSF1 * scale * base;
		flux_array.push_back(scale * base);
		xt::xarray<double> chi_PSF = w * (rPSF - scale_PSF);
		double chi_value = xt::sum(chi_PSF * chi_PSF, xt::evaluation_strategy::immediate)(0);
		chi_array.push_back(chi_value);
	}
	
	auto chi_squareds = xt::adapt(chi_array, {scale_length});
	auto fluxes = xt::adapt(flux_array, {scale_length});
	double min_chi = xt::amin(chi_squareds)(0);
	int min_chi_index = 0;
	int chi_length = chi_squareds.shape()[0];
	for (int i = 0; i < chi_length ; ++i) {
		if (chi_squareds(i) == min_chi) {
			min_chi_index = i;
			break;
		}
	}	
	
	double best_flux = fluxes(min_chi_index);
	
	return best_flux;
}


//MAIN 
int main() 
{
	int array_length = 100;
	xt::xarray<xt::xarray<double>> array_of_PSFs = xt::zeros<xt::xarray<double>>({array_length});
    	for (int i = 1; i <= array_length; ++i) {
    		string filename = " PSFmodel_" + to_string(i) + ".npy";
    		array_of_PSFs(i - 1) = xt::load_npy<double>(filename);
    	}
    	
	auto coordinates = xt::load_npy<double>("coords.npy");
	auto center_load = xt::load_npy<double>("center.npy");
	xt::xarray<int> center = {(int) center_load(0,0), (int) center_load(1,1)};
    	
    	xt::xarray<int> primary_coordinates = {2,2};
    	xt::xarray<int> secondary_coordinates = xt::zeros<double>({2});
    	
    	//vectors to store the output values (one set for each of the 9 binary fits)
    	vector<double> primary_fluxes;
    	vector<double> secondary_fluxes;
    	
	vector<double> primary_flux_distribution1;
	vector<double> secondary_flux_distribution1;
	vector<double> sum_flux_distribution1;
	
	vector<double> primary_flux_distribution2;
	vector<double> secondary_flux_distribution2;
	vector<double> sum_flux_distribution2;
	
	vector<double> primary_flux_distribution3;
	vector<double> secondary_flux_distribution3;
	vector<double> sum_flux_distribution3;
	
	vector<double> primary_flux_distribution4;
	vector<double> secondary_flux_distribution4;
	vector<double> sum_flux_distribution4;
	
	vector<double> primary_flux_distribution5;
	vector<double> secondary_flux_distribution5;
	vector<double> sum_flux_distribution5;
	
	vector<double> primary_flux_distribution6;
	vector<double> secondary_flux_distribution6;
	vector<double> sum_flux_distribution6;
	
	vector<double> primary_flux_distribution7;
	vector<double> secondary_flux_distribution7;
	vector<double> sum_flux_distribution7;
	
	vector<double> primary_flux_distribution8;
	vector<double> secondary_flux_distribution8;
	vector<double> sum_flux_distribution8;
	
	vector<double> primary_flux_distribution9;
	vector<double> secondary_flux_distribution9;
	vector<double> sum_flux_distribution9;
    	
    	
    	int iter = 0;
    	int iter2 = 0;
    	#pragma omp parallel for
    	for (int n = 0; n < 100; ++n) { 
		string filename = "rn_image" + to_string(n) + ".npy";
		auto image_psf = xt::load_npy<double> (filename);
	    	
	    	//do the binary fit
	    	iter2 = 0;
	    	for (int i = 0; i < coordinates.shape()[0]; ++i) {
	    		xt::xarray<int> secondary = xt::row(coordinates, i);
	    		secondary_coordinates(0) = secondary(0) - center(0) + 2;
	    		secondary_coordinates(1) = secondary(1) - center(1) + 2;
	    		
	    		xt::xarray<xt::xarray<double>> w_array = BinaryBase(primary_coordinates, secondary_coordinates);
			xt::xarray<double> w = w_array(0);
			xt::xarray<int> w1 = w_array(1);
			xt::xarray<int> w2 = w_array(2);
			auto sqrt_weights = xt::sqrt(xt::abs(image_psf));
			
	    		int iteration = 0;
	    		int best_primary = 0;
	    		int best_secondary = 0;
	    		double best_RF = 0.0;
	    		double base = xt::sum(w * image_psf, xt::evaluation_strategy::immediate)(0);
	    		
	    		double comp = 0;
	    		double flux = base * 1;
	    		
	    		while (abs(comp - flux) > .000001 && iteration < 10) {
	    			comp = flux;
	    			xt::xarray<int> best_PSFs =  BinaryPSF(array_of_PSFs, secondary_coordinates, image_psf, flux, sqrt_weights);
	    			best_primary = best_PSFs(0);
	    			best_secondary = best_PSFs(1);
				
	    			xt::xarray<double> best_PSFs_ref = BinaryRelativeFlux(array_of_PSFs, secondary_coordinates, image_psf, flux, sqrt_weights, best_primary, best_secondary);
	    			best_primary = best_PSFs_ref(0);
	    			best_secondary = best_PSFs_ref(1);
	    			best_RF = best_PSFs_ref(2);
	    			
				flux = BinaryFit(array_of_PSFs(best_primary), array_of_PSFs(best_secondary), secondary_coordinates, image_psf, base, w, best_RF);
	    			++iteration ;
	    		}
	    		
	    		xt::xarray<double> primary_PSF = array_of_PSFs(best_primary);
	    		xt::xarray<double> secondary_PSF = array_of_PSFs(best_secondary);

	    		double comp_primary = 1.0;
	    		double comp_secondary = 1.0;
	    		double flux_primary = 2.0;

	    		double flux_secondary = SecondaryBinaryFit(primary_PSF, secondary_PSF, secondary_coordinates, w2, image_psf, flux, best_RF);

	    		iteration = 0;
	    		
	    		//Final WHILE LOOP 
			while (abs(comp_primary - flux_primary) > .000001 && abs(comp_secondary - flux_secondary) > .000001) {
				comp_primary = flux_primary;
				comp_secondary = flux_secondary;
				if (iteration >= 10) {
					break;

				}
				
				flux_primary = RelativeFluxes(primary_PSF, primary_coordinates, secondary_PSF, secondary_coordinates, w1, image_psf, base, flux_secondary);
				flux_secondary = RelativeFluxes(secondary_PSF, secondary_coordinates, primary_PSF, primary_coordinates, w2, image_psf, base, flux_primary);
				++iteration;
			}
			
			double sum_flux = flux_primary + flux_secondary;
			primary_fluxes.push_back(flux_primary);
			secondary_fluxes.push_back(flux_secondary);
			if (i == 0) {
				primary_flux_distribution1.push_back(flux_primary);
				secondary_flux_distribution1.push_back(flux_secondary);
				sum_flux_distribution1.push_back(sum_flux);
			}
			else if (i == 1) {
				primary_flux_distribution2.push_back(flux_primary);
				secondary_flux_distribution2.push_back(flux_secondary);
				sum_flux_distribution2.push_back(sum_flux);
			}
			else if (i == 2) {
				primary_flux_distribution3.push_back(flux_primary);
				secondary_flux_distribution3.push_back(flux_secondary);
				sum_flux_distribution3.push_back(sum_flux);
			}
			else if (i == 3) {
				primary_flux_distribution4.push_back(flux_primary);
				secondary_flux_distribution4.push_back(flux_secondary);
				sum_flux_distribution4.push_back(sum_flux);
			}
			else if (i == 4) {
				primary_flux_distribution5.push_back(flux_primary);
				secondary_flux_distribution5.push_back(flux_secondary);
				sum_flux_distribution5.push_back(sum_flux);
			}
			else if (i == 5) {
				primary_flux_distribution6.push_back(flux_primary);
				secondary_flux_distribution6.push_back(flux_secondary);
				sum_flux_distribution6.push_back(sum_flux);
			}
			else if (i == 6) {
				primary_flux_distribution7.push_back(flux_primary);
				secondary_flux_distribution7.push_back(flux_secondary);
				sum_flux_distribution7.push_back(sum_flux);
			}
			else if (i == 7) {
				primary_flux_distribution8.push_back(flux_primary);
				secondary_flux_distribution8.push_back(flux_secondary);
				sum_flux_distribution8.push_back(sum_flux);
			}
			else if (i == 8) {
				primary_flux_distribution9.push_back(flux_primary);
				secondary_flux_distribution9.push_back(flux_secondary);
				sum_flux_distribution9.push_back(sum_flux);
			}
			
			++iter2;
	    	}
	    	++iter;
	}
	
	int size = 900;
	xt::xarray<double> xt_primary_fluxes = xt::adapt(primary_fluxes, {size});
	xt::xarray<double> xt_secondary_fluxes = xt::adapt(secondary_fluxes, {size});
	
	xt::dump_npy("primary_fluxes.npy", xt_primary_fluxes);
	xt::dump_npy("secondary_fluxes.npy", xt_secondary_fluxes);
	
	xt::xarray<double> primary_fluxes1 = xt::adapt(primary_flux_distribution1,{size});
	xt::xarray<double> secondary_fluxes1 = xt::adapt(secondary_flux_distribution1,{size});
	xt::xarray<double> sum_fluxes1 = xt::adapt(sum_flux_distribution1,{size});
	
	xt::xarray<double> primary_fluxes2 = xt::adapt(primary_flux_distribution2,{size});
	xt::xarray<double> secondary_fluxes2 = xt::adapt(secondary_flux_distribution2,{size});
	xt::xarray<double> sum_fluxes2 = xt::adapt(sum_flux_distribution2,{size});
	
	xt::xarray<double> primary_fluxes3 = xt::adapt(primary_flux_distribution3,{size});
	xt::xarray<double> secondary_fluxes3 = xt::adapt(secondary_flux_distribution3,{size});
	xt::xarray<double> sum_fluxes3 = xt::adapt(sum_flux_distribution3,{size});
	
	xt::xarray<double> primary_fluxes4 = xt::adapt(primary_flux_distribution4,{size});
	xt::xarray<double> secondary_fluxes4 = xt::adapt(secondary_flux_distribution4,{size});
	xt::xarray<double> sum_fluxes4 = xt::adapt(sum_flux_distribution4,{size});
	
	xt::xarray<double> primary_fluxes5 = xt::adapt(primary_flux_distribution5,{size});
	xt::xarray<double> secondary_fluxes5 = xt::adapt(secondary_flux_distribution5,{size});
	xt::xarray<double> sum_fluxes5 = xt::adapt(sum_flux_distribution5,{size});
	
	xt::xarray<double> primary_fluxes6 = xt::adapt(primary_flux_distribution6,{size});
	xt::xarray<double> secondary_fluxes6 = xt::adapt(secondary_flux_distribution6,{size});
	xt::xarray<double> sum_fluxes6 = xt::adapt(sum_flux_distribution6,{size});
	
	xt::xarray<double> primary_fluxes7 = xt::adapt(primary_flux_distribution7,{size});
	xt::xarray<double> secondary_fluxes7 = xt::adapt(secondary_flux_distribution7,{size});
	xt::xarray<double> sum_fluxes7 = xt::adapt(sum_flux_distribution7,{size});
	
	xt::xarray<double> primary_fluxes8 = xt::adapt(primary_flux_distribution8,{size});
	xt::xarray<double> secondary_fluxes8 = xt::adapt(secondary_flux_distribution8,{size});
	xt::xarray<double> sum_fluxes8 = xt::adapt(sum_flux_distribution8,{size});
	
	xt::xarray<double> primary_fluxes9 = xt::adapt(primary_flux_distribution9,{size});
	xt::xarray<double> secondary_fluxes9 = xt::adapt(secondary_flux_distribution9,{size});
	xt::xarray<double> sum_fluxes9 = xt::adapt(sum_flux_distribution9,{size});

	
	xt::dump_npy("primary_fluxes1.npy", primary_fluxes1);
	xt::dump_npy("secondary_fluxes1.npy", secondary_fluxes1);
	xt::dump_npy("sum_fluxes1.npy", sum_fluxes1);
	
	xt::dump_npy("primary_fluxes2.npy", primary_fluxes2);
	xt::dump_npy("secondary_fluxes2.npy", secondary_fluxes2);
	xt::dump_npy("sum_fluxes2.npy", sum_fluxes2);
	
	xt::dump_npy("primary_fluxes3.npy", primary_fluxes3);
	xt::dump_npy("secondary_fluxes3.npy", secondary_fluxes3);
	xt::dump_npy("sum_fluxes3.npy", sum_fluxes3);
	
	xt::dump_npy("primary_fluxes4.npy", primary_fluxes4);
	xt::dump_npy("secondary_fluxes4.npy", secondary_fluxes4);
	xt::dump_npy("sum_fluxes4.npy", sum_fluxes4);
	
	xt::dump_npy("primary_fluxes5.npy", primary_fluxes5);
	xt::dump_npy("secondary_fluxes5.npy", secondary_fluxes5);
	xt::dump_npy("sum_fluxes5.npy", sum_fluxes5);
	
	xt::dump_npy("primary_fluxes6.npy", primary_fluxes6);
	xt::dump_npy("secondary_fluxes6.npy", secondary_fluxes6);
	xt::dump_npy("sum_fluxes6.npy", sum_fluxes6);
	
	xt::dump_npy("primary_fluxes7.npy", primary_fluxes7);
	xt::dump_npy("secondary_fluxes7.npy", secondary_fluxes7);
	xt::dump_npy("sum_fluxes7.npy", sum_fluxes7);
	
	xt::dump_npy("primary_fluxes8.npy", primary_fluxes8);
	xt::dump_npy("secondary_fluxes8.npy", secondary_fluxes8);
	xt::dump_npy("sum_fluxes8.npy", sum_fluxes8);
	
	xt::dump_npy("primary_fluxes9.npy", primary_fluxes9);
	xt::dump_npy("secondary_fluxes9.npy", secondary_fluxes9);
	xt::dump_npy("sum_fluxes9.npy", sum_fluxes9);
	
	
	return 0;
}

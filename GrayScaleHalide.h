#pragma once

#include <opencv2/core.hpp>

#ifndef EXP_HALIDE_GRAYSCALE
	#ifndef BUILD_HALIDE_GRAYSCALE
		#pragma comment(lib, "Projeto_Final_Programacao_Paralela_Halide.lib")
		#define EXP_HALIDE_GRAYSCALE __declspec(dllimport)
	#else
		#define EXP_HALIDE_GRAYSCALE __declspec(dllexport)
	#endif 
#endif 

extern "C" EXP_HALIDE_GRAYSCALE cv::Mat grayScaleWithHalide(cv::Mat image);

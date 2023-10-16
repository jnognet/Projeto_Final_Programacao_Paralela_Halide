#pragma once

#include <stdio.h>
#include <string>

#ifndef EXP_HALIDECPU_GRAYSCALE
	#ifndef BUILD_HALIDE
		#pragma comment(lib, "Projeto_Final_Programacao_Paralela_Halide.lib")
		#define EXP_HALIDECPU_GRAYSCALE __declspec(dllimport)
	#else
		#define EXP_HALIDECPU_GRAYSCALE __declspec(dllexport)
	#endif
#endif

extern "C" EXP_HALIDECPU_GRAYSCALE bool grayScaleWithHalideCPU(std::string file_src, std::string file_dst);


#ifndef EXP_HALIDEGPU_GRAYSCALE
	#ifndef BUILD_HALIDE
		#pragma comment(lib, "Projeto_Final_Programacao_Paralela_Halide.lib")
		#define EXP_HALIDEGPU_GRAYSCALE __declspec(dllimport)
	#else
		#define EXP_HALIDEGPU_GRAYSCALE __declspec(dllexport)
	#endif
#endif

extern "C" EXP_HALIDEGPU_GRAYSCALE bool grayScaleWithHalideGPU(std::string file_src, std::string file_dst);


#ifndef EXP_HALIDECPU_COMPLEMENT
	#ifndef BUILD_HALIDE
		#pragma comment(lib, "Projeto_Final_Programacao_Paralela_Halide.lib")
		#define EXP_HALIDECPU_COMPLEMENT __declspec(dllimport)
	#else
		#define EXP_HALIDECPU_COMPLEMENT __declspec(dllexport)
	#endif
#endif

extern "C" EXP_HALIDECPU_COMPLEMENT bool complementWithHalideCPU(std::string file_src, std::string file_dst);


#ifndef EXP_HALIDEGPU_COMPLEMENT
	#ifndef BUILD_HALIDE
		#pragma comment(lib, "Projeto_Final_Programacao_Paralela_Halide.lib")
		#define EXP_HALIDEGPU_COMPLEMENT __declspec(dllimport)
	#else
		#define EXP_HALIDEGPU_COMPLEMENT __declspec(dllexport)
	#endif
#endif

extern "C" EXP_HALIDEGPU_COMPLEMENT bool complementWithHalideGPU(std::string file_src, std::string file_dst);

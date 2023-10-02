#define BUILD_HALIDE_GRAYSCALE

#include <stdio.h>
#include <string>

#include "GrayScaleHalide.h"

#include "Halide.h"
#include "halide_image_io.h"

#include <opencv2/core.hpp>
#include <opencv2/video.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>

using namespace Halide;
using namespace Halide::Tools;

void convertMat2Halide(cv::Mat& src, Buffer<uint8_t>& dest)
{
    const int ch = src.channels();
    if (ch == 1)
    {
        for (int j = 0; j < src.rows; j++)
        {
            for (int i = 0; i < src.cols; i++)
            {
                dest(i, j) = src.at<uchar>(j, i);
            }
        }
    }
    else if (ch == 3)
    {
        for (int j = 0; j < src.rows; j++)
        {
            for (int i = 0; i < src.cols; i++)
            {
                dest(i, j, 0) = src.at<uchar>(j, 3 * i);
                dest(i, j, 1) = src.at<uchar>(j, 3 * i + 1);
                dest(i, j, 2) = src.at<uchar>(j, 3 * i + 2);
            }
        }
    }
}

void convertHalide2Mat(const Buffer<uint8_t>& src, cv::Mat& dest)
{
    if (dest.empty()) dest.create(cv::Size(src.width(), src.height()), CV_MAKETYPE(CV_8U, src.channels()));
    const int ch = dest.channels();
    if (ch == 1)
    {
        for (int j = 0; j < dest.rows; j++)
        {
            for (int i = 0; i < dest.cols; i++)
            {
                dest.at<uchar>(j, i) = src(i, j);
            }
        }
    }
    else if (ch == 3)
    {
        for (int j = 0; j < dest.rows; j++)
        {
            for (int i = 0; i < dest.cols; i++)
            {
                dest.at<uchar>(j, 3 * i + 0) = src(i, j, 0);
                dest.at<uchar>(j, 3 * i + 1) = src(i, j, 1);
                dest.at<uchar>(j, 3 * i + 2) = src(i, j, 2);
            }
        }
    }
}

extern "C" EXP_HALIDECPU_GRAYSCALE bool grayScaleWithHalideCPU(std::string file_src, std::string file_dst)
{    
    Halide::Buffer<uint8_t> input = load_image(file_src);

	Halide::Var x, y, c;
	Halide::Func grayscale;

	grayscale(x, y, c) = Halide::cast<uint8_t>(
		min(
			0.299f * input(x, y, 0) + 0.587f * input(x, y, 1) +	0.114f * input(x, y, 2), 255.0f
		)
	);

	Halide::Var x_outer, y_outer, x_inner, y_inner, tile_index;
	grayscale
		.tile(x, y, x_outer, y_outer, x_inner, y_inner, 64, 64)
		.fuse(x_outer, y_outer, tile_index)
		.parallel(tile_index);

	Halide::Var x_inner_outer, y_inner_outer, x_vectors, y_pairs;
	grayscale
		.tile(x_inner, y_inner, x_inner_outer, y_inner_outer, x_vectors, y_pairs, 4, 2)
		.vectorize(x_vectors)
		.unroll(y_pairs);

	Halide::Buffer<uint8_t> output = grayscale.realize({ input.width(), input.height(), input.channels() });

    save_image(output, file_dst);

	return true;
}

Target find_gpu_target() {
    Target target = get_host_target();

    std::vector<Target::Feature> features_to_try;
    features_to_try.push_back(Target::OpenCL);

    for (Target::Feature f : features_to_try) {
        Target new_target = target.with_feature(f);
        if (host_supports_target_device(new_target)) {
            return new_target;
        }
    }
    return target;
}

extern "C" EXP_HALIDEGPU_GRAYSCALE bool grayScaleWithHalideGPU(std::string file_src, std::string file_dst)
{
    Target target = find_gpu_target();
    if (!target.has_gpu_feature()) {
        return false;
    }

    Halide::Var x, y, c, i;
    Halide::Func grayscale, graycalc, lut;
    Halide::Var block, thread;
    Halide::Buffer<uint8_t> input = load_image(file_src);

    // kernel
    lut(i) = Halide::cast<uint8_t>(i);
    graycalc(x, y, c) = Halide::cast<uint8_t>(
            min(
                0.299f * input(x, y, 0) + 0.587f * input(x, y, 1) + 0.114f * input(x, y, 2), 255.0f
            )
        );
    grayscale(x, y, c) = lut(graycalc(x, y, c));

    // schedule_for_gpu    
    lut.compute_root();
    lut.split(i, block, thread, 1024);
    lut.gpu_blocks(block)
        .gpu_threads(thread);
    grayscale.compile_jit(target);

    Buffer<uint8_t> output(input.width(), input.height(), input.channels());    
    // run
    grayscale.realize(output);
    output.copy_to_host();

    save_image(output, file_dst);

    return true;
}

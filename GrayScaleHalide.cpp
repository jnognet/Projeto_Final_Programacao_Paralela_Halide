#define BUILD_HALIDE_GRAYSCALE

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

extern "C" cv::Mat EXP_HALIDE_GRAYSCALE grayScaleWithHalide(cv::Mat image)
{    
    Halide::Buffer<uint8_t> input(image.cols, image.rows, image.channels());
    convertMat2Halide(image, input);

	Halide::Var x, y, c;
	Halide::Func grayscale("grayscale");

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

    cv::Mat outimage(cv::Size(output.width(), output.height()), CV_MAKETYPE(CV_8U, output.channels()));
    convertHalide2Mat(output, outimage);

	return outimage;
}
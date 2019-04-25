/*  The MIT License (MIT)
 *  
 *  Copyright (c) 2019 Matias Koskela / Tampere University
 *  Copyright (c) 2018 Kalle Immonen / Tampere University of Technology
 *  
 *  Permission is hereby granted, free of charge, to any person obtaining a copy
 *  of this software and associated documentation files (the "Software"), to deal
 *  in the Software without restriction, including without limitation the rights
 *  to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 *  copies of the Software, and to permit persons to whom the Software is
 *  furnished to do so, subject to the following conditions:
 *  
 *  The above copyright notice and this permission notice shall be included in
 *  all copies or substantial portions of the Software.
 *  
 *  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 *  IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 *  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 *  AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 *  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 *  OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 *  THE SOFTWARE.
 */

#include "OpenImageIO/imageio.h"
#include "CLUtils/CLUtils.hpp"

#define _CRT_SECURE_NO_WARNINGS
#define STR_HELPER(x) #x
#define STR(x) STR_HELPER(x)

// ### Choose your OpenCL device and platform with these defines ###
#define PLATFORM_INDEX 0
#define DEVICE_INDEX 0


// ### Edit these defines if you have different input ###
// TODO detect IMAGE_SIZES automatically from the input files
#define IMAGE_WIDTH 1280
#define IMAGE_HEIGHT 720
// TODO detect FRAME_COUNT from the input files
#define FRAME_COUNT 60
// Location where input frames and feature buffers are located
#define INPUT_DATA_PATH ../data/frames
#define INPUT_DATA_PATH_STR STR(INPUT_DATA_PATH)
// camera_matrices.h is expected to be in the same folder
#include STR(INPUT_DATA_PATH/camera_matrices.h)
// These names are appended with NN.exr, where NN is the frame number
#define NOISY_FILE_NAME INPUT_DATA_PATH_STR"/color"
#define NORMAL_FILE_NAME INPUT_DATA_PATH_STR"/shading_normal"
#define POSITION_FILE_NAME INPUT_DATA_PATH_STR"/world_position"
#define ALBEDO_FILE_NAME INPUT_DATA_PATH_STR"/albedo"
#define OUTPUT_FILE_NAME "outputs/output"


// ### Edit these defines if you want to experiment different parameters ###
// The amount of noise added to feature buffers to cancel sigularities
#define NOISE_AMOUNT 1e-2
// The amount of new frame used in accumulated frame (1.f would mean no accumulation).
#define BLEND_ALPHA 0.2f
#define SECOND_BLEND_ALPHA 0.1f
#define TAA_BLEND_ALPHA 0.2f
// NOTE: if you want to use other than normal and world_position data you have to make
// it available in the first accumulation kernel and in the weighted sum kernel
#define NOT_SCALED_FEATURE_BUFFERS \
"1.f,"\
"normal.x,"\
"normal.y,"\
"normal.z,"\
// The next features are not in the range from -1 to 1 so they are scaled to be from 0 to 1.
#define SCALED_FEATURE_BUFFERS \
"world_position.x,"\
"world_position.y,"\
"world_position.z,"\
"world_position.x*world_position.x,"\
"world_position.y*world_position.y,"\
"world_position.z*world_position.z"


// ### Edit these defines to change optimizations for your target hardware ###
// If 1 uses ~half local memory space for R, but computing indexes is more complicated
#define COMPRESSED_R 1
// If 1 stores tmp_data to private memory when it is loaded for dot product calculation
#define CACHE_TMP_DATA 1
// If 1 tmp_data buffer is in half precision for faster load and store.
// NOTE: if world position values are greater than 256 this cannot be used because
// 256*256 is infinity in half-precision
#define USE_HALF_PRECISION_IN_TMP_DATA 1
// If 1 adds __attribute__((reqd_work_group_size(256, 1, 1))) to fitter and
// accumulate_noisy_data kernels. With some codes, attribute made the kernels faster and
// with some it slowed them down.
#define ADD_REQD_WG_SIZE 1
// These local sizes are used with 2D kernels which do not require spesific local size
// (Global sizes are always a multiple of 32)
#define LOCAL_WIDTH 8
#define LOCAL_HEIGHT 8
// Fastest on AMD Radeon Vega Frontier Edition was (LOCAL_WIDTH = 256, LOCAL_HEIGHT = 1)
// Fastest on Nvidia Titan Xp was (LOCAL_WIDTH = 32, LOCAL_HEIGHT = 1)



// ### Do not edit defines after this line unless you know what you are doing ###
// For example, other than 32x32 blocks are not supported
#define BLOCK_EDGE_LENGTH 32
#define BLOCK_PIXELS (BLOCK_EDGE_LENGTH * BLOCK_EDGE_LENGTH)
// Rounds image sizes up to next multiple of BLOCK_EDGE_LENGTH
#define WORKSET_WIDTH (BLOCK_EDGE_LENGTH * \
   ((IMAGE_WIDTH + BLOCK_EDGE_LENGTH - 1) / BLOCK_EDGE_LENGTH))
#define WORKSET_HEIGHT (BLOCK_EDGE_LENGTH * \
   ((IMAGE_HEIGHT + BLOCK_EDGE_LENGTH - 1) / BLOCK_EDGE_LENGTH))
#define WORKSET_WITH_MARGINS_WIDTH (WORKSET_WIDTH + BLOCK_EDGE_LENGTH)
#define WORKSET_WITH_MARGINS_HEIGHT (WORKSET_HEIGHT + BLOCK_EDGE_LENGTH)
#define OUTPUT_SIZE (WORKSET_WIDTH * WORKSET_HEIGHT)
// 256 is the maximum local size on AMD GCN
// Synchronization within 32x32=1024 block requires unrollign four times
#define LOCAL_SIZE 256
#define FITTER_GLOBAL (LOCAL_SIZE * ((WORKSET_WITH_MARGINS_WIDTH / BLOCK_EDGE_LENGTH) * \
   (WORKSET_WITH_MARGINS_HEIGHT / BLOCK_EDGE_LENGTH)))

// Creates two same buffers and swap() call can be used to change which one is considered
// current and which one previous
template <class T>
class Double_buffer
{
    private:
        T a, b;
        bool swapped;

    public:
        template <typename... Args>
        Double_buffer(Args... args) : a(args...), b(args...), swapped(false){};
        T *current() { return swapped ? &a : &b; }
        T *previous() { return swapped ? &b : &a; }
        void swap() { swapped = !swapped; }
};

struct Operation_result
{
    bool success;
    std::string error_message;
    Operation_result(bool success, const std::string &error_message = "") :
        success(success), error_message(error_message) {}
};

Operation_result read_image_file(
    const std::string &file_name, const int frame, float *buffer)
{
    OpenImageIO::ImageInput *in = OpenImageIO::ImageInput::open(
        file_name + std::to_string(frame) + ".exr");
    if (!in || in->spec().width != IMAGE_WIDTH ||
        in->spec().height != IMAGE_HEIGHT || in->spec().nchannels != 3)
    {

        return {false, "Can't open image file or it has wrong type: " + file_name};
    }

    // NOTE: this converts .exr files that might be in halfs to single precision floats
    // In the dataset distributed with the BMFR paper all exr files are in single precision
    in->read_image(OpenImageIO::TypeDesc::FLOAT, buffer);
    in->close();

    return {true};
}

Operation_result load_image(cl_float *image, const std::string file_name, const int frame)
{
    Operation_result result = read_image_file(file_name, frame, image);
    if (!result.success)
        return result;

    return {true};
}

float clamp(float value, float minimum, float maximum)
{
    return std::max(std::min(value, maximum), minimum);
}

int tasks()
{

    printf("Initialize.\n");
    clutils::CLEnv clEnv;
    cl::Context &context(clEnv.addContext(PLATFORM_INDEX));

    // Find name of the used device
    std::string deviceName;
    clEnv.devices[0][DEVICE_INDEX].getInfo(CL_DEVICE_NAME, &deviceName);
    printf("Using device named: %s\n", deviceName.c_str());

    cl::CommandQueue &queue(clEnv.addQueue(0, DEVICE_INDEX, CL_QUEUE_PROFILING_ENABLE));

    std::string features_not_scaled(NOT_SCALED_FEATURE_BUFFERS);
    std::string features_scaled(SCALED_FEATURE_BUFFERS);
    const int features_not_scaled_count =
        std::count(features_not_scaled.begin(), features_not_scaled.end(), ',');
    // + 1 because last one does not have ',' after it.
    const int features_scaled_count =
        std::count(features_scaled.begin(), features_scaled.end(), ',') + 1;

    // + 3 stands for three noisy channels.
    const int buffer_count = features_not_scaled_count + features_scaled_count + 3;

    // Create and build the kernel
    std::stringstream build_options;
    build_options <<
        " -D BUFFER_COUNT=" << buffer_count <<
        " -D FEATURES_NOT_SCALED=" << features_not_scaled_count <<
        " -D FEATURES_SCALED=" << features_scaled_count <<
        " -D IMAGE_WIDTH=" << IMAGE_WIDTH <<
        " -D IMAGE_HEIGHT=" << IMAGE_HEIGHT <<
        " -D WORKSET_WIDTH=" << WORKSET_WIDTH <<
        " -D WORKSET_HEIGHT=" << WORKSET_HEIGHT <<
        " -D FEATURE_BUFFERS=" << NOT_SCALED_FEATURE_BUFFERS SCALED_FEATURE_BUFFERS <<
        " -D LOCAL_WIDTH=" << LOCAL_WIDTH <<
        " -D LOCAL_HEIGHT=" << LOCAL_HEIGHT <<
        " -D WORKSET_WITH_MARGINS_WIDTH=" << WORKSET_WITH_MARGINS_WIDTH <<
        " -D WORKSET_WITH_MARGINS_HEIGHT=" << WORKSET_WITH_MARGINS_HEIGHT <<
        " -D BLOCK_EDGE_LENGTH=" << STR(BLOCK_EDGE_LENGTH) <<
        " -D BLOCK_PIXELS=" << BLOCK_PIXELS <<
        " -D R_EDGE=" << buffer_count - 2 <<
        " -D NOISE_AMOUNT=" << STR(NOISE_AMOUNT) <<
        " -D BLEND_ALPHA=" << STR(BLEND_ALPHA) <<
        " -D SECOND_BLEND_ALPHA=" << STR(SECOND_BLEND_ALPHA) <<
        " -D TAA_BLEND_ALPHA=" << STR(TAA_BLEND_ALPHA) <<
        " -D POSITION_LIMIT_SQUARED=" << position_limit_squared <<
        " -D NORMAL_LIMIT_SQUARED=" << normal_limit_squared <<
        " -D COMPRESSED_R=" << STR(COMPRESSED_R) <<
        " -D CACHE_TMP_DATA=" << STR(CACHE_TMP_DATA) <<
        " -D ADD_REQD_WG_SIZE=" << STR(ADD_REQD_WG_SIZE) <<
        " -D LOCAL_SIZE=" << STR(LOCAL_SIZE) <<
        " -D USE_HALF_PRECISION_IN_TMP_DATA=" << STR(USE_HALF_PRECISION_IN_TMP_DATA);

    cl::Kernel &fitter_kernel(clEnv.addProgram(0, "bmfr.cl", "fitter",
        build_options.str().c_str()));
    cl::Kernel &weighted_sum_kernel(clEnv.addProgram(0, "bmfr.cl", "weighted_sum",
        build_options.str().c_str()));
    cl::Kernel &accum_noisy_kernel(clEnv.addProgram(0, "bmfr.cl", "accumulate_noisy_data",
        build_options.str().c_str()));
    cl::Kernel &accum_filtered_kernel(clEnv.addProgram(0, "bmfr.cl", 
        "accumulate_filtered_data", build_options.str().c_str()));
    cl::Kernel &taa_kernel(clEnv.addProgram(0, "bmfr.cl", "taa",
        build_options.str().c_str()));

    cl::NDRange accum_global(WORKSET_WITH_MARGINS_WIDTH, WORKSET_WITH_MARGINS_HEIGHT);
    cl::NDRange output_global(WORKSET_WIDTH, WORKSET_HEIGHT);
    cl::NDRange local(LOCAL_WIDTH, LOCAL_HEIGHT);
    cl::NDRange fitter_global(FITTER_GLOBAL);
    cl::NDRange fitter_local(LOCAL_SIZE);

    // Data arrays
    printf("Loading input data.\n");
    std::vector<cl_float> out_data[FRAME_COUNT];
    std::vector<cl_float> albedos[FRAME_COUNT];
    std::vector<cl_float> normals[FRAME_COUNT];
    std::vector<cl_float> positions[FRAME_COUNT];
    std::vector<cl_float> noisy_input[FRAME_COUNT];
    bool error = false;
#pragma omp parallel for
    for (int frame = 0; frame < FRAME_COUNT; ++frame)
    {
        if (error)
            continue;

        out_data[frame].resize(3 * OUTPUT_SIZE);

        albedos[frame].resize(3 * IMAGE_WIDTH * IMAGE_HEIGHT);
        Operation_result result = load_image(albedos[frame].data(), ALBEDO_FILE_NAME,
            frame);
        if (!result.success)
        {
            error = true;
            printf("Albedo buffer loading failed, reason: %s\n",
                   result.error_message.c_str());
            continue;
        }

        normals[frame].resize(3 * IMAGE_WIDTH * IMAGE_HEIGHT);
        result = load_image(normals[frame].data(), NORMAL_FILE_NAME, frame);
        if (!result.success)
        {
            error = true;
            printf("Normal buffer loading failed, reason: %s\n",
                   result.error_message.c_str());
            continue;
        }

        positions[frame].resize(3 * IMAGE_WIDTH * IMAGE_HEIGHT);
        result = load_image(positions[frame].data(), POSITION_FILE_NAME, frame);
        if (!result.success)
        {
            error = true;
            printf("Position buffer loading failed, reason: %s\n",
                   result.error_message.c_str());
            continue;
        }

        noisy_input[frame].resize(3 * IMAGE_WIDTH * IMAGE_HEIGHT);
        result = load_image(noisy_input[frame].data(), NOISY_FILE_NAME, frame);
        if (!result.success)
        {
            error = true;
            printf("Position buffer loading failed, reason: %s\n",
                   result.error_message.c_str());
            continue;
        }
    }

    if (error)
    {
        printf("One or more errors occurred during buffer loading\n");
        return 1;
    }

    // Create buffers
    Double_buffer<cl::Buffer> normals_buffer(context,
                                             CL_MEM_READ_WRITE, OUTPUT_SIZE * 3 * sizeof(cl_float));
    Double_buffer<cl::Buffer> positions_buffer(context,
                                               CL_MEM_READ_WRITE, OUTPUT_SIZE * 3 * sizeof(cl_float));
    Double_buffer<cl::Buffer> noisy_buffer(context,
                                           CL_MEM_READ_WRITE, OUTPUT_SIZE * 3 * sizeof(cl_float));
    size_t in_buffer_data_size = USE_HALF_PRECISION_IN_TMP_DATA ? sizeof(cl_half) : sizeof(cl_float);
    cl::Buffer in_buffer(context, CL_MEM_READ_WRITE, WORKSET_WITH_MARGINS_WIDTH * WORKSET_WITH_MARGINS_HEIGHT *
                         buffer_count * in_buffer_data_size, nullptr);
    cl::Buffer filtered_buffer(context, CL_MEM_READ_WRITE,
                               OUTPUT_SIZE * 3 * sizeof(cl_float));
    Double_buffer<cl::Buffer> out_buffer(context, CL_MEM_READ_WRITE,
                                         WORKSET_WITH_MARGINS_WIDTH * WORKSET_WITH_MARGINS_HEIGHT * 3 * sizeof(cl_float));
    Double_buffer<cl::Buffer> result_buffer(context, CL_MEM_READ_WRITE,
                                            OUTPUT_SIZE * 3 * sizeof(cl_float));
    cl::Buffer prev_pixels_buffer(context, CL_MEM_READ_WRITE,
                                  OUTPUT_SIZE * sizeof(cl_float2));
    cl::Buffer accept_buffer(context, CL_MEM_READ_WRITE, OUTPUT_SIZE * sizeof(cl_uchar));
    cl::Buffer albedo_buffer(context, CL_MEM_READ_ONLY,
                             IMAGE_WIDTH * IMAGE_HEIGHT * 3 * sizeof(cl_float));
    cl::Buffer tone_mapped_buffer(context, CL_MEM_READ_WRITE,
                                  IMAGE_WIDTH * IMAGE_HEIGHT * 3 * sizeof(cl_float));
    cl::Buffer weights_buffer(context, CL_MEM_READ_WRITE,
                              (FITTER_GLOBAL / 256) * (buffer_count - 3) * 3 * sizeof(cl_float));
    cl::Buffer mins_maxs_buffer(context, CL_MEM_READ_WRITE,
                                (FITTER_GLOBAL / 256) * 6 * sizeof(cl_float2));
    Double_buffer<cl::Buffer> spp_buffer(context, CL_MEM_READ_WRITE,
                                         OUTPUT_SIZE * sizeof(cl_char));

    std::vector<Double_buffer<cl::Buffer> *> all_double_buffers = {
        &normals_buffer, &positions_buffer, &noisy_buffer,
        &out_buffer, &result_buffer, &spp_buffer};

    // Set kernel arguments
    int arg_index = 0;
    accum_noisy_kernel.setArg(arg_index++, prev_pixels_buffer);
    accum_noisy_kernel.setArg(arg_index++, accept_buffer);

    arg_index = 0;
#if COMPRESSED_R
    const int r_size = ((buffer_count - 2) *
                        (buffer_count - 1) / 2) *
                       sizeof(cl_float3);
#else
    const int r_size = (buffer_count - 2) *
                       (buffer_count - 2) * sizeof(cl_float3);
#endif
    fitter_kernel.setArg(arg_index++, LOCAL_SIZE * sizeof(float), nullptr);
    fitter_kernel.setArg(arg_index++, BLOCK_PIXELS * sizeof(float), nullptr);
    fitter_kernel.setArg(arg_index++, r_size, nullptr);
    fitter_kernel.setArg(arg_index++, weights_buffer);
    fitter_kernel.setArg(arg_index++, mins_maxs_buffer);

    arg_index = 0;
    weighted_sum_kernel.setArg(arg_index++, weights_buffer);
    weighted_sum_kernel.setArg(arg_index++, mins_maxs_buffer);
    weighted_sum_kernel.setArg(arg_index++, filtered_buffer);

    arg_index = 0;
    accum_filtered_kernel.setArg(arg_index++, filtered_buffer);
    accum_filtered_kernel.setArg(arg_index++, prev_pixels_buffer);
    accum_filtered_kernel.setArg(arg_index++, accept_buffer);
    accum_filtered_kernel.setArg(arg_index++, albedo_buffer);
    accum_filtered_kernel.setArg(arg_index++, tone_mapped_buffer);

    arg_index = 0;
    taa_kernel.setArg(arg_index++, prev_pixels_buffer);
    taa_kernel.setArg(arg_index++, tone_mapped_buffer);
    queue.finish();

    std::vector<clutils::GPUTimer<std::milli>> accum_noisy_timer;
    std::vector<clutils::GPUTimer<std::milli>> copy_timer;
    std::vector<clutils::GPUTimer<std::milli>> fitter_timer;
    std::vector<clutils::GPUTimer<std::milli>> weighted_sum_timer;
    std::vector<clutils::GPUTimer<std::milli>> accum_filtered_timer;
    std::vector<clutils::GPUTimer<std::milli>> taa_timer;
    accum_noisy_timer.assign(FRAME_COUNT - 1, clutils::GPUTimer<std::milli>(clEnv.devices[0][0]));
    copy_timer.assign(FRAME_COUNT, clutils::GPUTimer<std::milli>(clEnv.devices[0][0]));
    fitter_timer.assign(FRAME_COUNT, clutils::GPUTimer<std::milli>(clEnv.devices[0][0]));
    weighted_sum_timer.assign(FRAME_COUNT, clutils::GPUTimer<std::milli>(clEnv.devices[0][0]));
    accum_filtered_timer.assign(FRAME_COUNT - 1, clutils::GPUTimer<std::milli>(clEnv.devices[0][0]));
    taa_timer.assign(FRAME_COUNT - 1, clutils::GPUTimer<std::milli>(clEnv.devices[0][0]));

    clutils::ProfilingInfo<FRAME_COUNT - 1> profile_info_accum_noisy(
        "Accumulation of noisy data");
    clutils::ProfilingInfo<FRAME_COUNT> profile_info_copy(
        "Copy input buffer");
    clutils::ProfilingInfo<FRAME_COUNT> profile_info_fitter(
        "Fitting feature buffers to noisy data");
    clutils::ProfilingInfo<FRAME_COUNT> profile_info_weighted_sum(
        "Weighted sum");
    clutils::ProfilingInfo<FRAME_COUNT - 1> profile_info_accum_filtered(
        "Accumulation of filtered data");
    clutils::ProfilingInfo<FRAME_COUNT - 1> profile_info_taa(
        "TAA");
    clutils::ProfilingInfo<FRAME_COUNT - 1> profile_info_total(
        "Total time in all kernels (including intermediate launch overheads)");

    printf("Run and profile kernels.\n");
    // Note: in real use case there would not be WriteBuffer and ReadBuffer function calls
    // because the input data comes from the path tracer and output goes to the screen
    for (int frame = 0; frame < FRAME_COUNT; ++frame)
    {

        queue.enqueueWriteBuffer(albedo_buffer, true, 0, IMAGE_WIDTH * IMAGE_HEIGHT * 3 *
                                 sizeof(cl_float), albedos[frame].data());
        queue.enqueueWriteBuffer(*normals_buffer.current(), true, 0, IMAGE_WIDTH *
                                 IMAGE_HEIGHT * 3 * sizeof(cl_float), normals[frame].data());
        queue.enqueueWriteBuffer(*positions_buffer.current(), true, 0, IMAGE_WIDTH *
                                 IMAGE_HEIGHT * 3 * sizeof(cl_float), positions[frame].data());
        queue.enqueueWriteBuffer(*noisy_buffer.current(), true, 0, IMAGE_WIDTH * IMAGE_HEIGHT *
                                 3 * sizeof(cl_float), noisy_input[frame].data());

        // On the first frame accum_noisy_kernel just copies to the in_buffer
        arg_index = 2;
        accum_noisy_kernel.setArg(arg_index++, *normals_buffer.current());
        accum_noisy_kernel.setArg(arg_index++, *normals_buffer.previous());
        accum_noisy_kernel.setArg(arg_index++, *positions_buffer.current());
        accum_noisy_kernel.setArg(arg_index++, *positions_buffer.previous());
        accum_noisy_kernel.setArg(arg_index++, *noisy_buffer.current());
        accum_noisy_kernel.setArg(arg_index++, *noisy_buffer.previous());
        accum_noisy_kernel.setArg(arg_index++, *spp_buffer.previous());
        accum_noisy_kernel.setArg(arg_index++, *spp_buffer.current());
        accum_noisy_kernel.setArg(arg_index++, in_buffer);
        const int matrix_index = frame == 0 ? 0 : frame - 1;
        accum_noisy_kernel.setArg(arg_index++, sizeof(cl_float16),
                                  &(camera_matrices[matrix_index][0][0]));
        accum_noisy_kernel.setArg(arg_index++, sizeof(cl_float2),
                                  &(pixel_offsets[frame][0]));
        accum_noisy_kernel.setArg(arg_index++, sizeof(cl_int), &frame);
        queue.enqueueNDRangeKernel(accum_noisy_kernel, cl::NullRange, accum_global, local,
                                   nullptr, &accum_noisy_timer[matrix_index].event());

        arg_index = 5;
        fitter_kernel.setArg(arg_index++, in_buffer);
        fitter_kernel.setArg(arg_index++, sizeof(cl_int), &frame);
        queue.enqueueNDRangeKernel(fitter_kernel, cl::NullRange, fitter_global,
                                   fitter_local, nullptr, &fitter_timer[frame].event());

        arg_index = 3;
        weighted_sum_kernel.setArg(arg_index++, *normals_buffer.current());
        weighted_sum_kernel.setArg(arg_index++, *positions_buffer.current());
        weighted_sum_kernel.setArg(arg_index++, *noisy_buffer.current());
        weighted_sum_kernel.setArg(arg_index++, sizeof(cl_int), &frame);
        queue.enqueueNDRangeKernel(weighted_sum_kernel, cl::NullRange, output_global,
                                   local, nullptr, &weighted_sum_timer[frame].event());

        arg_index = 5;
        accum_filtered_kernel.setArg(arg_index++, *spp_buffer.current());
        accum_filtered_kernel.setArg(arg_index++, *out_buffer.previous());
        accum_filtered_kernel.setArg(arg_index++, *out_buffer.current());
        accum_filtered_kernel.setArg(arg_index++, sizeof(cl_int), &frame);
        queue.enqueueNDRangeKernel(accum_filtered_kernel, cl::NullRange, output_global,
                                   local, nullptr, &accum_filtered_timer[matrix_index].event());

        arg_index = 2;
        taa_kernel.setArg(arg_index++, *result_buffer.current());
        taa_kernel.setArg(arg_index++, *result_buffer.previous());
        taa_kernel.setArg(arg_index++, sizeof(cl_int), &frame);
        queue.enqueueNDRangeKernel(taa_kernel, cl::NullRange, output_global, local,
                                   nullptr, &taa_timer[matrix_index].event());

        // This is not timed because in real use case the result is stored to frame buffer
        queue.enqueueReadBuffer(*result_buffer.current(), false, 0,
                                OUTPUT_SIZE * 3 * sizeof(cl_float), out_data[frame].data());

        // Swap all double buffers
        std::for_each(all_double_buffers.begin(), all_double_buffers.end(),
                      std::bind(&Double_buffer<cl::Buffer>::swap, std::placeholders::_1));
    }
    queue.finish();

    // Store profiling data
    for (int i = 0; i < FRAME_COUNT; ++i)
    {
        if (i > 0)
        {
            profile_info_accum_noisy[i - 1] = accum_noisy_timer[i - 1].duration();
            profile_info_accum_filtered[i - 1] = accum_filtered_timer[i - 1].duration();
            profile_info_taa[i - 1] = taa_timer[i - 1].duration();

            cl_ulong total_start =
                accum_noisy_timer[i - 1].event().getProfilingInfo<CL_PROFILING_COMMAND_START>();
            cl_ulong total_end =
                taa_timer[i - 1].event().getProfilingInfo<CL_PROFILING_COMMAND_END>();
            profile_info_total[i - 1] =
                (total_end - total_start) * taa_timer[i - 1].getUnit();
        }
        profile_info_fitter[i] = fitter_timer[i].duration();
        profile_info_weighted_sum[i] = weighted_sum_timer[i].duration();
    }

    if (FRAME_COUNT > 1)
        profile_info_accum_noisy.print();
    profile_info_fitter.print();
    profile_info_weighted_sum.print();
    if (FRAME_COUNT > 1)
    {
        profile_info_accum_filtered.print();
        profile_info_taa.print();
        profile_info_total.print();
    }

    // Store results
    error = false;
#pragma omp parallel for
    for (int frame = 0; frame < FRAME_COUNT; ++frame)
    {
        if (error)
            continue;

        // Output image
        std::string output_file_name = OUTPUT_FILE_NAME + std::to_string(frame) + ".png";
        // Crops back from WORKSET_SIZE to IMAGE_SIZE
        OpenImageIO::ImageSpec spec(IMAGE_WIDTH, IMAGE_HEIGHT, 3,
                                    OpenImageIO::TypeDesc::FLOAT);
        std::unique_ptr<OpenImageIO::ImageOutput>
            out(OpenImageIO::ImageOutput::create(output_file_name));
        if (out && out->open(output_file_name, spec))
        {
            out->write_image(OpenImageIO::TypeDesc::FLOAT, out_data[frame].data(),
                             3 * sizeof(cl_float), WORKSET_WIDTH * 3 * sizeof(cl_float), 0);
            out->close();
        }
        else
        {
            printf("Can't create image file on disk to location %s\n",
                   output_file_name.c_str());
            error = true;
            continue;
        }
    }

    if (error)
    {
        printf("One or more errors occurred during image saving\n");
        return 1;
    }

    return 0;
}

int main()
{
    try
    {
        return tasks();
    }
    catch (std::exception &err)
    {
        printf("Exception: %s", err.what());
        std::exception *err_ptr = &err;
        cl::Error *cl_err = dynamic_cast<cl::Error *>(err_ptr);
        if (cl_err != nullptr)
        {
            printf(" call with error code %i = %s\n",
                   cl_err->err(), clutils::getOpenCLErrorCodeString(cl_err->err()));
            return cl_err->err();
        }
        printf("\n");
        return 1;
    }
}
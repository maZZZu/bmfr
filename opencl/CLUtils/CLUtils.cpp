/*! \file CLUtils.cpp
 *  \brief Definitions of functions and methods for the CLUtils library
 *  \details CLUtils offers utilities that help 
 *           setup and manage an OpenCL environment.
 *  \author Nick Lamprianidis
 *  \version 0.2.2
 *  \date 2014-2015
 *  \copyright The MIT License (MIT)
 *  \par
 *  Copyright (c) 2014 Nick Lamprianidis
 *  \par
 *  Permission is hereby granted, free of charge, to any person obtaining a copy
 *  of this software and associated documentation files (the "Software"), to deal
 *  in the Software without restriction, including without limitation the rights
 *  to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 *  copies of the Software, and to permit persons to whom the Software is
 *  furnished to do so, subject to the following conditions:
 *  \par
 *  The above copyright notice and this permission notice shall be included in
 *  all copies or substantial portions of the Software.
 *  \par
 *  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 *  IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 *  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 *  AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 *  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 *  OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 *  THE SOFTWARE.
 */

#include "CLUtils.hpp"
#include <vector>
#include <algorithm>
#include <iostream>
#include <fstream>
#include <sstream>
#include <chrono>

#if defined(_WIN32)
#include <windows.h>
#elif defined(__linux__)
#include <GL/glx.h>
#elif defined(__APPLE__) || defined(__MACOSX)
#include <OpenGL/OpenGL.h>
#endif


namespace clutils
{

    /*! Gets an error code and returns its name.
     *
     *  \param[in] errorCode an error code.
     *  \return Its name as a char array.
     */
    const char* getOpenCLErrorCodeString (int errorCode)
    {
        switch (errorCode)
        {
            case CL_SUCCESS:
                return "CL_SUCCESS";
            case CL_DEVICE_NOT_FOUND:
                return "CL_DEVICE_NOT_FOUND";
            case CL_DEVICE_NOT_AVAILABLE:
                return "CL_DEVICE_NOT_AVAILABLE";
            case CL_COMPILER_NOT_AVAILABLE:
                return "CL_COMPILER_NOT_AVAILABLE";
            case CL_MEM_OBJECT_ALLOCATION_FAILURE:
                return "CL_MEM_OBJECT_ALLOCATION_FAILURE";
            case CL_OUT_OF_RESOURCES:
                return "CL_OUT_OF_RESOURCES";
            case CL_OUT_OF_HOST_MEMORY:
                return "CL_OUT_OF_HOST_MEMORY";
            case CL_PROFILING_INFO_NOT_AVAILABLE:
                return "CL_PROFILING_INFO_NOT_AVAILABLE";
            case CL_MEM_COPY_OVERLAP:
                return "CL_MEM_COPY_OVERLAP";
            case CL_IMAGE_FORMAT_MISMATCH:
                return "CL_IMAGE_FORMAT_MISMATCH";
            case CL_IMAGE_FORMAT_NOT_SUPPORTED:
                return "CL_IMAGE_FORMAT_NOT_SUPPORTED";
            case CL_BUILD_PROGRAM_FAILURE:
                return "CL_BUILD_PROGRAM_FAILURE";
            case CL_MAP_FAILURE:
                return "CL_MAP_FAILURE";
            case CL_MISALIGNED_SUB_BUFFER_OFFSET:
                return "CL_MISALIGNED_SUB_BUFFER_OFFSET";
            case CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST:
                return "CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST";
            case CL_COMPILE_PROGRAM_FAILURE:
                return "CL_COMPILE_PROGRAM_FAILURE";
            case CL_LINKER_NOT_AVAILABLE:
                return "CL_LINKER_NOT_AVAILABLE";
            case CL_LINK_PROGRAM_FAILURE:
                return "CL_LINK_PROGRAM_FAILURE";
            case CL_DEVICE_PARTITION_FAILED:
                return "CL_DEVICE_PARTITION_FAILED";
            case CL_KERNEL_ARG_INFO_NOT_AVAILABLE:
                return "CL_KERNEL_ARG_INFO_NOT_AVAILABLE";
            case CL_INVALID_VALUE:
                return "CL_INVALID_VALUE";
            case CL_INVALID_DEVICE_TYPE:
                return "CL_INVALID_DEVICE_TYPE";
            case CL_INVALID_PLATFORM:
                return "CL_INVALID_PLATFORM";
            case CL_INVALID_DEVICE:
                return "CL_INVALID_DEVICE";
            case CL_INVALID_CONTEXT:
                return "CL_INVALID_CONTEXT";
            case CL_INVALID_QUEUE_PROPERTIES:
                return "CL_INVALID_QUEUE_PROPERTIES";
            case CL_INVALID_COMMAND_QUEUE:
                return "CL_INVALID_COMMAND_QUEUE";
            case CL_INVALID_HOST_PTR:
                return "CL_INVALID_HOST_PTR";
            case CL_INVALID_MEM_OBJECT:
                return "CL_INVALID_MEM_OBJECT";
            case CL_INVALID_IMAGE_FORMAT_DESCRIPTOR:
                return "CL_INVALID_IMAGE_FORMAT_DESCRIPTOR";
            case CL_INVALID_IMAGE_SIZE:
                return "CL_INVALID_IMAGE_SIZE";
            case CL_INVALID_SAMPLER:
                return "CL_INVALID_SAMPLER";
            case CL_INVALID_BINARY:
                return "CL_INVALID_BINARY";
            case CL_INVALID_BUILD_OPTIONS:
                return "CL_INVALID_BUILD_OPTIONS";
            case CL_INVALID_PROGRAM:
                return "CL_INVALID_PROGRAM";
            case CL_INVALID_PROGRAM_EXECUTABLE:
                return "CL_INVALID_PROGRAM_EXECUTABLE";
            case CL_INVALID_KERNEL_NAME:
                return "CL_INVALID_KERNEL_NAME";
            case CL_INVALID_KERNEL_DEFINITION:
                return "CL_INVALID_KERNEL_DEFINITION";
            case CL_INVALID_KERNEL:
                return "CL_INVALID_KERNEL";
            case CL_INVALID_ARG_INDEX:
                return "CL_INVALID_ARG_INDEX";
            case CL_INVALID_ARG_VALUE:
                return "CL_INVALID_ARG_VALUE";
            case CL_INVALID_ARG_SIZE:
                return "CL_INVALID_ARG_SIZE";
            case CL_INVALID_KERNEL_ARGS:
                return "CL_INVALID_KERNEL_ARGS";
            case CL_INVALID_WORK_DIMENSION:
                return "CL_INVALID_WORK_DIMENSION";
            case CL_INVALID_WORK_GROUP_SIZE:
                return "CL_INVALID_WORK_GROUP_SIZE";
            case CL_INVALID_WORK_ITEM_SIZE:
                return "CL_INVALID_WORK_ITEM_SIZE";
            case CL_INVALID_GLOBAL_OFFSET:
                return "CL_INVALID_GLOBAL_OFFSET";
            case CL_INVALID_EVENT_WAIT_LIST:
                return "CL_INVALID_EVENT_WAIT_LIST";
            case CL_INVALID_EVENT:
                return "CL_INVALID_EVENT";
            case CL_INVALID_OPERATION:
                return "CL_INVALID_OPERATION";
            case CL_INVALID_GL_OBJECT:
                return "CL_INVALID_GL_OBJECT";
            case CL_INVALID_BUFFER_SIZE:
                return "CL_INVALID_BUFFER_SIZE";
            case CL_INVALID_MIP_LEVEL:
                return "CL_INVALID_MIP_LEVEL";
            case CL_INVALID_GLOBAL_WORK_SIZE:
                return "CL_INVALID_GLOBAL_WORK_SIZE";
            case CL_INVALID_PROPERTY:
                return "CL_INVALID_PROPERTY";
            case CL_INVALID_IMAGE_DESCRIPTOR:
                return "CL_INVALID_IMAGE_DESCRIPTOR";
            case CL_INVALID_COMPILER_OPTIONS:
                return "CL_INVALID_COMPILER_OPTIONS";
            case CL_INVALID_LINKER_OPTIONS:
                return "CL_INVALID_LINKER_OPTIONS";
            case CL_INVALID_DEVICE_PARTITION_COUNT:
                return "CL_INVALID_DEVICE_PARTITION_COUNT";
            default:
                return "UNKNOWN_ERROR_CODE";
        }
    }


    /*! \param[in] device a device for which to check the "GL Sharing" capability.
     *  \return Returns true if "GL Sharing" is available, false otherwise.
     */
    bool checkCLGLInterop (cl::Device &device)
    {
        std::string exts = device.getInfo<CL_DEVICE_EXTENSIONS> ();

        #if defined(__APPLE__) || defined(__MACOSX)
        std::string glShare("cl_apple_gl_sharing");
        #else
        std::string glShare("cl_khr_gl_sharing");
        #endif

        return exts.find (glShare) != std::string::npos;
    }


    /*! \param[in] kernel_filenames a vector of strings with 
     *                              the names of the kernel files (.cl).
     *  \param[out] sourceCodes a vector of strings with the contents of the files.
     */
    void readSource (const std::vector<std::string> &kernel_filenames, 
                     std::vector<std::string> &sourceCodes)
    {
        std::ifstream programSource;

        try
        {
            for (auto &fName : kernel_filenames)
            {
                programSource.exceptions (std::ifstream::failbit | std::ifstream::badbit);
                programSource.open (fName);
                sourceCodes.emplace_back (std::istreambuf_iterator<char> (programSource), 
                                         (std::istreambuf_iterator<char> ()));
                programSource.close ();
            }
        }
        catch (std::ifstream::failure &error)
        {
            std::cerr << "Error when accessing kernel file: " << error.what () 
                      << " (" << __FILE__ << ":" << __LINE__ << ")" << std::endl;
            exit (EXIT_FAILURE);
        }
    }


    /*! \param[in] str string containing the tokens.
     *  \param[in] delim delimiter on which to split the string.
     *  \param[out] names a vector of all the tokens.
     */
    void split (const std::string &str, char delim, 
                std::vector<std::string> &names)
    {
        std::stringstream ss (str);
        std::string item;
        while (std::getline (ss, item, delim))
            names.push_back (item);
    }

    /*! It initializes the OpenCL environment. If a `kernel_filenames` argument 
     *  is provided, it creates a context for all the devices in the first 
     *  platform, and a command queue for the first device in that platform. 
     *  It also builds a program object from all the requested kernel files, 
     *  and extracts all kernels in that program.
     *
     *  \param[in] kernel_filenames a vector of strings with 
     *                              the names of the kernel files (.cl).
     *  \param[in] build_options options that are forwarded to the OpenCL compiler.
     */
    CLEnv::CLEnv (const std::vector<std::string> &kernel_filenames, 
                  const char *build_options)
    {
        // Get the list of platforms
        cl::Platform::get (&platforms);

        if (!kernel_filenames.empty ())
        {
            // Get the list of devices in platform 0
            devices.emplace_back ();
            platforms[0].getDevices (CL_DEVICE_TYPE_ALL, &devices[0]);

            // Create a context for those devices (in platform 0)
            contexts.emplace_back (devices[0]);

            // Create a command queue for device 0 in platform 0
            queues.emplace_back ();
            queues[0].emplace_back (contexts[0], devices[0][0]);

            // Create a sources object with all kernel files
            cl::Program::Sources sources;
            readSource (kernel_filenames, sources);

            // Create a program object from the source codes, targeting context 0
            programs.emplace_back (contexts[0], sources);

            try
            {
                // Build the program for all devices in context 0 (platform 0)
                programs[0].build (devices[0], build_options);
            }
            catch (const cl::Error &error)
            {
                std::cerr << error.what ()
                          << " (" << clutils::getOpenCLErrorCodeString (error.err ()) 
                          << ")"  << std::endl << std::endl;
                
                std::string log = programs[0].getBuildInfo<CL_PROGRAM_BUILD_LOG> (devices[0][0]);
                std::cout << log << std::endl;

                exit (EXIT_FAILURE);
            }

            // Get the kernel names
            // Note: getInfo returns a ';' delimited string.
            std::string namesString = programs[0].getInfo<CL_PROGRAM_KERNEL_NAMES> ();
            std::vector<std::string> kernel_names;
            clutils::split (namesString, ';', kernel_names);
            
            // Retrieve the kernels from program 0
            kernels.emplace_back ();
            kernelIdx.emplace_back ();
            for (unsigned int idx = 0; idx < kernel_names.size (); ++idx)
            {
                kernels[0].emplace_back (programs[0], kernel_names[idx].c_str ());
                kernelIdx[0][kernel_names[idx]] = idx;
            }
        }
    }


    /*! \brief Delegating constructor
     *
     *  \param[in] kernel_filename a string with the name of the kernel file (.cl).
     *  \param[in] build_options options that are forwarded to the OpenCL compiler.
     */
    CLEnv::CLEnv (const std::string &kernel_filename, const char *build_options)
        : CLEnv (std::vector<std::string> { kernel_filename }, build_options)
    {
    }


    /*! \param[in] pIdx an index for the context. 
     *                  Indices follow the order the contexts were created in.
     *  \return The requested context.
     */
    cl::Context& CLEnv::getContext (unsigned int pIdx)
    {
        try
        {
            return contexts.at (pIdx);
        }
        catch (const std::out_of_range &error)
        {
            std::cerr << "Out of Range error: " << error.what () 
                      << " (" << __FILE__ << ":" << __LINE__ << ")" << std::endl;
            exit (EXIT_FAILURE);
        }
    }


    /*! \param[in] ctxIdx the index for the context the requested queue is in.
     *                    Indices follow the order the contexts were created in.
     *  \param[in] qIdx an index for the command queue. 
     *                  Indices follow the order the queues were created in.
     *  \return The requested command queue.
     */
    cl::CommandQueue& CLEnv::getQueue (unsigned int ctxIdx, unsigned int qIdx)
    {
        try
        {
            return queues.at (ctxIdx).at (qIdx);
        }
        catch (const std::out_of_range &error)
        {
            std::cerr << "Out of Range error: " << error.what () 
                      << " (" << __FILE__ << ":" << __LINE__ << ")" << std::endl;
            exit (EXIT_FAILURE);
        }
    }


    /*! \param[in] pgIdx the index of the requested program object. 
     *                   Indices follow the order the programs were created in.
     *  \return The requested program.
     */
    cl::Program& CLEnv::getProgram (unsigned int pgIdx)
    {
        try
        {
            return programs.at (pgIdx);
        }
        catch (const std::out_of_range &error)
        {
            std::cerr << "Out of Range error: " << error.what () 
                      << " (" << __FILE__ << ":" << __LINE__ << ")" << std::endl;
            exit (EXIT_FAILURE);
        }
    }


    /*! \param[in] kernel_name the name of the kernel.
     *  \param[in] pgIdx the index of the program the kernel belongs to. 
     *                   Indices follow the order the programs were created in.
     *  \return The requested kernel.
     */
    cl::Kernel& CLEnv::getKernel (const char *kernel_name, size_t pgIdx)
    {
        try
        {
            /*! \sa kernelIdx */
            unsigned int kIdx = kernelIdx.at (pgIdx).at (std::string (kernel_name));
            return kernels[pgIdx][kIdx];
        }
        catch (const std::out_of_range &error)
        {
            std::cerr << "Out of Range error: " << error.what () 
                      << " (" << __FILE__ << ":" << __LINE__ << ")" << std::endl;
            exit (EXIT_FAILURE);
        }
    }


    /*! \details It allows to create a GL-shared context. If GL-Sharing is not 
     *           supported for the associated device, an exception is thrown.
     *           It also calls `initGLMemObjects` to initialize the GL buffers.
     *  \param[in] pIdx an index for the platform for which to create the context. 
     *                  Indices follow the order the platforms got returned in 
     *                  by the OpenCL runtime.
     *  \param[in] gl_shared a flag for whether or not to create a GL-shared CL context.
     *  \return A reference to the created context.
     */
    cl::Context& CLEnv::addContext (unsigned int pIdx, const bool gl_shared)
    {
        try
        {
            size_t idx = devices.size ();
            devices.emplace_back ();
            platforms.at (pIdx).getDevices (CL_DEVICE_TYPE_ALL, &devices[idx]);
            cl_context_properties *props = nullptr;

            if (gl_shared)
            {
                // Set context properties (OS specific)
                #if defined(_WIN32)
                cl_context_properties _props[] = 
                {
                    CL_GL_CONTEXT_KHR, (cl_context_properties) wglGetCurrentContext (),
                    CL_WGL_HDC_KHR, (cl_context_properties) wglGetCurrentDC (),
                    CL_CONTEXT_PLATFORM, (cl_context_properties) (platforms.at (pIdx)) (),
                    0 
                };
                #elif defined(__APPLE__) || defined(__MACOSX)
                CGLContextObj kCGLContext = CGLGetCurrentContext ();
                CGLShareGroupObj kCGLShareGroup = CGLGetShareGroup (kCGLContext);
                cl_context_properties _props[] = 
                {
                    CL_CONTEXT_PROPERTY_USE_CGL_SHAREGROUP_APPLE, (cl_context_properties) kCGLShareGroup, 
                    0 
                };
                #else
                cl_context_properties _props[] = 
                {
                    CL_GL_CONTEXT_KHR, (cl_context_properties) glXGetCurrentContext (),
                    CL_GLX_DISPLAY_KHR, (cl_context_properties) glXGetCurrentDisplay (),
                    CL_CONTEXT_PLATFORM, (cl_context_properties) (platforms.at (pIdx)) (),
                    0 
                };
                #endif

                // Get the CL device associated with the GL context
                #if defined(__APPLE__) || defined(__MACOSX)
                cl::Device device = devices[idx][0];
                #else
                cl_device_id devID;
                clGetGLContextInfoKHR_fn clGetGLContextInfo = (clGetGLContextInfoKHR_fn) 
                    clGetExtensionFunctionAddressForPlatform ((platforms.at (pIdx)) (), "clGetGLContextInfoKHR");
                clGetGLContextInfo (_props, CL_CURRENT_DEVICE_FOR_GL_CONTEXT_KHR, sizeof (cl_device_id), &devID, nullptr);
                cl::Device device (devID);
                #endif

                if (!checkCLGLInterop (device))
                    throw cl::Error (CL_INVALID_DEVICE, "CLEnv::addContext");

                props = _props;
            }
            
            contexts.emplace_back (devices[idx], props);
            // Initialize the vector for the queues 
            // that will be handled by this context
            queues.emplace_back ();

            initGLMemObjects ();

            return contexts[idx];
        }
        catch (const std::out_of_range &error)
        {
            std::cerr << "Out of Range error: " << error.what () 
                      << " (" << __FILE__ << ":" << __LINE__ << ")" << std::endl;
            exit (EXIT_FAILURE);
        }
    }


    /*! \param[in] ctxIdx the index of the context the device is handled by. 
     *                    Indices follow the order the contexts were created in.
     *  \param[in] dIdx the index of the device among those handled by the 
     *                  specified context. Indices follow the order the devices 
     *                  got returned in by the call to getDevices 
     *                  on the proper platform.
     *  \param[in] props bitfield to enable command queue properties.
     *  \return A reference to the created queue.
     */
    cl::CommandQueue& CLEnv::addQueue (unsigned int ctxIdx, unsigned int dIdx, 
                                       cl_command_queue_properties props)
    {
        try
        {
            if (ctxIdx >= contexts.size ())
                throw std::out_of_range ("vector::_M_range_check");

            std::vector<cl::Device> devs = contexts[ctxIdx].getInfo<CL_CONTEXT_DEVICES> ();
    
            size_t qIdx = queues[ctxIdx].size ();
            queues[ctxIdx].emplace_back (contexts[ctxIdx], devs.at (dIdx), props);

            return queues[ctxIdx][qIdx];
        }
        catch (const std::out_of_range &error)
        {
            std::cerr << "Out of Range error: " << error.what () 
                      << " (" << __FILE__ << ":" << __LINE__ << ")" << std::endl;
            exit (EXIT_FAILURE);
        }
    }


    /*! \param[in] ctxIdx the index of the context the GL-shared device is handled by. 
     *                    Indices follow the order the contexts were created in.
     *  \param[in] props bitfield to enable command queue properties.
     *  \return A reference to the created queue.
     */
    cl::CommandQueue& CLEnv::addQueueGL (unsigned int ctxIdx, cl_command_queue_properties props)
    {
        try
        {
            if (ctxIdx >= contexts.size ())
                throw std::out_of_range ("vector::_M_range_check");

            #if defined(__APPLE__) || defined(__MACOSX)
            std::vector<cl::Device> devs = contexts[ctxIdx].getInfo<CL_CONTEXT_DEVICES> ();
            cl::Device device = devs[0];
            #else
            cl::Device device;
            std::vector<cl_context_properties> ctxProps = contexts[ctxIdx].getInfo<CL_CONTEXT_PROPERTIES> ();
            clGetGLContextInfoKHR_fn clGetGLContextInfo = (clGetGLContextInfoKHR_fn) 
                clGetExtensionFunctionAddressForPlatform ((cl_platform_id) ctxProps[5], "clGetGLContextInfoKHR");
            clGetGLContextInfo (ctxProps.data (), CL_CURRENT_DEVICE_FOR_GL_CONTEXT_KHR, 
                                sizeof (cl_device_id), &(device ()), nullptr);
            #endif

            size_t qIdx = queues[ctxIdx].size ();
            queues[ctxIdx].emplace_back (contexts[ctxIdx], device, props);

            return queues[ctxIdx][qIdx];
        }
        catch (const std::out_of_range &error)
        {
            std::cerr << "Out of Range error: " << error.what () 
                      << " (" << __FILE__ << ":" << __LINE__ << ")" << std::endl;
            exit (EXIT_FAILURE);
        }
    }


    /*! \param[in] ctxIdx the index of the context the program is associated with. 
     *                    Indices follow the order the contexts were created in.
     *  \param[in] kernel_filenames a vector of strings with 
     *                              the names of the kernel files (.cl).
     *  \param[in] kernel_name the name of a requested kernel.
     *  \param[in] build_options options that are forwarded to the OpenCL compiler.
     *  \return The requested kernel. If kernel_name is NULL, the first kernel 
     *          of the program gets returned.
     */
    cl::Kernel& CLEnv::addProgram (unsigned int ctxIdx, 
                                   const std::vector<std::string> &kernel_filenames, 
                                   const char *kernel_name, const char *build_options)
    {
        try
        {
            // Create a sources object with all kernel files
            cl::Program::Sources sources;
            readSource (kernel_filenames, sources);

            // Create a program object from the source codes, 
            // targeting the requested context
            programs.emplace_back (contexts.at (ctxIdx), sources);

            // Build the program for all devices in the requested context
            std::vector<cl::Device> devs = contexts[ctxIdx].getInfo<CL_CONTEXT_DEVICES> ();
            size_t pgIdx = programs.size () - 1;

            try
            {
                programs[pgIdx].build (devs, build_options);
            }
            catch (const cl::Error &error)
            {
                std::cerr << error.what ()
                          << " (" << clutils::getOpenCLErrorCodeString (error.err ()) 
                          << ")"  << std::endl << std::endl;
                
                std::string log = programs[pgIdx].getBuildInfo<CL_PROGRAM_BUILD_LOG> (devs[0]);
                std::cout << log << std::endl;

                exit (EXIT_FAILURE);
            }

            // Get the kernel names
            // Note: getInfo returns a ';' delimited string.
            std::string namesString = programs[pgIdx].getInfo<CL_PROGRAM_KERNEL_NAMES> ();
            std::vector<std::string> kernel_names;
            clutils::split (namesString, ';', kernel_names);
            
            // Retrieve the kernels from the newly created program
            kernels.emplace_back ();
            kernelIdx.emplace_back ();
            for (unsigned int idx = 0; idx < kernel_names.size (); ++idx)
            {
                kernels[pgIdx].emplace_back (programs[pgIdx], kernel_names[idx].c_str ());
                kernelIdx[pgIdx][kernel_names[idx]] = idx;
            }

            if (kernel_name == nullptr)
                return getKernel (kernel_names.at (0).c_str (), pgIdx);
            else
                return getKernel (kernel_name, pgIdx);
        }
        catch (const std::out_of_range &error)
        {
            std::cerr << "Out of Range error: " << error.what () 
                      << " (" << __FILE__ << ":" << __LINE__ << ")" << std::endl;
            exit (EXIT_FAILURE);
        }
    }


    /*! \param[in] ctxIdx the index of the context the program is associated with. 
     *                    Indices follow the order the contexts were created in.
     *  \param[in] kernel_filename a string with the name of the kernel file (.cl).
     *  \param[in] kernel_name the name of a requested kernel.
     *  \param[in] build_options options that are forwarded to the OpenCL compiler.
     *  \return The requested kernel. If kernel_name is NULL, the first kernel 
     *          of the program gets returned.
     *  \sa addProgram
     */
    cl::Kernel& CLEnv::addProgram (unsigned int ctxIdx, 
                                   const std::string &kernel_filename, 
                                   const char *kernel_name, const char *build_options)
    {
        return addProgram (ctxIdx, std::vector<std::string> { kernel_filename }, 
                           kernel_name, build_options);
    }

}

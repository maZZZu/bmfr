/*! \file CLUtils.hpp
 *  \brief Declarations of objects, 
 *         functions and classes for the CLUtils library.
 *  \details CLUtils offers utilities that help 
             setup and manage an OpenCL environment.
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

#ifndef CLUTILS_HPP
#define CLUTILS_HPP

#define CL_HPP_ENABLE_EXCEPTIONS
#define CL_HPP_TARGET_OPENCL_VERSION 120
#define CL_HPP_MINIMUM_OPENCL_VERSION 120

#if defined(__APPLE__) || defined(__MACOSX)
#include <OpenCL/cl.hpp>
#else
#include <CL/cl2.hpp>
#endif

#include <iostream>
#include <iomanip>
#include <vector>
#include <algorithm>
#include <numeric>
#include <unordered_map>
#include <chrono>
#include <cassert>
#include <cmath>

/*! \brief It brings together functionality common to all OpenCL projects.
 *  
 *  It offers structures that aim to ease the process of setting up and 
 *  maintaining an OpenCL environment.
 */
namespace clutils
{
    /*! \brief Returns the name of an error code. */
    const char* getOpenCLErrorCodeString (int errorCode);

    /*! \brief Checks the availability of the "GL Sharing" capability. */
    bool checkCLGLInterop (cl::Device &device);

    /*! \brief Reads in the contents from the requested files. */
    void readSource (const std::vector<std::string> &kernel_filenames, 
                     std::vector<std::string> &sourceCodes);

    /*! \brief Splits a string on the requested delimiter. */
    void split (const std::string &str, char delim, 
                std::vector<std::string> &names);

    /*! \brief Sets up an OpenCL environment.
     *  \details Prepares the essential OpenCL objects for the execution of 
     *           kernels. This class aims to allow rapid prototyping by hiding 
     *           away all the boilerplate code necessary for establishing 
     *           an OpenCL environment.
     */
    class CLEnv
    {
    public:
        CLEnv (const std::vector<std::string> &kernel_filenames = std::vector<std::string> (), 
               const char *build_options = nullptr);
        CLEnv (const std::string &kernel_filename, 
               const char *build_options = nullptr);
        virtual ~CLEnv () {};
        /*! \brief Gets back one of the existing contexts. */
        cl::Context& getContext (unsigned int pIdx = 0);
        /*! \brief Gets back one of the existing command queues 
         *         in the specified context. */
        cl::CommandQueue& getQueue (unsigned int ctxIdx = 0, unsigned int qIdx = 0);
        /*! \brief Gets back one of the existing programs. */
        cl::Program& getProgram (unsigned int pgIdx = 0);
        /*! \brief Gets back one of the existing kernels in some program. */
        cl::Kernel& getKernel (const char *kernelName, size_t pgIdx = 0);
        /*! \brief Creates a context for all devices in the requested platform. */
        cl::Context& addContext (unsigned int pIdx, const bool gl_shared = false);
        /*! \brief Creates a queue for the specified device in the specified context. */
        cl::CommandQueue& addQueue (unsigned int ctxIdx, unsigned int dIdx, cl_command_queue_properties props = 0);
        /*! \brief Creates a queue for the GL-shared device in the specified context. */
        cl::CommandQueue& addQueueGL (unsigned int ctxIdx, cl_command_queue_properties props = 0);
        /*! \brief Creates a program for the specified context. */
        cl::Kernel& addProgram (unsigned int ctxIdx, 
                                const std::vector<std::string> &kernel_filenames, 
                                const char *kernel_name = nullptr, 
                                const char *build_options = nullptr);
        cl::Kernel& addProgram (unsigned int ctxIdx, 
                                const std::string &kernel_filename, 
                                const char *kernel_name = nullptr, 
                                const char *build_options = nullptr);

        // Objects associated with an OpenCL environment.
        // For each of a number of objects, there is a vector that 
        // can hold all instances of that object.

        std::vector<cl::Platform> platforms;  /*!< List of platforms. */
        /*! \brief List of devices per platform.
         *  \details Holds a vector of devices per platform. */
        std::vector< std::vector<cl::Device> > devices;

    private:
        std::vector<cl::Context> contexts;  /*!< List of contexts. */
        /*! \brief List of queues per context.
         *  \details Holds a vector of queues per context. */
        std::vector< std::vector<cl::CommandQueue> > queues;
        std::vector<cl::Program> programs;  /*!< List of programs. */
        /*! \brief List of kernels per program.
         *  \details Holds a vector of kernels per program. */
        std::vector< std::vector<cl::Kernel> > kernels;

    protected:
        /*! \brief Initializes the OpenGL memory buffers.
         *  \details If CL-GL interop is desirable, CLEnv has to be derived and
         *           `initGLMemObjects` be implemented. `initGLMemObjects` will 
         *           have to create all necessary OpenGL memory buffers.
         *  \note Setting up CL-GL interop requires the following procedure:
         *        (i) Initialize OpenGL context, (ii) Initilize OpenCL context,
         *        (iii) Create OpenGL buffers, (iv) Create OpenCL buffers.
         *  \note Do not call `initGLMemObjects` directly. `initGLMemObjects`  
         *        will be called by `addContext` when it is asked for a 
         *        GL-shared CL context to be created.
         */
        virtual void initGLMemObjects () {};

    private:
        /*! \brief Maps kernel names to kernel indices.
         *         There is one unordered_map for every program.
         *  
         *  For every program in programs, there is an element in kernelIdx.
         *  For every kernel in program i, there is a mapping from the kernel 
         *  name to the kernel index in kernels[i].
         */
        std::vector< std::unordered_map<std::string, unsigned int> > kernelIdx;
    };


    /*! \brief Facilitates the conveyance of `CLEnv` arguments.
     *  \details `CLEnv` creates an OpenCL environment. A `CLEnv` object 
     *           potentially contains many platforms, contexts, queues, etc, 
     *           that are to be used by different (independent) subsystems. 
     *           Those subsystems will have to know where to look inside CLEnv 
     *           for their associated CL objects. `CLEnvInfo` organizes this 
     *           process of information transfer between OpenCL systems.
     *           
     *  \tparam nQueues the number of command queue indices to be held by `CLEnvInfo`.
     */
    template<unsigned int nQueues = 1>
    class CLEnvInfo
    {
    public:
        /*! \brief Initializes a `CLEnvInfo` object.
         *  \details All provided indices are supposed to follow the order the 
         *           associated objects were created in the associated `CLEnv` instance.
         *           
         *  \param[in] _pIdx platform index.
         *  \param[in] _dIdx device index.
         *  \param[in] _ctxIdx context index.
         *  \param[in] _qIdx vector with command queue indices.
         *  \param[in] _pgIdx program index.
         */
        CLEnvInfo (unsigned int _pIdx = 0, unsigned int _dIdx = 0, unsigned int _ctxIdx = 0, 
                   const std::vector<unsigned int> _qIdx = { 0 }, unsigned int _pgIdx = 0) : 
            pIdx (_pIdx), dIdx (_dIdx), ctxIdx (_ctxIdx), pgIdx (_pgIdx)
        {
            try
            {
                if (_qIdx.size () != nQueues)
                    throw "The provided vector of command queue indices has the wrong size";

                qIdx = _qIdx;
            }
            catch (const char *error)
            {
                std::cerr << "Error[CLEnvInfo]: " << error << std::endl;
                exit (EXIT_FAILURE);
            }
        }


        /*! \brief Creates a new `CLEnvInfo` object with the specified command queue.
         *  \details Maintains the same OpenCL configuration, but chooses only one
         *           of the available command queues to include.
         *           
         *  \param[in] idx an index for the `qIdx` vector.
         */
        CLEnvInfo<1> getCLEnvInfo (unsigned int idx)
        {
            try
            {
                return CLEnvInfo<1> (pIdx, dIdx, ctxIdx, { qIdx.at (idx) }, pgIdx);
            }
            catch (const std::out_of_range &error)
            {
                std::cerr << "Out of Range error: " << error.what () 
                          << " (" << __FILE__ << ":" << __LINE__ << ")" << std::endl;
                exit (EXIT_FAILURE);
            }
        }


        unsigned int pIdx;               /*!< Platform index. */
        unsigned int dIdx;               /*!< Device index. */
        unsigned int ctxIdx;             /*!< Context index. */
        std::vector<unsigned int> qIdx;  /*!< Vector of queue indices. */
        unsigned int pgIdx;              /*!< Program index. */
    };


    /*! \brief A class that collects and manipulates timing information 
     *         about a test.
     *  \details It stores the execution times of a test in a vector, 
     *           and then offers summarizing results.
     *           
     *  \tparam nSize the number of test repetitions.
     *  \tparam rep the type of the values the class stores and returns.
     */
    template <unsigned int nSize, typename rep = double>
    class ProfilingInfo
    {
    public:
        /*! \param[in] pLabel a label characterizing the test.
         *  \param[in] pUnit a name for the time unit to be printed 
         *                  when displaying the results.
         */
        ProfilingInfo (std::string pLabel = std::string (), std::string pUnit = std::string ("ms")) 
            : label (pLabel), tExec (nSize), tWidth (4 + log10 (nSize)), tUnit (pUnit)
        {
        }

        /*! \param[in] idx subscript index. */
        rep& operator[] (const unsigned int idx)
        {
            assert (idx >= 0 && idx < nSize);
            return tExec[idx];
        }

        /*! \brief Returns the sum of the \#nSize executon times.
         *  
         *  \param[in] initVal an initial value from which to start counting.
         *  \return The sum of the vector elements.
         */
        rep total (rep initVal = 0.0)
        {
            return std::accumulate (tExec.begin (), tExec.end (), initVal);
        }

        /*! \brief Returns the mean time of the \#nSize executon times.
         *  
         *  \return The mean of the vector elements.
         */
        rep mean ()
        {
            return total() / (rep) tExec.size ();
        }

        /*! \brief Returns the min time of the \#nSize executon times.
         *  
         *  \return The min of the vector elements.
         */
        rep min ()
        {
            return *std::min_element (tExec.begin (), tExec.end ());
        }

        /*! \brief Returns the max time of the \#nSize executon times.
         *  
         *  \return The max of the vector elements.
         */
        rep max ()
        {
            return *std::max_element (tExec.begin (), tExec.end ());
        }

        /*! \brief Returns the relative performance speedup wrt `refProf`.
         *  
         *  \param[in] refProf a reference test.
         *  \return The factor of execution time decrease.
         */
        rep speedup (ProfilingInfo &refProf)
        {
            return refProf.mean () / mean ();
        }

        /*! \brief Displays summarizing results on the test.
         *  
         *  \param[in] title a title for the table of results.
         *  \param[in] bLine a flag for whether or not to print a newline 
         *                   at the end of the table.
         */
        void print (const char *title = nullptr, bool bLine = true)
        {
            std::ios::fmtflags f (std::cout.flags ());
            std::cout << std::fixed << std::setprecision (3);

            if (title)
                std::cout << std::endl << title << std::endl << std::endl;
            else
                std::cout << std::endl;

            std::cout << " " << label << std::endl;
            std::cout << " " << std::string (label.size (), '-') << std::endl;
            std::cout << "   Mean   : " << std::setw (tWidth) << mean ()  << " " << tUnit << std::endl;
            std::cout << "   Min    : " << std::setw (tWidth) << min ()   << " " << tUnit << std::endl;
            std::cout << "   Max    : " << std::setw (tWidth) << max ()   << " " << tUnit << std::endl;
            std::cout << "   Total  : " << std::setw (tWidth) << total () << " " << tUnit << std::endl;
            if (bLine) std::cout << std::endl;

            std::cout.flags (f);
        }

        /*! \brief Displays summarizing results on two tests.
         *  \details Compares the two tests by calculating the speedup 
         *           on the mean execution times.
         *  \note I didn't bother handling the units. It's your responsibility 
         *        to enforce the same unit of time on the two objects.
         *  
         *  \param[in] refProf a reference test.
         *  \param[in] title a title for the table of results.
         */
        void print (ProfilingInfo &refProf, const char *title = nullptr)
        {
            if (title)
                std::cout << std::endl << title << std::endl;

            refProf.print (nullptr, false);
            print (nullptr, false);

            std::cout << std::endl << " Benchmark" << std::endl << " ---------" << std::endl;
            
            std::cout << "   Speedup: " << std::setw (tWidth) << speedup (refProf) << std::endl << std::endl;
        }

    private:
        std::string label;  /*!< A label characterizing the test. */
        std::vector<rep> tExec;  /*!< Execution times. */
        uint8_t tWidth;  /*!< Width of the results when printing. */
        std::string tUnit;  /*!< Time unit to display when printing the results. */
    };


    /*! \brief A class for measuring execution times.
     *  \details CPUTimer is an interface for `std::chrono::duration`.
     *  
     *  \tparam rep the type of the value returned by `duration`.
     *  \tparam period the unit of time for the value returned by `duration`.
     *                 It is declared as an `std::ratio<std::intmax_t num, std::intmax_t den>`.
     */
    template <typename rep = int64_t, typename period = std::milli>
    class CPUTimer
    {
    public:
        /*! \brief Constructs a timer.
         *  \details The timer doesn't start automatically.
         * 
         *  \param[in] initVal a value to initialize the timer with.
         */
        CPUTimer (int initVal = 0) : tDuration (initVal)
        {
        }

        /*! \brief Starts the timer.
         *  
         *  \param[in] tReset a flag for resetting the timer before the timer starts. 
         *                    If `false`, the timer starts counting from 
         *                    the point it reached the last time it stopped.
         */
        void start (bool tReset = true)
        {
            if (tReset)
                reset ();

            tReference = std::chrono::high_resolution_clock::now ();
        }

        /*! \brief Stops the timer. 
         *
         *  \return The time measured up to this point in `period` units.
         */
        rep stop ()
        {
            tDuration += std::chrono::duration_cast< std::chrono::duration<rep, period> > 
                (std::chrono::high_resolution_clock::now () - tReference);

            return duration ();
        }

        /*! \brief Returns the time measured by the timer.
         *  \details This time is measured up to the point the timer last time stopped.
         *  
         *  \return The time in `period` units.
         */
        rep duration ()
        {
            return tDuration.count ();
        }

        /*! \brief Resets the timer. */
        void reset ()
        {
            tDuration = std::chrono::duration<rep, period>::zero ();
        }

    private:
        /*! A reference point for when the timer started. */
        std::chrono::time_point<std::chrono::high_resolution_clock> tReference;
        /*! The time measured by the timer. */
        std::chrono::duration<rep, period> tDuration;
    };


    /*! \brief A class for profiling CL devices.
     *
     *  \tparam period the unit of time for the value returned by `duration`.
     *                 It is declared as an `std::ratio<std::intmax_t num, std::intmax_t den>`.
     */
    template <typename period = std::milli>
    class GPUTimer
    {
    public:
        /*! \param[in] device the targeted for profiling CL device.
         */
        GPUTimer (cl::Device &device)
        {
            period tPeriod;
            // Converts nanoseconds to seconds and then to the requested scale
            tUnit = (double) tPeriod.den / (double) tPeriod.num / 1000000000.0;
        }

        /*! \brief Returns a new unpopulated event.
         *  \details The last populated event gets dismissed.
         *  
         *  \return An event for the profiling process.
         */
        cl::Event& event ()
        {
            return pEvent;
        }

        /*! \brief This is an interface for `cl::Event::wait`.
         */
        void wait ()
        {
            pEvent.wait ();
        }

        /*! \brief Returns the time measured by the timer.
         *  \note It's important that it's called after a call to `wait`.
         *
         *  \return The time in `period` units.
         */
        double duration ()
        {
            cl_ulong start = pEvent.getProfilingInfo<CL_PROFILING_COMMAND_START> ();
            cl_ulong end = pEvent.getProfilingInfo<CL_PROFILING_COMMAND_END> ();

            return (end - start) * tUnit;
        }

        /*! \brief Returns the unit of the timer.
         *
         *  \return The time unit in seconds.
         */
        double getUnit ()
        {
            return tUnit;
        }

    private:
        cl::Event pEvent;  /*!< The profiling event. */
        double tUnit;  /*!< A factor to set the scale for the measured time. */
    };

}

#endif  // CLUTILS_HPP

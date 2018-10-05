#include <iostream>
#include <fstream>
#include <utility>
#include <CL/cl.hpp>

int main()
{
    // Create the two input vectors
    const auto LIST_SIZE = 10;
    auto A = std::make_unique<int[]>(LIST_SIZE);
    auto B = std::make_unique<int[]>(LIST_SIZE);
    for (auto i = 0; i < LIST_SIZE; i++) {
        A[i] = i;
        B[i] = LIST_SIZE - i;
    }

    try {
        // Get available platforms
        std::vector<cl::Platform> platforms;
        cl::Platform::get(&platforms);

        std::cout << "Available platforms: " << std::endl;
        for(const auto &platform : platforms)
        {
            std::cout << platform.getInfo<CL_PLATFORM_NAME>() << std::endl;
        }
        // Select the default platform and create a context using this platform and the GPU
        cl_context_properties cps[3] = {
            CL_CONTEXT_PLATFORM,
            reinterpret_cast<cl_context_properties>((platforms[0])()),
            0
        };

        cl::Context context(CL_DEVICE_TYPE_ALL, cps);

        // Get a list of devices on this platform
        auto devices = context.getInfo<CL_CONTEXT_DEVICES>();
        if(devices.empty())
        {
            std::cerr << "No devices available!" << std::endl;
            return -1;
        }

        std::cout << "Available devices: " << std::endl;
        for(const auto &device : devices)
        {
            std::cout << device.getInfo<CL_DEVICE_NAME>() << std::endl;
        }
        // Create a command queue and use the first device
        auto queue = cl::CommandQueue(context, devices[0]);

        // Read source file
        std::ifstream sourceFile(std::string(OPEN_CL_KERNELS_DIR) + "//vector_add_kernel.cl");
        if(!sourceFile.good())
        {
            std::cerr << "Error reading cl source file" << std::endl;
        }
        std::string sourceCode(
            std::istreambuf_iterator<char>(sourceFile),
            (std::istreambuf_iterator<char>()));
        cl::Program::Sources source(1, std::make_pair(sourceCode.c_str(), sourceCode.length() + 1));

        // Make program of the source code in the context
        cl::Program program = cl::Program(context, source);

        // Build program for these specific devices
        auto program_build_result = program.build(devices);

        // Make kernel
        cl::Kernel kernel(program, "vector_add");

        // Create memory buffers
        cl::Buffer bufferA = cl::Buffer(context, CL_MEM_READ_ONLY, LIST_SIZE * sizeof(int));
        cl::Buffer bufferB = cl::Buffer(context, CL_MEM_READ_ONLY, LIST_SIZE * sizeof(int));
        cl::Buffer bufferC = cl::Buffer(context, CL_MEM_WRITE_ONLY, LIST_SIZE * sizeof(int));

        // Copy lists A and B to the memory buffers
        queue.enqueueWriteBuffer(bufferA, CL_TRUE, 0, LIST_SIZE * sizeof(int), A.get());
        queue.enqueueWriteBuffer(bufferB, CL_TRUE, 0, LIST_SIZE * sizeof(int), B.get());

        // Set arguments to kernel
        kernel.setArg(0, bufferA);
        kernel.setArg(1, bufferB);
        kernel.setArg(2, bufferC);

        // Run the kernel on specific ND range
        cl::NDRange global(LIST_SIZE);
        cl::NDRange local(1);
        queue.enqueueNDRangeKernel(kernel, cl::NullRange, global, local);

        // Read buffer C into a local list
        auto C = new int[LIST_SIZE];
        queue.enqueueReadBuffer(bufferC, CL_TRUE, 0, LIST_SIZE * sizeof(int), C);

        for (auto i = 0; i < LIST_SIZE; i++)
            std::cout << A[i] << " + " << B[i] << " = " << C[i] << std::endl;

        delete[] C;
    }
    catch (cl::Error &error) {
        std::cout << error.what() << "(" << error.err() << ")" << std::endl;
    }
    return 0;
}
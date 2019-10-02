#include <algorithm>
#include <cassert>
#include <chrono>
#include <cmath>
#include <cuda_runtime_api.h>
#include <fstream>
#include <functional>
#include <iostream>
#include <iterator>
#include <map>
#include <random>
#include <sstream>
#include <string.h>
#include <sys/stat.h>
#include <time.h>
#include <vector>
#include <dirent.h>
#include <Python.h>

#include "NvCaffeParser.h"
#include "NvInfer.h"
#include "NvInferPlugin.h"
#include "NvUffParser.h"
#include <opencv2/opencv.hpp>
#include "common.h"

using namespace nvinfer1;
using namespace nvcaffeparser1;
using namespace nvuffparser;
using namespace cv;
using namespace std;
using namespace std::chrono;

struct Params
{
    std::string deployFile;
    std::string modelFile;
    std::string engine;
    std::string calibrationCache{"CalibrationTable"};
    std::string modelName;
    std::string uffFile;
    std::vector<std::string> inputs;
    std::vector<std::string> outputs;
    std::vector<std::pair<std::string, Dims3>> uffInputs;
    char *imagePath;
    int device{0};
    int batchSize{1};
    int workspaceSize{16};
    int iterations{10};
    int avgRuns{10};
    int useDLACore{-1};
    bool fp16{false};
    bool int8{false};
    bool verbose{false};
    bool allowGPUFallback{false};
    float pct{99};
    bool kernel{false};
} gParams;

inline int volume(Dims dims)
{
    return std::accumulate(dims.d, dims.d + dims.nbDims, 1, std::multiplies<int>());
}

std::map<std::string, Dims3> gInputDimensions;

std::vector<std::string> split(const std::string& s, char delim)
{
    std::vector<std::string> res;
    std::stringstream ss;
    ss.str(s);
    std::string item;
    while (std::getline(ss, item, delim))
    {
        res.push_back(item);
    }
    return res;
}

float percentile(float percentage, std::vector<float>& times)
{
    int all = static_cast<int>(times.size());
    int exclude = static_cast<int>((1 - percentage / 100) * all);
    if (0 <= exclude && exclude <= all)
    {
        std::sort(times.begin(), times.end());
        return times[all == exclude ? 0 : all - 1 - exclude];
    }
    return std::numeric_limits<float>::infinity();
}

// Logger for TensorRT info/warning/errors
class iLogger : public ILogger
{
    void log(Severity severity, const char* msg) override
    {
        // suppress info-level messages
        if (severity != Severity::kINFO || gParams.verbose)
            std::cout << msg << std::endl;
    }
} gLogger;

std::vector<std::string> mClassSynset, mClassDesc;
DimsCHW outputDims;
bool loadClassInfo( const char* filename )
{
	if( !filename )
		return false;
	
	FILE* f = fopen(filename, "r");
	
	if( !f )
	{
		printf("imageNet -- failed to open %s\n", filename);
		return false;
	}
	
	char str[512];

	while( fgets(str, 512, f) != NULL )
	{
		const int syn = 9;  // length of synset prefix (in characters)
		const int len = strlen(str);
		
		if( len < syn + 1 )
			continue;
		
		str[syn]   = 0;
		str[len-1] = 0;
		
		const std::string a = str;
		const std::string b = (str + syn + 1);
		
		//printf("a=%s b=%s\n", a.c_str(), b.c_str());
		mClassSynset.push_back(a);
		mClassDesc.push_back(b);
	}
	
	fclose(f);
	
	printf("imageNet -- loaded %zu class info entries\n", mClassSynset.size());
	
	if( mClassSynset.size() == 0 )
		return false;
	
	return true;
}

class RndInt8Calibrator : public IInt8EntropyCalibrator
{
public:
    RndInt8Calibrator(int totalSamples, std::string cacheFile)
        : mTotalSamples(totalSamples)
        , mCurrentSample(0)
        , mCacheFile(cacheFile)
    {
    }

    ~RndInt8Calibrator()
    {

    }

    int getBatchSize() const override
    {
        return 1;
    }

    bool getBatch(void* bindings[], const char* names[], int nbBindings) override
    {
        return true;
    }

    const void* readCalibrationCache(size_t& length) override
    {
        mCalibrationCache.clear();
        std::ifstream input(mCacheFile, std::ios::binary);
        input >> std::noskipws;
        if (input.good())
            std::copy(std::istream_iterator<char>(input), std::istream_iterator<char>(), std::back_inserter(mCalibrationCache));

        length = mCalibrationCache.size();
        return length ? &mCalibrationCache[0] : nullptr;
    }

    virtual void writeCalibrationCache(const void*, size_t) override
    {
    }

private:
    int mTotalSamples;
    int mCurrentSample;
    std::string mCacheFile;
    std::map<std::string, void*> mInputDeviceBuffers;
    std::vector<char> mCalibrationCache;
};

void configureBuilder(IBuilder* builder, RndInt8Calibrator& calibrator)
{
    builder->setMaxBatchSize(gParams.batchSize);
    builder->setMaxWorkspaceSize(gParams.workspaceSize << 20);
    builder->setFp16Mode(gParams.fp16);

    if (gParams.int8)
    {
        builder->setInt8Mode(true);
        builder->setInt8Calibrator(&calibrator);
    }

    if (gParams.useDLACore >= 0)
    {
        builder->setDefaultDeviceType(DeviceType::kDLA);
        builder->setDLACore(gParams.useDLACore);
        if (gParams.allowGPUFallback)
            builder->allowGPUFallback(true);
    }
}

ICudaEngine* caffeToTRTModel()
{
    // create the builder
    IBuilder* builder = createInferBuilder(gLogger);

    // parse the caffe model to populate the network, then set the outputs
    INetworkDefinition* network = builder->createNetwork();
    ICaffeParser* parser = createCaffeParser();
    const IBlobNameToTensor* blobNameToTensor = parser->parse(gParams.deployFile.c_str(),
                                                              gParams.modelFile.empty() ? 0 : gParams.modelFile.c_str(),
                                                              *network,
                                                              gParams.fp16 ? nvinfer1::DataType::kHALF : nvinfer1::DataType::kFLOAT);

    if (!blobNameToTensor)
        return nullptr;

    for (int i = 0, n = network->getNbInputs(); i < n; i++)
    {
        Dims3 dims = static_cast<Dims3&&>(network->getInput(i)->getDimensions());
        gParams.inputs.push_back(network->getInput(i)->getName());
        gInputDimensions.insert(std::make_pair(network->getInput(i)->getName(), dims));
        std::cout << "Input \"" << network->getInput(i)->getName() << "\": " << dims.d[0] << "x" << dims.d[1] << "x" << dims.d[2] << std::endl;
    }

    // specify which tensors are outputs
    for (auto& s : gParams.outputs)
    {
        if (blobNameToTensor->find(s.c_str()) == nullptr)
        {
            std::cout << "could not find output blob " << s << std::endl;
            return nullptr;
        }
        network->markOutput(*blobNameToTensor->find(s.c_str()));
    }

    for (int i = 0, n = network->getNbOutputs(); i < n; i++)
    {
        Dims3 dims = static_cast<Dims3&&>(network->getOutput(i)->getDimensions());
        std::cout << "Output \"" << network->getOutput(i)->getName() << "\": " << dims.d[0] << "x" << dims.d[1] << "x"
                  << dims.d[2] << std::endl;
    }

    // Build the engine
    RndInt8Calibrator calibrator(1, gParams.calibrationCache);
    configureBuilder(builder, calibrator);

    ICudaEngine* engine = builder->buildCudaEngine(*network);
    if (engine == nullptr)
        std::cout << "could not build engine" << std::endl;

    parser->destroy();
    network->destroy();
    builder->destroy();
    return engine;
}

ICudaEngine* uffToTRTModel()
{
    // create the builder
    IBuilder* builder = createInferBuilder(gLogger);

    // parse the caffe model to populate the network, then set the outputs
    INetworkDefinition* network = builder->createNetwork();
    IUffParser* parser = createUffParser();

    // specify which tensors are outputs
    for (auto& s : gParams.outputs)
    {
        if (!parser->registerOutput(s.c_str()))
        {
            std::cerr << "Failed to register output " << s << std::endl;
            return nullptr;
        }
    }

    // specify which tensors are inputs (and their dimensions)
    for (auto& s : gParams.uffInputs)
    {
        if (!parser->registerInput(s.first.c_str(), s.second, UffInputOrder::kNCHW))
        {
            std::cerr << "Failed to register input " << s.first << std::endl;
            return nullptr;
        }
    }

    if (!parser->parse(gParams.uffFile.c_str(), *network, gParams.fp16 ? nvinfer1::DataType::kHALF : nvinfer1::DataType::kFLOAT))
        return nullptr;

    for (int i = 0, n = network->getNbInputs(); i < n; i++)
    {
        Dims3 dims = static_cast<Dims3&&>(network->getInput(i)->getDimensions());
        gParams.inputs.push_back(network->getInput(i)->getName());
        gInputDimensions.insert(std::make_pair(network->getInput(i)->getName(), dims));
    }

    // Build the engine
    RndInt8Calibrator calibrator(1, gParams.calibrationCache);
    configureBuilder(builder, calibrator);

    ICudaEngine* engine = builder->buildCudaEngine(*network);
    if (engine == nullptr)
        std::cout << "could not build engine" << std::endl;

    parser->destroy();
    network->destroy();
    builder->destroy();
    return engine;
}

void createMemoryInput(const ICudaEngine& engine, std::vector<void*>& buffers, const std::string& name)
{
	size_t bindingIndex = engine.getBindingIndex(name.c_str());
	printf("name=%s, bindingIndex=%d, buffers.size()=%d\n", name.c_str(), (int)bindingIndex, (int)buffers.size());
	assert(bindingIndex < buffers.size());
	DimsCHW dimensions = static_cast<DimsCHW&&>(engine.getBindingDimensions((int)bindingIndex));
	size_t eltCount = dimensions.c()*dimensions.h()*dimensions.w()*gParams.batchSize, memSize = eltCount * sizeof(float);
	float* localMem = new float[eltCount];

	size_t singleImage = dimensions.c()*dimensions.h()*dimensions.w();
	struct dirent **namelist;
	int n;
	n = scandir(gParams.imagePath, &namelist, 0, versionsort);
	if (n<0)
	{
		printf("Invalid arguments..................");
	}else{
		int count = 1;
		while( count <= gParams.batchSize)
		{	
			for(int i=2 ; i<n ; ++i) {
					if (count > gParams.batchSize) break;
					std::string filename = std::string(gParams.imagePath) + std::string(namelist[i]->d_name);
					const char* imgFilename = filename.c_str();
					Mat image = imread( imgFilename, IMREAD_COLOR );

					printf("Sucessfully loaded image %s\n", imgFilename);
					cvtColor(image, image, cv::COLOR_BGR2RGB, 3);
					resize(image, image, Size(dimensions.w(), dimensions.h()));
					
					const float mean[3] = { 103.94 , 116.78 ,123.68};				
					const size_t height = image.rows;
				  	const size_t width = image.cols;
				  	const size_t channels = image.channels();
				  	const size_t numel = height * width * channels;
				  	const size_t stridesCv[3] = { width * channels, channels, 1 };
				  	const size_t strides[3] = { height * width, width, 1 };
					int map[3] = {2,1,0};
					for (int i = 0; i < height; i++) 
					  {
					    for (int j = 0; j < width; j++) 
					    {
					      for (int k = 0; k < channels; k++) 
					      {
						const size_t offsetCv = i * width * channels + j * channels + map[k];
						const size_t offset = k * strides[0] + i * strides[1] + j * strides[2];
						localMem[((count-1)* singleImage) +offset] = ((float) image.data[offsetCv] - mean[k]) ;
					      }
					    }
					  }
					count += 1;
			}
		}
	}
	buffers[bindingIndex] = localMem;	
}

void createMemory(const ICudaEngine& engine, std::vector<void*>& buffers, const std::string& name)
{
    const int bindingIndex = engine.getBindingIndex(name.c_str());
    printf("name=%s, bindingIndex=%d, buffers.size()=%d\n", name.c_str(), bindingIndex, (int) buffers.size());
    assert((bindingIndex < (int) buffers.size()) && "Input/output name not found in network");

    const Dims dims = engine.getBindingDimensions((int) bindingIndex);
    const size_t eltCount = volume(dims) * gParams.batchSize;
    const size_t memSize = eltCount * sizeof(float);

    // Init host memory with random values
   // std::vector<float> localMem(eltCount);
float* localMem = new float[eltCount];
    for (size_t i = 0; i < eltCount; i++)
        localMem[i] = (float(rand()) / RAND_MAX) * 2 - 1;

    buffers[bindingIndex] = localMem;
}

void createMemoryInput_kernel(const ICudaEngine& engine, std::vector<void*>& buffers, const std::string& name)
{
	size_t bindingIndex = engine.getBindingIndex(name.c_str());
	printf("name=%s, bindingIndex=%d, buffers.size()=%d\n", name.c_str(), (int)bindingIndex, (int)buffers.size());
	assert(bindingIndex < buffers.size());
	DimsCHW dimensions = static_cast<DimsCHW&&>(engine.getBindingDimensions((int)bindingIndex));
	size_t eltCount = dimensions.c()*dimensions.h()*dimensions.w()*gParams.batchSize, memSize = eltCount * sizeof(float);
	float* localMem = new float[eltCount];

	size_t singleImage = dimensions.c()*dimensions.h()*dimensions.w();
	struct dirent **namelist;
	int n;
	n = scandir(gParams.imagePath, &namelist, 0, versionsort);
	if (n<0)
	{
		printf("Invalid arguments..................");
	}else{
		int count = 1;
		while( count <= gParams.batchSize)
		{	
			for(int i=2 ; i<n ; ++i) {
					if (count > gParams.batchSize) break;
					std::string filename = std::string(gParams.imagePath) + std::string(namelist[i]->d_name);
					const char* imgFilename = filename.c_str();
					Mat image = imread( imgFilename, IMREAD_COLOR );

					printf("Sucessfully loaded image %s\n", imgFilename);
					cvtColor(image, image, cv::COLOR_BGR2RGB, 3);
					resize(image, image, Size(dimensions.w(), dimensions.h()));
					
					const float mean[3] = { 103.94 , 116.78 ,123.68};				
					const size_t height = image.rows;
				  	const size_t width = image.cols;
				  	const size_t channels = image.channels();
				  	const size_t numel = height * width * channels;
				  	const size_t stridesCv[3] = { width * channels, channels, 1 };
				  	const size_t strides[3] = { height * width, width, 1 };
					int map[3] = {2,1,0};
					for (int i = 0; i < height; i++) 
					  {
					    for (int j = 0; j < width; j++) 
					    {
					      for (int k = 0; k < channels; k++) 
					      {
						const size_t offsetCv = i * width * channels + j * channels + map[k];
						const size_t offset = k * strides[0] + i * strides[1] + j * strides[2];
						localMem[((count-1)* singleImage) +offset] = ((float) image.data[offsetCv] - mean[k]) ;
					      }
					    }
					  }
					count += 1;
			}
		}
	}
        
    // Alloc and copy host values to device
    void* deviceMem;
    CHECK(cudaMalloc(&deviceMem, memSize));
    if (deviceMem == nullptr)
    {
        std::cerr << "Out of memory allocating bytes: " << memSize << std::endl;
        exit(1);
    }
    CHECK(cudaMemcpy(deviceMem, localMem, memSize, cudaMemcpyHostToDevice));
	buffers[bindingIndex] = deviceMem;	
}

void createMemory_kernel(const ICudaEngine& engine, std::vector<void*>& buffers, const std::string& name)
{
    const int bindingIndex = engine.getBindingIndex(name.c_str());
    printf("name=%s, bindingIndex=%d, buffers.size()=%d\n", name.c_str(), bindingIndex, (int) buffers.size());
    assert((bindingIndex < (int) buffers.size()) && "Input/output name not found in network");

    const Dims dims = engine.getBindingDimensions((int) bindingIndex);
    const size_t eltCount = volume(dims) * gParams.batchSize;
    const size_t memSize = eltCount * sizeof(float);

    // Init host memory with random values
   // std::vector<float> localMem(eltCount);
float* localMem = new float[eltCount];
    for (size_t i = 0; i < eltCount; i++)
        localMem[i] = (float(rand()) / RAND_MAX) * 2 - 1;

    // Alloc and copy host values to device
    void* deviceMem;
    CHECK(cudaMalloc(&deviceMem, memSize));
    if (deviceMem == nullptr)
    {
        std::cerr << "Out of memory allocating bytes: " << memSize << std::endl;
        exit(1);
    }
    CHECK(cudaMemcpy(deviceMem, localMem, memSize, cudaMemcpyHostToDevice));
	buffers[bindingIndex] = deviceMem;	
}

void doInference(ICudaEngine& engine)
{
	IExecutionContext *context = engine.createExecutionContext();
	// input and output buffer pointers that we pass to the engine - the engine requires exactly IEngine::getNbBindings(),
	// of these, but in this case we know that there is exactly one input and one output.

	std::vector<void*> buffers(gParams.inputs.size() + gParams.outputs.size());
	size_t inputBindingIndex;
	for (size_t i = 0; i < gParams.inputs.size(); i++){
		createMemoryInput(engine, buffers, gParams.inputs[i]);
		inputBindingIndex = engine.getBindingIndex(gParams.inputs[i].c_str());
	}
	size_t outputBindingIndex;
	for (size_t i = 0; i < gParams.outputs.size(); i++){
		createMemory(engine, buffers, gParams.outputs[i]);
                outputBindingIndex = engine.getBindingIndex(gParams.outputs[i].c_str());
	}
	cudaStream_t stream;
	CHECK(cudaStreamCreate(&stream));
	//cudaEvent_t start, end;
	//CHECK(cudaEventCreateWithFlags(&start, cudaEventBlockingSync));
	//CHECK(cudaEventCreateWithFlags(&end, cudaEventBlockingSync));
	std::vector<void*> deviceBufferrs(gParams.inputs.size() + gParams.outputs.size());
	float total = 0, ms; 
	double total_system = 0;
	
        std::vector<float> times(gParams.iterations);
	for (int j = 0; j < gParams.iterations+10; j++)
	{
		
		DimsCHW dimensions = static_cast<DimsCHW&&>(engine.getBindingDimensions((int)inputBindingIndex));
		DimsCHW dimensionsOut = static_cast<DimsCHW&&>(engine.getBindingDimensions((int)outputBindingIndex));
		size_t eltCount = dimensions.c()*dimensions.h()*dimensions.w()*gParams.batchSize, memSize = eltCount * sizeof(float);
		size_t eltCountOut =  dimensionsOut.c()*gParams.batchSize, memSizeOut = eltCountOut * sizeof(float);
                cout << "OP dims ==  " << dimensionsOut.c() << endl;
		//System time start	
		auto t0 = high_resolution_clock::now();

		void* deviceInputMem; void* deviceOutputMem;
		CHECK(cudaMalloc(&deviceInputMem, memSize));
		if (deviceInputMem == nullptr)
		{
			std::cerr << "Out of memory" << std::endl;
			exit(1);
		}
		CHECK(cudaMalloc(&deviceOutputMem, memSizeOut));
		if (deviceOutputMem == nullptr)
		{
			std::cerr << "Out of memory" << std::endl;
			exit(1);
		}
		
		cudaMemcpyAsync(deviceInputMem, buffers[inputBindingIndex], 
			memSize, 
			cudaMemcpyHostToDevice, stream);
		deviceBufferrs[inputBindingIndex]=deviceInputMem;
		deviceBufferrs[outputBindingIndex]=deviceOutputMem;
		
		context->enqueue(gParams.batchSize, &deviceBufferrs[0], stream, nullptr);
		
		cudaMemcpyAsync(buffers[outputBindingIndex], deviceBufferrs[outputBindingIndex], 
			memSizeOut, 
			cudaMemcpyDeviceToHost, stream);
		
		cudaStreamSynchronize(stream);
		//System time end	
		auto t1 = high_resolution_clock::now();
		std::chrono::duration<double, std::milli> elapsed_ms = t1 - t0;
		if(j>9){
			total_system += elapsed_ms.count();
		        times[j-10] = elapsed_ms.count();
		}
		
		if(j < gParams.iterations-1)
		{
			cudaFree(deviceInputMem);
			cudaFree(deviceInputMem);
		}
		
		std::cout << "System Latency of iteration  #" << j+1 << " is " << elapsed_ms.count() << " ms." << std::endl;
		
	}
	
	total_system /= gParams.iterations;
	
	std::cout << "System:  Average over " << gParams.iterations << " iterations is " << total_system << " ms." << std::endl;
	std::cout << "System: Imgs/sec"  << (gParams.batchSize*1000)/total_system << std::endl;
	std::string precision;	
	if(gParams.fp16){ precision = "fp16";}
	else if(gParams.int8){precision = "int8";}
	else { precision = "fp32"; }        
        float pct50 = percentile(50, times);
        float pct90 = percentile(90, times);
        float pct95 = percentile(95, times);
        float pct99 = percentile(99, times);
        double max = *max_element(times.begin(), times.end());
        double min = *min_element(times.begin(), times.end());
	std::string const command = "python3 cpp_to_python_api.py " + gParams.modelName +" "+std::to_string(gParams.batchSize)+" gpu "+precision+" NA NA "+std::to_string((gParams.batchSize*1000)/total_system)+" "+std::to_string(total_system)+" ImagesPerSec msec "+std::to_string(gParams.iterations)+" ILSVRC-2012 "+std::to_string(pct50)+" "+std::to_string(pct90)+" "+std::to_string(pct95)+" "+std::to_string(pct99)+" "+std::to_string(max)+" "+std::to_string(min)+" 1";
	if(system(command.c_str()) == 0) // c++ system call returns 0 if successful
	    {
		std::cout << "Execution Completed!" << std::endl;
	    }
	    else{
		std::cout << "Execution Unsuccessfull!" << std::endl;
	    }
	if(!loadClassInfo("synset_words.txt"))
	{
		printf("imageNet -- failed to load synset class descriptions ");
	}
	DimsCHW dimensionsOut = static_cast<DimsCHW&&>(engine.getBindingDimensions((int)outputBindingIndex));
	size_t eltCount = dimensionsOut.c()*gParams.batchSize, memSize = eltCount * sizeof(float);
	float* localMem = new float[eltCount];
	CHECK(cudaMemcpy(localMem, deviceBufferrs[outputBindingIndex], memSize, cudaMemcpyDeviceToHost));
	size_t mOutputClasses =  dimensionsOut.c();
	std::cout << "Output Classes = " << mOutputClasses << std::endl;
	for (int count = 0; count < gParams.batchSize; count ++ )
	{
		
		int classIndex = -1;
		float classMax = -1.0f;
		for( size_t n=0 ; n< mOutputClasses ; n++)
		{
			size_t ne = n + ( mOutputClasses * count);
			float value = localMem[ne];
		
			if( value > classMax )
			{
				classIndex = n;
				classMax = value;
			}
			//std::cout << "Class index  = " << n << "  Value = " << value << std::endl;
		}
		printf("Class output = %d\n ", classIndex);
		std::cout << "Class Max  " << classMax << std::endl;
	}
   //Adding output code*
	cudaStreamDestroy(stream);
    	context->destroy();
}

void doInference_kernel(ICudaEngine& engine)
{
    IExecutionContext* context = engine.createExecutionContext();
    // input and output buffer pointers that we pass to the engine - the engine requires exactly IEngine::getNbBindings(),
    // of these, but in this case we know that there is exactly one input and one output.

    std::vector<void*> buffers(gParams.inputs.size() + gParams.outputs.size());
    for (size_t i = 0; i < gParams.inputs.size(); i++)
        createMemoryInput_kernel(engine, buffers, gParams.inputs[i]);
    size_t outputBindingIndex;
    for (size_t i = 0; i < gParams.outputs.size(); i++){
        createMemory_kernel(engine, buffers, gParams.outputs[i]);
        outputBindingIndex = engine.getBindingIndex(gParams.outputs[i].c_str());
   
     }
    cudaStream_t stream;
    CHECK(cudaStreamCreate(&stream));
    cudaEvent_t start, end;
    CHECK(cudaEventCreateWithFlags(&start, cudaEventBlockingSync));
    CHECK(cudaEventCreateWithFlags(&end, cudaEventBlockingSync));

    std::vector<float> times(gParams.iterations);
    float totalGpu{0};
    for (int j = 0; j < gParams.iterations+10; j++)
    {
            cudaEventRecord(start, stream);
            context->enqueue(gParams.batchSize, &buffers[0], stream, nullptr);
            cudaEventRecord(end, stream);
            cudaEventSynchronize(end);
            float ms;
            cudaEventElapsedTime(&ms, start, end);
            if(j>9){
		times[j-10] = ms;
                totalGpu += ms;
	    }
    }
    totalGpu /= gParams.iterations;
    std::cout << "Average over " << gParams.iterations << " runs is " << totalGpu << " ms " << static_cast<int>(gParams.pct) << "\% percentile time is " << percentile(gParams.pct, times) << ")." << std::endl;
    std::cout << "Kernel: Imgs/sec"  << (gParams.batchSize*1000)/totalGpu << std::endl;
    std::string precision;	
    if(gParams.fp16){ precision = "fp16";}
    else if(gParams.int8){precision = "int8";}
    else { precision = "fp32"; }
    float pct50 = percentile(50, times);
    float pct90 = percentile(90, times);
    float pct95 = percentile(95, times);
    float pct99 = percentile(99, times);
    double max = *max_element(times.begin(), times.end());
    double min = *min_element(times.begin(), times.end());
    std::string const command = "python3 cpp_to_python_api.py " + gParams.modelName +" "+std::to_string(gParams.batchSize)+" gpu "+precision+" "+std::to_string((gParams.batchSize*1000)/totalGpu)+" "+std::to_string(totalGpu)+" NA NA ImagesPerSec msec "+std::to_string(gParams.iterations)+" ILSVRC-2012 "+std::to_string(pct50)+" "+std::to_string(pct90)+" "+std::to_string(pct95)+" "+std::to_string(pct99)+" "+std::to_string(max)+" "+std::to_string(min);
   if(system(command.c_str()) == 0) // c++ system call returns 0 if successful
    {
	std::cout << "Execution Completed!" << std::endl;
    }
    else{
	std::cout << "Execution Unsuccessfull!" << std::endl;
    }
    if(!loadClassInfo("synset_words.txt"))
    {
	printf("imageNet -- failed to load synset class descriptions ");
    }
    DimsCHW dimensionsOut = static_cast<DimsCHW&&>(engine.getBindingDimensions((int)outputBindingIndex));
    size_t eltCount = dimensionsOut.c()*gParams.batchSize, memSize = eltCount * sizeof(float);
    float* localMem = new float[eltCount];
    CHECK(cudaMemcpy(localMem, buffers[outputBindingIndex], memSize, cudaMemcpyDeviceToHost));
    size_t mOutputClasses =  dimensionsOut.c();
    std::cout << "Output Classes = " << mOutputClasses << std::endl;
    for (int count = 0; count < gParams.batchSize; count ++ )
    {
	int classIndex = -1;
	float classMax = -1.0f;
	for( size_t n=0 ; n< mOutputClasses ; n++)
	{
	   size_t ne = n + ( mOutputClasses * count);
	   float value = localMem[ne];
	   if( value > classMax )
	   {
		classIndex = n;
		classMax = value;
	   }
	 }
	 printf("Class output = %d\n ", classIndex);
	 std::cout << "Class Max  " << classMax << std::endl;
    }
    cudaStreamDestroy(stream);
    cudaEventDestroy(start);
    cudaEventDestroy(end);
    context->destroy();
}

static void printUsage()
{
    printf("\n");
    printf("Mandatory params:\n");
    printf("  --deploy=<file>      Caffe deploy file\n");
    printf("  OR --uff=<file>      UFF file\n");
    printf("  --output=<name>      Output blob name (can be specified multiple times)\n");

    printf("\nMandatory params for onnx:\n");
    printf("  --onnx=<file>        ONNX Model file\n");

    printf("\nOptional params:\n");

    printf("  --uffInput=<name>,C,H,W Input blob name and its dimensions for UFF parser (can be specified multiple times)\n");
    printf("  --input=<name>          Input blob name (can be specified multiple times)\n");
    printf("  --model=<file>          Caffe model file (default = no model, random weights used)\n");
    printf("  --batch=N               Set batch size (default = %d)\n", gParams.batchSize);
    printf("  --device=N              Set cuda device to N (default = %d)\n", gParams.device);
    printf("  --iterations=N          Run N iterations (default = %d)\n", gParams.iterations);
    printf("  --avgRuns=N             Set avgRuns to N - perf is measured as an average of avgRuns (default=%d)\n", gParams.avgRuns);
    printf("  --percentile=P          For each iteration, report the percentile time at P percentage (0<=P<=100, with 0 representing min, and 100 representing max; default = %.1f%%)\n", gParams.pct);
    printf("  --workspace=N           Set workspace size in megabytes (default = %d)\n", gParams.workspaceSize);
    printf("  --fp16                  Run in fp16 mode (default = false). Permits 16-bit kernels\n");
    printf("  --int8                  Run in int8 mode (default = false). Currently no support for ONNX model.\n");
    printf("  --verbose               Use verbose logging (default = false)\n");
    printf("  --engine=<file>         Engine file to serialize to or deserialize from\n");
    printf("  --calib=<file>          Read INT8 calibration cache file.  Currently no support for ONNX model.\n");
    printf("  --useDLACore=N          Specify a DLA engine for layers that support DLA. Value can range from 0 to n-1, where n is the number of DLA engines on the platform.\n");
    printf("  --allowGPUFallback      If --useDLACore flag is present and if a layer can't run on DLA, then run on GPU. \n");
    fflush(stdout);
}

bool parseString(const char* arg, const char* name, std::string& value)
{
    size_t n = strlen(name);
    bool match = arg[0] == '-' && arg[1] == '-' && !strncmp(arg + 2, name, n) && arg[n + 2] == '=';
    if (match)
    {
        value = arg + n + 3;
        std::cout << name << ": " << value << std::endl;
    }
    return match;
}

bool parseInt(const char* arg, const char* name, int& value)
{
    size_t n = strlen(name);
    bool match = arg[0] == '-' && arg[1] == '-' && !strncmp(arg + 2, name, n) && arg[n + 2] == '=';
    if (match)
    {
        value = atoi(arg + n + 3);
        std::cout << name << ": " << value << std::endl;
    }
    return match;
}

bool parseBool(const char* arg, const char* name, bool& value)
{
    size_t n = strlen(name);
    bool match = arg[0] == '-' && arg[1] == '-' && !strncmp(arg + 2, name, n);
    if (match)
    {
        std::cout << name << std::endl;
        value = true;
    }
    return match;
}

bool parseFloat(const char* arg, const char* name, float& value)
{
    size_t n = strlen(name);
    bool match = arg[0] == '-' && arg[1] == '-' && !strncmp(arg + 2, name, n) && arg[n + 2] == '=';
    if (match)
    {
        value = atof(arg + n + 3);
        std::cout << name << ": " << value << std::endl;
    }
    return match;
}

bool parseArgs(int argc, char* argv[])
{
    if (argc < 2)
    {
        printUsage();
        return false;
    }

    for (int j = 1; j < argc; j++)
    {
        if (parseString(argv[j], "model", gParams.modelFile)
            || parseString(argv[j], "deploy", gParams.deployFile)
            || parseString(argv[j], "engine", gParams.engine))
            continue;
        if (parseString(argv[j], "mod", gParams.modelName) )
	    continue;
        if (string(argv[j]) == "-d"){
		gParams.imagePath = argv[j + 1];
		return true;
	}
        if (parseString(argv[j], "uff", gParams.uffFile))
            continue;

        if (parseString(argv[j], "calib", gParams.calibrationCache))
            continue;

        std::string input;
        if (parseString(argv[j], "input", input))
        {
            gParams.inputs.push_back(input);
            continue;
        }

        std::string output;
        if (parseString(argv[j], "output", output))
        {
            gParams.outputs.push_back(output);
            continue;
        }

        std::string uffInput;
        if (parseString(argv[j], "uffInput", uffInput))
        {
            std::vector<std::string> uffInputStrs = split(uffInput, ',');
            if (uffInputStrs.size() != 4)
            {
                printf("Invalid uffInput: %s\n", uffInput.c_str());
                return false;
            }

            gParams.uffInputs.push_back(std::make_pair(uffInputStrs[0], Dims3(atoi(uffInputStrs[1].c_str()), atoi(uffInputStrs[2].c_str()), atoi(uffInputStrs[3].c_str()))));
            continue;
        }

        if (parseInt(argv[j], "batch", gParams.batchSize)
            || parseInt(argv[j], "iterations", gParams.iterations)
            || parseInt(argv[j], "avgRuns", gParams.avgRuns)
            || parseInt(argv[j], "device", gParams.device)
            || parseInt(argv[j], "workspace", gParams.workspaceSize)
            || parseInt(argv[j], "useDLACore", gParams.useDLACore))
            continue;

        if (parseFloat(argv[j], "percentile", gParams.pct))
            continue;

        if (parseBool(argv[j], "fp16", gParams.fp16)
            || parseBool(argv[j], "int8", gParams.int8)
            || parseBool(argv[j], "verbose", gParams.verbose)
            || parseBool(argv[j], "kernel", gParams.kernel)
            || parseBool(argv[j], "allowGPUFallback", gParams.allowGPUFallback))
            continue;

        printf("Unknown argument: %s\n", argv[j]);
        return false;
    }

    return true;
}

static ICudaEngine* createEngine()
{
    ICudaEngine* engine;
    if ((!gParams.deployFile.empty()) || (!gParams.uffFile.empty()) )
    {

        if (!gParams.uffFile.empty())
        {
            engine = uffToTRTModel();
        }
        else
        {
            engine = caffeToTRTModel();
        }

        if (!engine)
        {
            std::cerr << "Engine could not be created" << std::endl;
            return nullptr;
        }

        if (!gParams.engine.empty())
        {
            std::ofstream p(gParams.engine);
            if (!p)
            {
                std::cerr << "could not open plan output file" << std::endl;
                return nullptr;
            }
            IHostMemory* ptr = engine->serialize();
            assert(ptr);
            p.write(reinterpret_cast<const char*>(ptr->data()), ptr->size());
            ptr->destroy();
        }
        return engine;
    }

    // load directly from serialized engine file if deploy not specified
    if (!gParams.engine.empty())
    {
        std::vector<char> trtModelStream;
        size_t size{0};
        std::ifstream file(gParams.engine, std::ios::binary);
        if (file.good())
        {
            file.seekg(0, file.end);
            size = file.tellg();
            file.seekg(0, file.beg);
            trtModelStream.resize(size);
            file.read(trtModelStream.data(), size);
            file.close();
        }

        IRuntime* infer = createInferRuntime(gLogger);
        if (gParams.useDLACore >= 0)
        {
            infer->setDLACore(gParams.useDLACore);
        }

        engine = infer->deserializeCudaEngine(trtModelStream.data(), size, nullptr);

        if (gParams.inputs.empty())
        {
            // Specify input blob name because user has not specified any
            gParams.inputs.push_back("data");
        }

        return engine;
    }

    // complain about empty deploy file
    std::cerr << "Deploy file not specified" << std::endl;
    return nullptr;
}

int main(int argc, char** argv)
{
    // create a TensorRT model from the caffe model and serialize it to a stream

    if (!parseArgs(argc, argv))
        return -1;

    cudaSetDevice(gParams.device);

    if (gParams.outputs.size() == 0 && !gParams.deployFile.empty())
    {
        std::cerr << "At least one network output must be defined" << std::endl;
        return -1;
    }

    initLibNvInferPlugins(&gLogger, "");

    ICudaEngine* engine = createEngine();
    if (!engine)
    {
        std::cerr << "Engine could not be created" << std::endl;
        return -1;
    }

    if (gParams.uffFile.empty() )
        nvcaffeparser1::shutdownProtobufLibrary();
    else if (gParams.deployFile.empty() )
        nvuffparser::shutdownProtobufLibrary();

    if(gParams.kernel){
        doInference_kernel(*engine);
    }else{
        doInference(*engine);
    }
    engine->destroy();

    return 0;
}


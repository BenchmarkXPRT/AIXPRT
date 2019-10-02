#include <iostream>
#include <iterator>
#include <random>
#include <fstream>
#include "NvCaffeParser.h"
#include "NvInfer.h"
#include "NvInferPlugin.h"
#include "NvUtils.h"

#include "BatchStreamPPM.h"
#include <cuda_runtime_api.h>
#include <map>
#include <opencv2/opencv.hpp>
#include <vector>
#include <chrono>
#include <dirent.h>

#include "common.h"
using namespace cv;
using namespace nvinfer1;

int batch_size;
const char* imagePath;
struct Logger : public nvinfer1::ILogger
{
public:
	void log(Severity severity, const char* msg) override
	{
		switch (severity)
		{
		case Severity::kINTERNAL_ERROR:
		case Severity::kERROR:
			std::cerr << "ERROR: " << msg << std::endl;
			break;
		case Severity::kWARNING:
			std::cerr << "WARNING: " << msg << std::endl;
			break;
		case Severity::kINFO:
			break;
		}
	}
};
std::map<std::string, Dims3> gInputDimensions;
//INT8 Calibration, currently set to calibrate over 500 images
static constexpr int CAL_BATCH_SIZE = 50;
static constexpr int FIRST_CAL_BATCH = 0, NB_CAL_BATCHES = 10;
std::string calibrationCache{ "../../../packages/models/resnet-50/CalibrationTable" };

std::string locateFile(const std::string& input)
{
	std::vector<std::string> dirs{ "../../../packages/models/ssd-mobilenet_v1/","../../../packages/COCO_PPM/",
				"data/ssd/",
								  "data/ssd/VOC2007/",
								  "data/ssd/VOC2007/PPMImages/",
								  "data/samples/ssd/",
								  "data/samples/ssd/VOC2007/",
								  "data/samples/ssd/VOC2007/PPMImages/" };
	return locateFile(input, dirs);
}


inline int volume(Dims dims)
{
	return std::accumulate(dims.d, dims.d + dims.nbDims, 1, std::multiplies<int>());
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
		std::cout << length << std::endl;
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

struct TensorRTStreamData
{
	nvinfer1::IExecutionContext* context;
	cudaStream_t stream;
	std::vector<void*> host_buffers;
	std::vector<void*> device_buffers;
};

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

void createMemoryInput(const ICudaEngine& engine, std::vector<void*>& buffers, const std::string& name)
{
	size_t bindingIndex = engine.getBindingIndex(name.c_str());
	printf("name=%s, bindingIndex=%d, buffers.size()=%d\n", name.c_str(), (int)bindingIndex, (int)buffers.size());
	//assert(bindingIndex < buffers.size());
	DimsCHW dimensions = static_cast<DimsCHW&&>(engine.getBindingDimensions((int)bindingIndex));
	size_t eltCount = dimensions.c() * dimensions.h() * dimensions.w() * batch_size, memSize = eltCount * sizeof(float);
	cout << "Elt cnt   " << eltCount << endl;
	float* localMem = new float[eltCount];

	size_t singleImage = dimensions.c() * dimensions.h() * dimensions.w();
	struct dirent** namelist;
	int n;
	n = scandir(imagePath, &namelist, 0, versionsort);
	if (n < 0)
	{
		printf("Invalid arguments..................");
	}
	else {
		int count = 1;
		while (count <= batch_size)
		{
			for (int i = 2; i < n; ++i) {
				if (count > batch_size) break;
				std::string filename = std::string(imagePath) + std::string(namelist[i]->d_name);

				const char* imgFilename = filename.c_str();
				Mat image = imread(imgFilename, IMREAD_COLOR);
				cout << filename << endl;
				printf("Sucessfully loaded image %s\n", imgFilename);
				cvtColor(image, image, cv::COLOR_BGR2RGB, 3);
				cv::resize(image, image, Size(dimensions.w(), dimensions.h()));

				const float mean[3] = { 103.94 , 116.78 ,123.68 };
				const size_t height = image.rows;
				const size_t width = image.cols;
				const size_t channels = image.channels();
				const size_t numel = height * width * channels;
				const size_t stridesCv[3] = { width * channels, channels, 1 };
				const size_t strides[3] = { height * width, width, 1 };
				int map[3] = { 2,1,0 };
				for (int i = 0; i < height; i++)
				{
					for (int j = 0; j < width; j++)
					{
						for (int k = 0; k < channels; k++)
						{
							const size_t offsetCv = i * width * channels + j * channels + map[k];
							const size_t offset = k * strides[0] + i * strides[1] + j * strides[2];
							localMem[((count - 1) * singleImage) + offset] = ((float)image.data[offsetCv] - mean[k]);
						}
					}
				}
				count += 1;
			}
		}
	}
	buffers[bindingIndex] = localMem;
}
int main(int argc, char** argv)
{
	if (argc < 3)
	{
		std::cerr << "Usage: ./tensorrt_bench <prototxt> <caffemode> [streams_count(default=8)]" << std::endl;
		exit(-1);
	}

	std::string model_file = argv[1];
	std::string weights_file = argv[2];
	size_t max_requests_in_fly = argc >= 4 ? static_cast<size_t>(std::atoi(argv[3])) : 8;
	int runs = std::atoi(argv[4]);//Add parameter
	batch_size = argc >= 6 ? std::atoi(argv[5]) : 1;
	imagePath = argc >= 7 ? argv[6] : "images/";
	std::string precision = argc >= 8 ? argv[7] : "fp32";

	cout << "model_file =============" << model_file << endl;
	cout << "weights_file =============" << weights_file << endl;
	cout << "runs =============" << runs << endl;
	cout << "weights_file =============" << weights_file << endl;
	cout << "precision =============" << precision << endl;
	cout << "batch_size =============" << batch_size << endl;
	cout << "image Path =============" << imagePath << endl;
	nvinfer1::DataType data_type = nvinfer1::DataType::kFLOAT;
	int data_type_size = data_type == nvinfer1::DataType::kFLOAT ? 4 : 2;
	//calibrationCache = argc >= 8 ? argv[7] : "";
	std::string output_name = "prob";
	std::string input_name = "data";
	std::vector<int> input_shape = { batch_size, 3, 224, 224 };
	std::vector<int> output_shape = { batch_size, 1000 };

	Logger logger;
	IBuilder* builder = createInferBuilder(logger);
	INetworkDefinition* network = builder->createNetwork();
	nvcaffeparser1::ICaffeParser* parser = nvcaffeparser1::createCaffeParser();

	if (precision == "fp16")
	{
		data_type == nvinfer1::DataType::kHALF;
		if (!builder->platformHasFastFp16()) {
			std::cout << "Platform has no native FP16 support!" << std::endl;
		}
		builder->setFp16Mode(true);
	}
	BatchStream calibrationStream(CAL_BATCH_SIZE, NB_CAL_BATCHES);
	//RndInt8Calibrator calibrator(1, calibrationCache);
	Int8EntropyCalibrator calibrator(calibrationStream, FIRST_CAL_BATCH, calibrationCache);

	if (precision == "int8")
	{
		builder->setInt8Mode(true);
		builder->setInt8Calibrator(&calibrator);
	}

	if (!parser)
		throw std::runtime_error("Can't create Caffe parser!");

	std::cout << "Load network\n";
	const nvcaffeparser1::IBlobNameToTensor* blob_name_to_tensor = parser->parse(model_file.c_str(),
		weights_file.c_str(),
		*network,
		//DataType::kFLOAT);
		data_type);
	if (!blob_name_to_tensor)
		throw std::runtime_error("Can't parse Caffe network!");

	if (network->getNbLayers() == 0)
		throw std::runtime_error("Number of layers is zero!");

	ITensor* output_tensor = blob_name_to_tensor->find(output_name.c_str());
	if (!output_tensor)
		throw std::runtime_error("Can't find output tensor " + output_name);
	network->markOutput(*output_tensor);

	builder->setMaxBatchSize(batch_size);
	builder->setMaxWorkspaceSize(32 * (1 << 20));

	int input_byte_size = input_shape[0] * input_shape[1] * input_shape[2] * input_shape[3] * data_type_size;
	int output_byte_size = output_shape[0] * output_shape[1] * data_type_size;



	nvinfer1::ICudaEngine* engine = builder->buildCudaEngine(*network);
	if (!engine)
		throw std::runtime_error("Can't build engine!");

	parser->destroy();
	network->destroy();
	builder->destroy();

	nvcaffeparser1::shutdownProtobufLibrary();

	std::cout << "Init streams data\n";

	//std::vector<uint8_t> buf(input_byte_size);
	std::vector<void*> buffers(1);
	createMemoryInput(*engine, buffers, "data");

	int input_index = engine->getBindingIndex(input_name.c_str());
	if (input_index == -1)
		throw std::runtime_error("Can't get binding index for blob " + input_name);

	int output_index = engine->getBindingIndex(output_name.c_str());
	if (output_index == -1)
		throw std::runtime_error("Can't get binding index for blob " + output_name);

	cudaError_t rc = cudaSuccess;
	size_t bindings_number = 2;
	std::vector<TensorRTStreamData> streams_data(max_requests_in_fly);
	for (auto& s : streams_data)
	{
		s.context = nullptr;
		s.stream = nullptr;

		s.host_buffers.resize(bindings_number);
		std::fill(s.host_buffers.begin(), s.host_buffers.end(), nullptr);

		s.device_buffers.resize(bindings_number);
		std::fill(s.device_buffers.begin(), s.device_buffers.end(), nullptr);

		rc = cudaMallocHost(&s.host_buffers[input_index], input_byte_size);
		if (rc != cudaSuccess)
			throw std::runtime_error("Allocation failed: " + std::string(cudaGetErrorName(rc)));

		if (!s.context)
		{
			s.context = engine->createExecutionContext();
			if (!s.context)
				throw std::runtime_error("Can't create context!");

			rc = cudaStreamCreate(&s.stream);
			if (rc != cudaSuccess)
				throw std::runtime_error("cudaStreamCreate: " + std::string(cudaGetErrorName(rc)));
		}

		if (s.device_buffers.size() != engine->getNbBindings())
			throw std::runtime_error("Wrong number of bindings: " + std::to_string(engine->getNbBindings()));

		// Allocate inputs memory on device
		if (!s.device_buffers[input_index])
		{
			rc = cudaMalloc(&s.device_buffers[input_index], input_byte_size);
			if (rc != cudaSuccess)
				throw std::runtime_error("Allocation failed: " + std::string(cudaGetErrorName(rc)));
		}

		// Allocate outputs memory on device
		if (!s.device_buffers[output_index])
		{
			rc = cudaMalloc(&s.device_buffers[output_index], output_byte_size);
			if (rc != cudaSuccess)
				throw std::runtime_error("Allocation failed: " + std::string(cudaGetErrorName(rc)));
		}

		// Allocate outputs memory on host
		if (!s.host_buffers[output_index])
		{
			rc = cudaMallocHost(&s.host_buffers[output_index], output_byte_size);
			if (rc != cudaSuccess)
				throw std::runtime_error("Allocation failed: " + std::string(cudaGetErrorName(rc)));
		}

		// Refill data on host
		rc = cudaMemcpyAsync(s.host_buffers[input_index],
			buffers[0],
			input_byte_size,
			cudaMemcpyHostToHost,
			s.stream);
		if (rc != cudaSuccess)
			throw std::runtime_error("HostToHost: " + std::string(cudaGetErrorName(rc)));
	}

	int queued_stream_id = -1;
	int synced_stream_id = -1;

	std::cout << "Run..\n";
	double full_time = 0;
	auto start_time = std::chrono::high_resolution_clock::now();
	chrono::high_resolution_clock::time_point run_start_times[10000];
	std::vector<float> diffs(runs);
	int count = 0;
	for (int i = 0; i < runs; i++)
	{
		queued_stream_id = (int)((queued_stream_id + 1) % streams_data.size());

		auto start_run = std::chrono::high_resolution_clock::now();
		run_start_times[i] = start_run;

		rc = cudaMemcpyAsync(streams_data.at(queued_stream_id).device_buffers[input_index],
			streams_data.at(queued_stream_id).host_buffers[input_index],
			input_byte_size,
			cudaMemcpyHostToDevice,
			streams_data.at(queued_stream_id).stream);
		if (rc != cudaSuccess)
			throw std::runtime_error("HostToDevice: " + std::string(cudaGetErrorName(rc)));

		streams_data.at(queued_stream_id).context->enqueue(batch_size,
			streams_data.at(queued_stream_id).device_buffers.data(),
			streams_data.at(queued_stream_id).stream,
			nullptr);

		rc = cudaMemcpyAsync(streams_data.at(queued_stream_id).host_buffers[output_index],
			streams_data.at(queued_stream_id).device_buffers[output_index],
			output_byte_size,
			cudaMemcpyDeviceToHost,
			streams_data.at(queued_stream_id).stream);
		if (rc != cudaSuccess)
			throw std::runtime_error("DeviceToHost: " + std::string(cudaGetErrorName(rc)));

		if (((synced_stream_id == queued_stream_id) ||
			((synced_stream_id == -1) && (((queued_stream_id + 1) % streams_data.size()) == 0))))
		{
			synced_stream_id = (int)((synced_stream_id + 1) % streams_data.size());
			rc = cudaStreamSynchronize(streams_data.at(synced_stream_id).stream);
			auto end_run = std::chrono::high_resolution_clock::now();
			diffs[count] = (float)std::chrono::duration<double, std::milli>(end_run - run_start_times[count]).count();
			cout << "Start  " << (float)std::chrono::duration_cast<std::chrono::microseconds>(run_start_times[count] - start_time).count(); //<< "   End   " << std::to_string(end_run) << endl; 
			cout << "Times  " << count << "  =  " << diffs[count] << endl;
			count++;
			if (rc != cudaSuccess)
				throw std::runtime_error("Can't synchronize stream " +
					std::to_string(synced_stream_id) +
					std::string(cudaGetErrorName(rc)));
		}

		//auto end_time = std::chrono::high_resolution_clock::now();
		//full_time += (double)std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time).count();
	}


	// Wait for all
	//auto start_time = std::chrono::high_resolution_clock::now();
	while (synced_stream_id != queued_stream_id)
	{
		synced_stream_id = (int)((synced_stream_id + 1) % streams_data.size());
		rc = cudaStreamSynchronize(streams_data.at(synced_stream_id).stream);
		auto end_run = std::chrono::high_resolution_clock::now();
		diffs[count] = (float)std::chrono::duration<double, std::milli>(end_run - run_start_times[count]).count();
		//cout << "Start  "<< std::to_string(run_start_times[count]) << "   End   " << std::to_string(end_run) << endl;
		cout << "Start  " << (float)std::chrono::duration<double, std::milli>(run_start_times[count] - start_time).count(); //<< "   End   " << std::to_string(end_run) << endl; 
		cout << "Times  " << count << "  =  " << diffs[count] << endl;
		count++;
		if (rc != cudaSuccess)
			throw std::runtime_error("Can't synchronize stream " +
				std::to_string(synced_stream_id) +
				std::string(cudaGetErrorName(rc)));

	}
	auto end_time = std::chrono::high_resolution_clock::now();
	float pct50 = percentile(50, diffs);
	float pct90 = percentile(90, diffs);
	float pct95 = percentile(95, diffs);
	float pct99 = percentile(99, diffs);
	double max = *max_element(diffs.begin(), diffs.end());
	double min = *min_element(diffs.begin(), diffs.end());
	std::cout << "50 \% percentile time is " << pct50 << ")." << std::endl;
	std::cout << "90 \% percentile time is " << pct90 << ")." << std::endl;
	std::cout << "95 \% percentile time is " << pct95 << ")." << std::endl;
	std::cout << "99 \% percentile time is " << pct99 << ")." << std::endl;
	full_time = (double)std::chrono::duration<double, std::milli>(end_time - start_time).count();

	std::cout << "FULL TIME: " << full_time << std::endl;
	std::cout << "FPS: " << (1000.0) / (full_time / runs) << std::endl;

	std::string const command = "python3 cpp_to_python_api.py resnet-50 " + std::to_string(batch_size) + " gpu " + precision + " NA NA " + std::to_string((1000.0 * batch_size) / (full_time / runs)) + " " + std::to_string(full_time / (runs)) + " ImagesPerSec msec " + std::to_string(runs) + " ILSVRC-2012 " + std::to_string(pct50) + " " + std::to_string(pct90) + " " + std::to_string(pct95) + " " + std::to_string(pct99) + " " + std::to_string(max) + " " + std::to_string(min) + " " + std::to_string(max_requests_in_fly);
	if (system(command.c_str()) == 0) // c++ system call returns 0 if successful
	{
		std::cout << "Execution Completed!" << std::endl;
	}
	else {
		std::cout << "Execution Unsuccessfull!" << std::endl;
	}
	return 0;
}


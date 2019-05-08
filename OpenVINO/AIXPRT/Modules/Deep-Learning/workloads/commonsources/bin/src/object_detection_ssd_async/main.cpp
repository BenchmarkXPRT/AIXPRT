/*
// Copyright (c) 2018 Intel Corporation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
*/

#include <gflags/gflags.h>
#include <functional>
#include <iostream>
#include <fstream>
#include <random>
#include <string>
#include <memory>
#include <vector>
#include <time.h>
#include <limits>
#include <chrono>
#include <algorithm>

#include <format_reader_ptr.h>
#include <inference_engine.hpp>
#include <ext_list.hpp>

#include <samples/common.hpp>
#include <samples/slog.hpp>
#include <samples/args_helper.hpp>
#include "object_detection_ssd_async.h"
#include <vpu/vpu_plugin_config.hpp>

using namespace InferenceEngine;

ConsoleErrorListener error_listener;

bool ParseAndCheckCommandLine(int argc, char *argv[]) {
    // ---------------------------Parsing and validation of input args--------------------------------------
    gflags::ParseCommandLineNonHelpFlags(&argc, &argv, true);
    if (FLAGS_h) {
        showUsage();
        return false;
    }

    slog::info << "Parsing input parameters" << slog::endl;

    if (FLAGS_ni < 1) {
        throw std::logic_error("Parameter -ni should be greater than 0 (default: 1)");
    }

    if (FLAGS_i.empty()) {
        throw std::logic_error("Parameter -i is not set");
    }

    if (FLAGS_m.empty()) {
        throw std::logic_error("Parameter -m is not set");
    }

    if (FLAGS_nireq < 1) {
        throw std::logic_error("Parameter -nireq must be more than 0 ! (default 1)");
    }

    if (FLAGS_ni < FLAGS_nireq) {
        throw std::logic_error("Number of iterations could not be less than requests quantity");
    }

    return true;
}

//0 =======================================================================================================================================================================

// -------------------------- Compute percentile -----------------------------------------------------------------

double computePercentile( std::vector<double> arr, int perc){

   float loc = (float(perc)/100.0)*arr.size() - 1.0; // Index starts from zero
   int l = static_cast<int> (floor(loc));
   int h = static_cast<int> (ceil(loc));
   
   double lower = arr[l];
   double upper = arr[h];
   double value = lower + (upper - lower)*(loc - l);
   return value;

}
//0=======================================================================================================================================================================

// -------------------------- Main -----------------------------------------------------------------

int main(int argc, char *argv[]) {
    try {
        /** This sample covers certain topology and cannot be generalized for any object detection one **/
        slog::info << "InferenceEngine: " << GetInferenceEngineVersion() << "\n";

        // --------------------------- 1. Parsing and validation of input args ---------------------------------
        if (!ParseAndCheckCommandLine(argc, argv)) {
            return 0;
        }
        // -----------------------------------------------------------------------------------------------------

        // --------------------------- 2. Read input -----------------------------------------------------------
        /** This vector stores paths to the processed images **/
        std::vector<std::string> images;
        parseInputFilesArguments(images);
        if (images.empty()) throw std::logic_error("No suitable images were found");
        // -----------------------------------------------------------------------------------------------------

        if (FLAGS_d.find("MYRIAD") != std::string::npos){
            slog::info << " == MYRIAD == " << "iterations: " << FLAGS_ni << " request: " << FLAGS_nireq << slog::endl;
        }
        if (FLAGS_d.find("HDDL") != std::string::npos){
            //FLAGS_nireq = FLAGS_nireq * 50; // automatically add some more requests to fullfill extra HW accelerators
            slog::info << " == HDDL == " << "iterations: " << FLAGS_ni << " request : " << FLAGS_nireq << slog::endl;
        }

        // --------------------------- 3. Load Plugin for inference engine -------------------------------------
        slog::info << "Loading plugin" << slog::endl;
        InferencePlugin plugin = PluginDispatcher({ FLAGS_pp, "../../../lib/intel64", "" }).getPluginByDevice(FLAGS_d);
        //InferencePlugin plugin = PluginDispatcher({ FLAGS_pp, "" }).getPluginByDevice(FLAGS_d);
        if (FLAGS_p_msg) {
            static_cast<InferenceEngine::InferenceEnginePluginPtr>(plugin)->SetLogCallback(error_listener);
        }

        /*If CPU device, load default library with extensions that comes with the product*/
        if (FLAGS_d.find("CPU") != std::string::npos) {
            /**
            * cpu_extensions library is compiled from "extension" folder containing
            * custom MKLDNNPlugin layer implementations. These layers are not supported
            * by mkldnn, but they can be useful for inferring custom topologies.
            **/
            plugin.AddExtension(std::make_shared<Extensions::Cpu::CpuExtensions>());
        }

        if (!FLAGS_l.empty()) {
            // CPU(MKLDNN) extensions are loaded as a shared library and passed as a pointer to base extension
            IExtensionPtr extension_ptr = make_so_pointer<IExtension>(FLAGS_l);
            plugin.AddExtension(extension_ptr);
            slog::info << "CPU Extension loaded: " << FLAGS_l << slog::endl;
        }

        if (!FLAGS_c.empty()) {
            // clDNN Extensions are loaded from an .xml description and OpenCL kernel files
            plugin.SetConfig({ { PluginConfigParams::KEY_CONFIG_FILE, FLAGS_c } });
            slog::info << "GPU Extension loaded: " << FLAGS_c << slog::endl;
        }

        /** Setting plugin parameter for per layer metrics **/
        if (FLAGS_pc) {
            plugin.SetConfig({ { PluginConfigParams::KEY_PERF_COUNT, PluginConfigParams::YES } });
        }

        ResponseDesc resp;
        /** Printing plugin version **/
        printPluginVersion(plugin, std::cout);
        // -----------------------------------------------------------------------------------------------------

        // --------------------------- 4. Read IR Generated by ModelOptimizer (.xml and .bin files) ------------
        std::string binFileName = fileNameNoExt(FLAGS_m) + ".bin";
        slog::info << "Loading network files:"
            "\n\t" << FLAGS_m <<
            "\n\t" << binFileName <<
            slog::endl;

        CNNNetReader networkReader;
        /** Read network model **/
        networkReader.ReadNetwork(FLAGS_m);

        /** Extract model name and load weights **/
        networkReader.ReadWeights(binFileName);
        CNNNetwork network = networkReader.getNetwork();
        network.setBatchSize(images.size());
        // -----------------------------------------------------------------------------------------------------
        //Adding harness variables
        std::string model_name = FLAGS_a, aarch = FLAGS_aarch, precision = FLAGS_prec, output_dir = FLAGS_dir;
        int batch_size = FLAGS_b;

        std::cout << "[ INFO ] " << model_name << "\t" << precision <<
            "\t" << batch_size << "\t" << output_dir << std::endl;
        // --------------------------- 5. Prepare input blobs --------------------------------------------------
        slog::info << "Preparing input blobs" << slog::endl;

        /** Taking information about all topology inputs **/
        InputsDataMap inputsInfo(network.getInputsInfo());

        /** SSD network has one input and one output **/
        if (inputsInfo.size() != 1 && inputsInfo.size() != 2) throw std::logic_error("Sample supports topologies only with 1 or 2 inputs");

        /**
         * Some networks have SSD-like output format (ending with DetectionOutput layer), but
         * having 2 inputs as Faster-RCNN: one for image and one for "image info".
         *
         * Although object_datection_sample_ssd's main task is to support clean SSD, it could score
         * the networks with two inputs as well. For such networks imInfoInputName will contain the "second" input name.
         */
        std::string imageInputName, imInfoInputName;

        InputInfo::Ptr inputInfo = inputsInfo.begin()->second;

        SizeVector inputImageDims;
        /** Stores input image **/

        /** Iterating over all input blobs **/
        for (auto & item : inputsInfo) {
            /** Working with first input tensor that stores image **/
            if (item.second->getInputData()->getTensorDesc().getDims().size() == 4) {
                imageInputName = item.first;

                slog::info << "Network Batch size is " << std::to_string(networkReader.getNetwork().getBatchSize()) << slog::endl;

                /** Creating first input blob **/
                Precision inputPrecision = Precision::U8;
                item.second->setPrecision(inputPrecision);
            } else if (item.second->getInputData()->getTensorDesc().getDims().size() == 2) {
                imInfoInputName = item.first;

                Precision inputPrecision = Precision::FP32;
                item.second->setPrecision(inputPrecision);
                if ((item.second->getTensorDesc().getDims()[1] != 3 && item.second->getTensorDesc().getDims()[1] != 6) ||
                     item.second->getTensorDesc().getDims()[0] != 1) {
                    throw std::logic_error("Invalid input info. Should be 3 or 6 values length");
                }
            }
        }
        // -----------------------------------------------------------------------------------------------------

        // --------------------------- 6. Prepare output blobs -------------------------------------------------
        slog::info << "Preparing output blobs" << slog::endl;

        OutputsDataMap outputsInfo(network.getOutputsInfo());

        std::string outputName;
        DataPtr outputInfo;
        for (const auto& out : outputsInfo) {
            if (out.second->creatorLayer.lock()->type == "DetectionOutput") {
                outputName = out.first;
                outputInfo = out.second;
            }
        }

        if (outputInfo == nullptr) {
            throw std::logic_error("Can't find a DetectionOutput layer in the topology");
        }

        const SizeVector outputDims = outputInfo->getTensorDesc().getDims();

        const int maxProposalCount = outputDims[2];
        const int objectSize = outputDims[3];

        if (objectSize != 7) {
            throw std::logic_error("Output item should have 7 as a last dimension");
        }

        if (outputDims.size() != 4) {
            throw std::logic_error("Incorrect output dimensions for SSD model");
        }

        /** Set the precision of output data provided by the user, should be called before load of the network to the plugin **/
        outputInfo->setPrecision(Precision::FP32);

        std::vector <Blob::Ptr> outputBlobs;
        for (size_t i = 0; i < FLAGS_nireq; i++) {
            auto outputBlob = make_shared_blob<PrecisionTrait<Precision::FP32>::value_type>(outputInfo->getTensorDesc());
            outputBlob->allocate();
            outputBlobs.push_back(outputBlob);
        }
        // -----------------------------------------------------------------------------------------------------

        // --------------------------- 7. Loading model to the plugin ------------------------------------------
        slog::info << "Loading model to the plugin" << slog::endl;
        //set config values
        std::map<std::string, std::string> config;
        if ((FLAGS_d.find("MYRIAD") != std::string::npos) || (FLAGS_d.find("HDDL") != std::string::npos)) {
          config[VPU_CONFIG_KEY(LOG_LEVEL)] = CONFIG_VALUE(LOG_WARNING);
          config[CONFIG_KEY(LOG_LEVEL)] = CONFIG_VALUE(LOG_WARNING);
          config[VPU_CONFIG_KEY(HW_STAGES_OPTIMIZATION)] = CONFIG_VALUE(YES); //This is the important one for HW acceleration
          config[VPU_CONFIG_KEY(PRINT_RECEIVE_TENSOR_TIME)] = CONFIG_VALUE(YES);
          //config[VPU_CONFIG_KEY(INTERNAL_PROFILING)] = CONFIG_VALUE(NO);
          //HDDL networkConfig[InferenceEngine::PluginConfigParams::KEY_LOG_LEVEL] = InferenceEngine::PluginConfigParams::LOG_INFO;
          //HDDL networkConfig["VPU_RESHAPE_OPTIMIZATION"] = CONFIG_VALUE(NO);
        }
        if (FLAGS_d.find("CPU") != std::string::npos) {  // CPU supports few special performance-oriented keys
            // limit threading for CPU portion of inference
            config[PluginConfigParams::KEY_CPU_THREADS_NUM] = std::to_string(FLAGS_nthreads);
            // pin threads for CPU portion of inference
            config[PluginConfigParams::KEY_CPU_BIND_THREAD] = FLAGS_pin;
            // for pure CPU execution, more throughput-oriented execution via streams
            if (FLAGS_d == "CPU")
                config[PluginConfigParams::KEY_CPU_THROUGHPUT_STREAMS] = std::to_string(FLAGS_nireq);
        }
        ExecutableNetwork executable_network = plugin.LoadNetwork(network, config);
        // -----------------------------------------------------------------------------------------------------

        // --------------------------- 8. Create infer request -------------------------------------------------
        std::vector<InferRequest> inferRequests;
        for (size_t i = 0; i < FLAGS_nireq; i++) {
            InferRequest infer_request = executable_network.CreateInferRequest();
            inferRequests.push_back(infer_request);
        }
        // -----------------------------------------------------------------------------------------------------

        // --------------------------- 9. Prepare input --------------------------------------------------------
        /** Stores all input blobs data **/
        BlobMap inputBlobs;

        /** Collect images data ptrs **/
        std::vector<std::shared_ptr<unsigned char>> imagesData, originalImagesData;
        std::vector<int> imageWidths, imageHeights;
        for (auto & i : images) {
            FormatReader::ReaderPtr reader(i.c_str());
            if (reader.get() == nullptr) {
                slog::warn << "Image " + i + " cannot be read!" << slog::endl;
                continue;
            }
            /** Store image data **/
            std::shared_ptr<unsigned char> originalData(reader->getData());
            std::shared_ptr<unsigned char> data(reader->getData(inputInfo->getTensorDesc().getDims()[3], inputInfo->getTensorDesc().getDims()[2]));
            if (data.get() != nullptr) {
                originalImagesData.push_back(originalData);
                imagesData.push_back(data);
                imageWidths.push_back(reader->width());
                imageHeights.push_back(reader->height());

                auto input = make_shared_blob<PrecisionTrait<Precision::U8>::value_type>(inputInfo->getTensorDesc());
                input->allocate();
                inputBlobs[i] = input;

            }
        }
        if (imagesData.empty()) throw std::logic_error("Valid input images were not found!");

        size_t batchSize = network.getBatchSize();
        slog::info << "Network Batch size is " << std::to_string(batchSize) << slog::endl;
        if (batchSize != imagesData.size()) {
            slog::warn << "Number of images " + std::to_string(imagesData.size()) + \
                " doesn't match batch size " + std::to_string(batchSize) << slog::endl;
            batchSize = std::min(batchSize, imagesData.size());
            slog::warn << "Number of images to be processed is "<< std::to_string(batchSize) << slog::endl;
        }

        /** Creating input blob **/
        for (size_t i = 0; i < FLAGS_nireq; i++) {
            Blob::Ptr imageInput = inferRequests[i].GetBlob(imageInputName);

            /** Filling input tensor with images. First b channel, then g and r channels **/
            size_t num_channels = imageInput->getTensorDesc().getDims()[1];
            size_t image_size = imageInput->getTensorDesc().getDims()[3] * imageInput->getTensorDesc().getDims()[2];

            unsigned char* data = static_cast<unsigned char*>(imageInput->buffer());
            /** Iterate over all input images **/
            for (size_t image_id = 0; image_id < std::min(imagesData.size(), batchSize); ++image_id) {
                /** Iterate over all pixel in image (b,g,r) **/
                for (size_t pid = 0; pid < image_size; pid++) {
                    /** Iterate over all channels **/
                    for (size_t ch = 0; ch < num_channels; ++ch) {
                        /**          [images stride + channels stride + pixel id ] all in bytes            **/
                        data[image_id * image_size * num_channels + ch * image_size + pid] = imagesData.at(image_id).get()[pid*num_channels + ch];
                    }
                }
            }

            if (imInfoInputName != "") {
                Blob::Ptr input2 = inferRequests[i].GetBlob(imInfoInputName);
                auto imInfoDim = inputsInfo.find(imInfoInputName)->second->getTensorDesc().getDims()[1];

                /** Fill input tensor with values **/
                float *p = input2->buffer().as<PrecisionTrait<Precision::FP32>::value_type*>();

                for (size_t image_id = 0; image_id < std::min(imagesData.size(), batchSize); ++image_id) {
                    p[image_id * imInfoDim + 0] = static_cast<float>(inputsInfo[imageInputName]->getTensorDesc().getDims()[2]);
                    p[image_id * imInfoDim + 1] = static_cast<float>(inputsInfo[imageInputName]->getTensorDesc().getDims()[3]);
                    for (int k = 2; k < imInfoDim; k++) {
                        p[image_id * imInfoDim + k] = 1.0f;  // all scale factors are set to 1.0
                    }
                }
            }
        }

        // -----------------------------------------------------------------------------------------------------

        // --------------------------- 10. Do inference ---------------------------------------------------------
        slog::info << "Start inference (" << FLAGS_ni << " iterations)" << slog::endl;

        typedef std::chrono::high_resolution_clock Time;
        typedef std::chrono::duration<double, std::ratio<1, 1000>> ms;
        typedef std::chrono::duration<float> fsec;

//1 =======================================================================================================================================================================

        /** warmup the inference engine **/

        size_t currentInfer = 0;
        size_t prevInfer = (FLAGS_nireq > 1) ? 1 : 0;

	auto t_ = Time::now();	
	std::map < int, std::vector<decltype(t_)> > req_start_times;
        std::map < int, std::vector<double> > req_total_times;
        ms req_d;
	std::chrono::duration<double> st_d;

// -------------------------------------------------------------------------------------------------------------------------------------	
	slog::info << " Initializing request times "<<slog::endl;
        for (int i=0; i < FLAGS_nireq; i++)
        {
           std::vector<double> times_vec = {};
	   //std::vector<double> st_vec = {};

           std::vector<decltype(t_)> runtime_vec {Time::now()};

	   req_start_times[i] = runtime_vec;
	   req_total_times[i] = times_vec;
        }

// -------------------------------------------------------------------------------------------------------------------------------------
        double total = 0.0;
        /** Start inference & calc performance **/
	std::vector<double> inference_times_array = {};// All inference request times
        slog::info << "Starting iterations"<< slog::endl;

        auto t0 = Time::now();
        for (int iter = 0; iter < FLAGS_ni + FLAGS_nireq; ++iter) {
            if (iter < FLAGS_ni) {

                req_start_times[currentInfer].push_back(Time::now());
                inferRequests[currentInfer].StartAsync();
            }

            inferRequests[prevInfer].Wait(InferenceEngine::IInferRequest::WaitMode::RESULT_READY);
            req_d = std::chrono::duration_cast<ms>( Time::now() - req_start_times[prevInfer].back() ); // Time it took for prevInfer inference
            req_total_times[prevInfer].push_back(req_d.count());

            currentInfer++;
            if (currentInfer >= FLAGS_nireq) {
                currentInfer = 0;
            }
            prevInfer++;
            if (prevInfer >= FLAGS_nireq) {
                prevInfer = 0;
            }
        }

// -------------------------------------------------------------------------------------------------------------------------------------
        auto t1 = Time::now();
        fsec fs = t1 - t0;
        ms d = std::chrono::duration_cast<ms>(fs);
        total = d.count();

// -------------------------------------------------------------------------------------------------------------------------------------
	/** Populate inference_times_array **/
        for (size_t i = 0; i < FLAGS_nireq; i++ ){
            slog::info << "============== Times for request "<< i <<" ==================== " << slog::endl;
            int idx = 0;
            for ( auto it = req_total_times[i].cbegin()+1; it != req_total_times[i].cend(); ++it){
                    inference_times_array.push_back(*it); 
                    slog::info <<"Request " << i << " " << idx << " : " << *it << slog::endl;
                idx++;
	    }
        }

// -------------------------------------------------------------------------------------------------------------------------------------

        /** Compute Percentiles **/
	std::sort (inference_times_array.begin(), inference_times_array.end()); // Sort the times
        double max_time= inference_times_array.back();
	double perc_99 = computePercentile( inference_times_array, 99);
	double perc_95 = computePercentile( inference_times_array, 95);
        double perc_90 = computePercentile( inference_times_array, 90);
        double perc_50 = computePercentile( inference_times_array, 50);
        double min_time= inference_times_array[0];

// -------------------------------------------------------------------------------------------------------------------------------------
        std::cout << "total: " << total << std::endl;
        /** Show performance results **/
        std::cout << std::endl << "Throughput: " << 1000 * static_cast<double>(FLAGS_ni) * batchSize / total << " FPS" << std::endl;
        std::cout << std::endl;

        double avg_time = 0.0;
        avg_time = total / static_cast<double>(FLAGS_ni);
        double imgpersec = 0.0;
        imgpersec =  1000 * static_cast<double>(FLAGS_ni) * batchSize / total;
        std::cout << "Result: " << imgpersec << " images/sec" << std::endl;

        //double standard_deviation = 0.0; // not possible to compute
	std::string standard_deviation = "Undefined";
        std::string const command = "python3 cpp_to_python_api.py "  + model_name +" " + std::to_string(batch_size) + " " + aarch + " " + precision + " " + std::to_string(imgpersec) +\
                                     " " + std::to_string(FLAGS_ni) + " " + std::to_string(avg_time) + " " + standard_deviation + " " + std::to_string(FLAGS_nireq) +\
					" " + std::to_string(perc_99) + " " + std::to_string(perc_95) + " " + std::to_string(perc_90) + " " + std::to_string(perc_50) +\
					" " + std::to_string(min_time) + " " + std::to_string(max_time);

//1 =======================================================================================================================================================================

        std::cout << command << std::endl;
        int ret_val = system(command.c_str());
        if (ret_val == 0)
        {
          std::cout << "Inference Successful" << std::endl;
        } else {
          std::cout << "Error Running inference" << std::endl;
          std::cout << "     "<< command << " RETURNS: " << std::to_string(ret_val) <<std::endl;
        }
	
        // --------------------------- 11. Process output -------------------------------------------------------
        slog::info << "Processing output blobs" << slog::endl;

        for (int iter = 0; iter < FLAGS_nireq; ++iter) {

          const Blob::Ptr output_blob = inferRequests[iter].GetBlob(outputName);
          const float* detection = static_cast<PrecisionTrait<Precision::FP32>::value_type*>(output_blob->buffer());

          std::vector<std::vector<int> > boxes(batchSize);
          std::vector<std::vector<int> > classes(batchSize);

          /* Each detection has image_id that denotes processed image */
          for (int curProposal = 0; curProposal < maxProposalCount; curProposal++) {
              float image_id = detection[curProposal * objectSize + 0];
              /* CPU and GPU plugins have difference in DetectionOutput layer, so we need both checks */
              if (image_id < 0) {
                break;
            }

            float label = detection[curProposal * objectSize + 1];
            float confidence = detection[curProposal * objectSize + 2];
            float xmin = detection[curProposal * objectSize + 3] * imageWidths[image_id];
            float ymin = detection[curProposal * objectSize + 4] * imageHeights[image_id];
            float xmax = detection[curProposal * objectSize + 5] * imageWidths[image_id];
            float ymax = detection[curProposal * objectSize + 6] * imageHeights[image_id];

            if (confidence > 0.5) {
                /** Drawing only objects with >50% probability **/
                classes[image_id].push_back(static_cast<int>(label));
                boxes[image_id].push_back(static_cast<int>(xmin));
                boxes[image_id].push_back(static_cast<int>(ymin));
                boxes[image_id].push_back(static_cast<int>(xmax - xmin));
                boxes[image_id].push_back(static_cast<int>(ymax - ymin));
                std::cout << " WILL BE PRINTED!";
            }
            std::cout << std::endl;
        }

          for (size_t batch_id = 0; batch_id < batchSize; ++batch_id) {
              addRectangles(originalImagesData[batch_id].get(), imageHeights[batch_id], imageWidths[batch_id], boxes[batch_id], classes[batch_id]);
              const std::string image_path = "../../"+output_dir+"/result/output/out_"+model_name+std::to_string(batch_id)+".bmp";
              //const std::string image_path = std::to_string(batch_id)+".bmp";
              if (writeOutputBmp(image_path, originalImagesData[batch_id].get(), imageHeights[batch_id], imageWidths[batch_id])) {
                  slog::info << "Image " + image_path + " created!" << slog::endl;
              } else {
                  throw std::logic_error(std::string("Can't create a file: ") + image_path);
              }
          }
        }
        // -----------------------------------------------------------------------------------------------------
        std::cout << std::endl << "total inference time: " << total << std::endl;
        std::cout << "Average running time of one iteration: " << total / static_cast<double>(FLAGS_ni) << " ms" << std::endl;
        std::cout << std::endl << "Throughput: " << 1000 * static_cast<double>(FLAGS_ni) * batchSize / total << " FPS" << std::endl;
        std::cout << std::endl;

        // -----------------------------------------------------------------------------------------------------
    }
    catch (const std::exception& error) {
        slog::err << error.what() << slog::endl;
        return 1;
    }
    catch (...) {
        slog::err << "Unknown/internal exception happened." << slog::endl;
        return 1;
    }

    slog::info << "Execution successful" << slog::endl;
    return 0;
}

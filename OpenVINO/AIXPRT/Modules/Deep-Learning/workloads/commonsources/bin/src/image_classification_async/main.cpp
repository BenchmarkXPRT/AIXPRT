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

/**
* @brief The entry point the Inference Engine sample application
* @file classification_sample/main.cpp
* @example classification_sample/main.cpp
*/

#include <fstream>
#include <iomanip>
#include <vector>
#include <chrono>
#include <memory>
#include <string>
#include <map>

#include <inference_engine.hpp>

#include <format_reader_ptr.h>

#include <samples/common.hpp>
#include <samples/slog.hpp>
#include <samples/args_helper.hpp>

#include <sys/stat.h>
#include <ext_list.hpp>

#include "image_classification_async.h"
#include <vpu/vpu_plugin_config.hpp>

using namespace InferenceEngine;

ConsoleErrorListener error_listener;

bool ParseAndCheckCommandLine(int argc, char *argv[]) {
    // ---------------------------Parsing and validation of input args--------------------------------------
    slog::info << "Parsing input parameters" << slog::endl;

    gflags::ParseCommandLineNonHelpFlags(&argc, &argv, true);
    if (FLAGS_h) {
        showUsage();
        return false;
    }
    slog::info << "Parsing input parameters" << slog::endl;

    if (FLAGS_ni < 1) {
        throw std::logic_error("Parameter -ni must be more than 0 ! (default 1)");
    }

    if (FLAGS_nireq < 1) {
        throw std::logic_error("Parameter -nireq must be more than 0 ! (default 1)");
    }

    if (FLAGS_i.empty()) {
        throw std::logic_error("Parameter -i is not set");
    }

    if (FLAGS_m.empty()) {
        throw std::logic_error("Parameter -m is not set");
    }

    if (FLAGS_ni < FLAGS_nireq) {
        throw std::logic_error("Number of iterations could not be less than requests quantity");
    }

    return true;
}

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

int main(int argc, char *argv[]) {
    try {
        slog::info << "InferenceEngine: " << GetInferenceEngineVersion() << slog::endl;

        // ------------------------------ Parsing and validation of input args ---------------------------------
        slog::info << "Parsing input parameters " << slog::endl;
        std::string model_name,aarch,precision;
        int batch_size = 0;

        for (int i = 0; i < argc; i++) {
            std::string word = argv[i];
            if (word == "-a") {
                // We know the next argument *should* be the filename:
                model_name = argv[i + 1];
            }else if(word == "-aarch"){
                aarch = argv[i + 1];
            }else if(word == "-b"){
                batch_size = atoi(argv[i + 1]);
            }else if(word == "-prec"){
                precision = argv[i + 1];
            }
       	}

        if (!ParseAndCheckCommandLine(argc, argv)) {
            return 0;
        }

        /** This vector stores paths to the processed images **/
        std::vector<std::string> imageNames;
        parseInputFilesArguments(imageNames);
        if (imageNames.empty()) throw std::logic_error("No suitable images were found");
        // -----------------------------------------------------------------------------------------------------

        if (FLAGS_d.find("MYRIAD") != std::string::npos){
            slog::info << " == MYRIAD == " << "iterations: " << FLAGS_ni << " request: " << FLAGS_nireq << slog::endl;
        }
        if (FLAGS_d.find("HDDL") != std::string::npos){
            //FLAGS_nireq = FLAGS_nireq * 50; // automatically add some more requests to fullfill extra HW accelerators
            slog::info << " == HDDL == " << "iterations: " << FLAGS_ni << " request : " << FLAGS_nireq << slog::endl;
        }

        // --------------------------- 1. Load Plugin for inference engine -------------------------------------
        slog::info << "Loading plugin" << slog::endl;
        InferencePlugin plugin = PluginDispatcher({ FLAGS_pp, "../../../lib/intel64" , "" }).getPluginByDevice(FLAGS_d);
        if (FLAGS_p_msg) {
            static_cast<InferenceEngine::InferenceEnginePluginPtr>(plugin)->SetLogCallback(error_listener);
        }

        /** Loading default extensions **/
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
            plugin.SetConfig({{PluginConfigParams::KEY_CONFIG_FILE, FLAGS_c}});
            slog::info << "GPU Extension loaded: " << FLAGS_c << slog::endl;
        }

        ResponseDesc resp;
        /** Printing plugin version **/
        printPluginVersion(plugin, std::cout);
        // -----------------------------------------------------------------------------------------------------

        // --------------------------- 2. Read IR Generated by ModelOptimizer (.xml and .bin files) ------------
        slog::info << "Loading network files" << slog::endl;

        CNNNetReader networkReader;
        /** Read network model **/
        networkReader.ReadNetwork(FLAGS_m);

        /** Extract model name and load weights **/
        std::string binFileName = fileNameNoExt(FLAGS_m) + ".bin";
        networkReader.ReadWeights(binFileName);

        CNNNetwork network = networkReader.getNetwork();
        // -----------------------------------------------------------------------------------------------------

        // --------------------------- 3. Configure input & output ---------------------------------------------

        // --------------------------- Prepare input blobs -----------------------------------------------------
        slog::info << "Preparing input blobs" << slog::endl;

        /** Taking information about all topology inputs **/
        InputsDataMap inputInfo(network.getInputsInfo());
        if (inputInfo.size() != 1) throw std::logic_error("Sample supports topologies only with 1 input");

        auto inputInfoItem = *inputInfo.begin();

        /** Specifying the precision and layout of input data provided by the user.
         * This should be called before load of the network to the plugin **/
        inputInfoItem.second->setPrecision(Precision::U8);
        inputInfoItem.second->setLayout(Layout::NCHW);

        std::vector<std::shared_ptr<unsigned char>> imagesData;
        for (auto & i : imageNames) {
            FormatReader::ReaderPtr reader(i.c_str());
            if (reader.get() == nullptr) {
                slog::warn << "Image " + i + " cannot be read!" << slog::endl;
                continue;
            }
            /** Store image data **/
            std::shared_ptr<unsigned char> data(
                    reader->getData(inputInfoItem.second->getTensorDesc().getDims()[3],
                                    inputInfoItem.second->getTensorDesc().getDims()[2]));
            if (data.get() != nullptr) {
                imagesData.push_back(data);
            }
        }
        if (imagesData.empty()) throw std::logic_error("Valid input images were not found!");

        /** Setting batch size using image count **/
        network.setBatchSize(imagesData.size());
        size_t batchSize = network.getBatchSize();
        slog::info << "Batch size is " << std::to_string(batchSize) << slog::endl;

        // ------------------------------ Prepare output blobs -------------------------------------------------
        slog::info << "Preparing output blobs" << slog::endl;

        OutputsDataMap outputInfo(network.getOutputsInfo());
        std::vector <Blob::Ptr> outputBlobs;
        for (size_t i = 0; i < FLAGS_nireq; i++) {
            auto outputBlob = make_shared_blob<PrecisionTrait<Precision::FP32>::value_type>(outputInfo.begin()->second->getTensorDesc());
            outputBlob->allocate();
            outputBlobs.push_back(outputBlob);
        }
        // -----------------------------------------------------------------------------------------------------

        // --------------------------- 4. Loading model to the plugin ------------------------------------------
        slog::info << "Loading model to the plugin" << slog::endl;

        std::map<std::string, std::string> config;
        if (FLAGS_pc) {
            config[PluginConfigParams::KEY_PERF_COUNT] = PluginConfigParams::YES;
        }

        // set config values
        // Enable HW acceleration
        if ((FLAGS_d.find("MYRIAD") != std::string::npos) or (FLAGS_d.find("HDDL") != std::string::npos)) {
            config[VPU_CONFIG_KEY(LOG_LEVEL)] = CONFIG_VALUE(LOG_WARNING);
            config[CONFIG_KEY(LOG_LEVEL)] = CONFIG_VALUE(LOG_WARNING);
            config[VPU_CONFIG_KEY(HW_STAGES_OPTIMIZATION)] = CONFIG_VALUE(YES); //This is the important one for HW acceleration
            config[VPU_CONFIG_KEY(PRINT_RECEIVE_TENSOR_TIME)] = CONFIG_VALUE(YES);
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

        // --------------------------- 5. Create infer request -------------------------------------------------
        std::vector<InferRequest> inferRequests;
        for (size_t i = 0; i < FLAGS_nireq; i++) {
            InferRequest inferRequest = executable_network.CreateInferRequest();
            inferRequests.push_back(inferRequest);
        }
        // -----------------------------------------------------------------------------------------------------

        // --------------------------- 6. Prepare input --------------------------------------------------------
        BlobMap inputBlobs;
        for (auto & item : inputInfo) {
            auto input = make_shared_blob<PrecisionTrait<Precision::U8>::value_type>(item.second->getTensorDesc());
            input->allocate();
            inputBlobs[item.first] = input;

            auto dims = input->getTensorDesc().getDims();
            /** Fill input tensor with images. First b channel, then g and r channels **/
            size_t num_channels = dims[1];
            size_t image_size = dims[3] * dims[2];

            /** Iterate over all input images **/
            for (size_t image_id = 0; image_id < imagesData.size(); ++image_id) {
                /** Iterate over all pixel in image (b,g,r) **/
                for (size_t pid = 0; pid < image_size; pid++) {
                    /** Iterate over all channels **/
                    for (size_t ch = 0; ch < num_channels; ++ch) {
                        /**          [images stride + channels stride + pixel id ] all in bytes            **/
                        input->data()[image_id * image_size * num_channels + ch * image_size + pid] = imagesData.at(image_id).get()[pid*num_channels + ch];
                    }
                }
            }
        }

        for (size_t i = 0; i < FLAGS_nireq; i++) {
            inferRequests[i].SetBlob(inputBlobs.begin()->first, inputBlobs.begin()->second);
            inferRequests[i].SetBlob(outputInfo.begin()->first, outputBlobs[i]);
        }
        // -----------------------------------------------------------------------------------------------------

        // --------------------------- 7. Do inference ---------------------------------------------------------
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
        std::string const command = "python cpp_to_python_api.py "  + model_name +" " + std::to_string(batch_size) + " " + aarch + " " + precision + " " + std::to_string(imgpersec) +\
                                     " " + std::to_string(FLAGS_ni) + " " + std::to_string(avg_time) + " " + standard_deviation + " " + std::to_string(FLAGS_nireq) +\
					" " + std::to_string(perc_99) + " " + std::to_string(perc_95) + " " + std::to_string(perc_90) + " " + std::to_string(perc_50) +\
					" " + std::to_string(min_time) + " " + std::to_string(max_time);

//1 =======================================================================================================================================================================
        int ret_val = system(command.c_str());
        if (ret_val == 0)
        {
          std::cout << "Inference Successful" << std::endl;
        } else {
          std::cout << "Error Running inference" << std::endl;
          std::cout << "     " << command << " RETURNS: " << std::to_string(ret_val) <<std::endl;
        }

        // --------------------------- 8. Process output -------------------------------------------------------
        slog::info << "Processing output blobs" << slog::endl;

        for (size_t i = 0; i < FLAGS_nireq; i++) {
            /** Validating -nt value **/
            const int resultsCnt = outputBlobs[i]->size() / batchSize;
            if (FLAGS_nt > resultsCnt || FLAGS_nt < 1) {
                slog::warn << "-nt " << FLAGS_nt << " is not available for this network (-nt should be less than " \
                          << resultsCnt+1 << " and more than 0)\n            will be used maximal value : " << resultsCnt << slog::endl;
                FLAGS_nt = resultsCnt;
            }
            /** This vector stores id's of top N results **/
            std::vector<unsigned> results;
            TopResults(FLAGS_nt, *outputBlobs[i], results);

            std::cout << std::endl << "Top " << FLAGS_nt << " results:" << std::endl << std::endl;

            /** Read labels from file (e.x. AlexNet.labels) **/
            bool labelsEnabled = false;
            std::string labelFileName = fileNameNoExt(FLAGS_m) + ".labels";
            std::vector<std::string> labels;

            std::ifstream inputFile;
            inputFile.open(labelFileName, std::ios::in);
            if (inputFile.is_open()) {
                std::string strLine;
                while (std::getline(inputFile, strLine)) {
                    trim(strLine);
                    labels.push_back(strLine);
                }
                labelsEnabled = true;
            }

            /** Print the result iterating over each batch **/
            std::cout << "FLAGS_nireq: " << i << std::endl;
            int batchSize = network.getBatchSize();
            for (int image_id = 0; image_id < batchSize; ++image_id) {
                std::cout << "Image " << imageNames[image_id] << std::endl << std::endl;
                for (size_t id = image_id * FLAGS_nt, cnt = 0; cnt < FLAGS_nt; ++cnt, ++id) {
                    std::cout.precision(7);
                    /** Getting probability for resulting class **/
                    auto result = outputBlobs[i]->buffer().
                            as<PrecisionTrait<Precision::FP32>::value_type*>()[results[id] + image_id*(outputBlobs[i]->size() / batchSize)];
                    std::cout << std::left << std::fixed << results[id] << " " << result;
                    if (labelsEnabled) {
                        std::cout << " label " << labels[results[id]] << std::endl;
                    } else {
                        std::cout << " label #" << results[id] << std::endl;
                    }
                }
                std::cout << std::endl;
            }
        }
        // -----------------------------------------------------------------------------------------------------
        std::cout << std::endl << "total inference time: " << total << std::endl;
        std::cout << std::endl << "Throughput: " << 1000 * static_cast<double>(FLAGS_ni) * batchSize / total << " FPS" << std::endl;
        std::cout << std::endl;

        /** Show performance results **/
        std::map<std::string, InferenceEngineProfileInfo> performanceMap;
        if (FLAGS_pc) {
            for (size_t nireq = 0; nireq < FLAGS_nireq; nireq++) {
                performanceMap = inferRequests[nireq].GetPerformanceCounts();
                printPerformanceCounts(performanceMap, std::cout);
            }
        }
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

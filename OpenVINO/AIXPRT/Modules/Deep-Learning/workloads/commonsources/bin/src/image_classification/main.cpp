/*============================================================================*
* Copyright (C) 2018 BenchmarkXPRT Development Community
* Licensed under the BENCHMARKXPRT DEVELOPMENT COMMUNITY MEMBER LICENSE AGREEMENT (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License by contacting Principled Technologies, Inc.
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing grants and
* restrictions under the License.
*============================================================================*/

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

#include <fstream>
#include <vector>
#include <chrono>
#include <memory>
#include <string>

#include <inference_engine.hpp>
#include <ext_list.hpp>
#include <format_reader_ptr.h>

#include <samples/common.hpp>
#include <samples/slog.hpp>
#include <samples/args_helper.hpp>

#include "image_classification.h"

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
        throw std::logic_error("Parameter -ni should be greater than zero (default 1)");
    }

    if (FLAGS_i.empty()) {
        throw std::logic_error("Parameter -i is not set");
    }

    if (FLAGS_m.empty()) {
        throw std::logic_error("Parameter -m is not set");
    }

    return true;
}

/**
* @brief The entry point the Inference Engine sample application
* @file classification_sample/main.cpp
* @example classification_sample/main.cpp
*/
int main(int argc, char *argv[]) {
    try {
        slog::info << "InferenceEngine: " << GetInferenceEngineVersion() << slog::endl;

        // ------------------------------ Parsing and validation of input args ---------------------------------
        slog::info << "Parsing input parameters " << slog::endl;
        std::string model_name,aarch,precision,output_dir;
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
            auto extension_ptr = make_so_pointer<IExtension>(FLAGS_l);
            plugin.AddExtension(extension_ptr);
            slog::info << "CPU Extension loaded: " << FLAGS_l << slog::endl;
        }
        if (!FLAGS_c.empty()) {
            // clDNN Extensions are loaded from an .xml description and OpenCL kernel files
            plugin.SetConfig({{PluginConfigParams::KEY_CONFIG_FILE, FLAGS_c}});
            slog::info << "GPU Extension loaded: " << FLAGS_c << slog::endl;
        }

        /** Setting plugin parameter for collecting per layer metrics **/
        if (FLAGS_pc) {
            plugin.SetConfig({ { PluginConfigParams::KEY_PERF_COUNT, PluginConfigParams::YES } });
        }

        /** Printing plugin version **/
        printPluginVersion(plugin, std::cout);
        // -----------------------------------------------------------------------------------------------------

        // --------------------------- 2. Read IR Generated by ModelOptimizer (.xml and .bin files) ------------
        std::string binFileName = fileNameNoExt(FLAGS_m) + ".bin";
        slog::info << "Loading network files:"
                "\n\t" << FLAGS_m <<
                "\n\t" << binFileName <<
        slog::endl;

        CNNNetReader networkReader;
        /** Reading network model **/
        networkReader.ReadNetwork(FLAGS_m);

        /** Extracting model name and loading weights **/
        networkReader.ReadWeights(binFileName);
        CNNNetwork network = networkReader.getNetwork();
        // -----------------------------------------------------------------------------------------------------

        // --------------------------- 3. Configure input & output ---------------------------------------------

        // --------------------------- Prepare input blobs -----------------------------------------------------
        slog::info << "Preparing input blobs" << slog::endl;

        /** Taking information about all topology inputs **/
        InputsDataMap inputInfo = network.getInputsInfo();
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
        if(batch_size != network.getBatchSize()){
            std::cout << "Batch sizes do not match: batchSize " << network.getBatchSize() << " batch_size: "<< batch_size << std::endl;
            return 0;
        }
        // -----------------------------------------------------------------------------------------------------

        // ------------------------------ Prepare output blobs -------------------------------------------------
        slog::info << "Preparing output blobs" << slog::endl;

        OutputsDataMap outputInfo(network.getOutputsInfo());
        // BlobMap outputBlobs;
        std::string firstOutputName;

        for (auto & item : outputInfo) {
            if (firstOutputName.empty()) {
                firstOutputName = item.first;
            }
            DataPtr outputData = item.second;
            if (!outputData) {
                throw std::logic_error("output data pointer is not valid");
            }

            item.second->setPrecision(Precision::FP32);
        }

        const SizeVector outputDims = outputInfo.begin()->second->getDims();

        bool outputCorrect = false;
        if (outputDims.size() == 2 /* NC */) {
            outputCorrect = true;
        } else if (outputDims.size() == 4 /* NCHW */) {
            /* H = W = 1 */
            if (outputDims[2] == 1 && outputDims[3] == 1) outputCorrect = true;
        }

        if (!outputCorrect) {
            throw std::logic_error("Incorrect output dimensions for classification model");
        }
        // -----------------------------------------------------------------------------------------------------

        // --------------------------- 4. Loading model to the plugin ------------------------------------------
        slog::info << "Loading model to the plugin" << slog::endl;

        ExecutableNetwork executable_network = plugin.LoadNetwork(network, {});
        inputInfoItem.second = {};
        outputInfo = {};
        network = {};
        networkReader = {};
        // -----------------------------------------------------------------------------------------------------

        // --------------------------- 5. Create infer request -------------------------------------------------
        InferRequest infer_request = executable_network.CreateInferRequest();
        // -----------------------------------------------------------------------------------------------------

        // --------------------------- 6. Prepare input --------------------------------------------------------
        /** Iterate over all the input blobs **/
        for (const auto & item : inputInfo) {
            /** Creating input blob **/
            Blob::Ptr input = infer_request.GetBlob(item.first);

            /** Filling input tensor with images. First b channel, then g and r channels **/
            size_t num_channels = input->getTensorDesc().getDims()[1];
            size_t image_size = input->getTensorDesc().getDims()[2] * input->getTensorDesc().getDims()[3];

            auto data = input->buffer().as<PrecisionTrait<Precision::U8>::value_type*>();

            /** Iterate over all input images **/
            for (size_t image_id = 0; image_id < imagesData.size(); ++image_id) {
                /** Iterate over all pixel in image (b,g,r) **/
                for (size_t pid = 0; pid < image_size; pid++) {
                    /** Iterate over all channels **/
                    for (size_t ch = 0; ch < num_channels; ++ch) {
                        /**          [images stride + channels stride + pixel id ] all in bytes            **/
                        data[image_id * image_size * num_channels + ch * image_size + pid ] = imagesData.at(image_id).get()[pid*num_channels + ch];
                    }
                }
            }
        }
        inputInfo = {};
        // -----------------------------------------------------------------------------------------------------

        // --------------------------- 7. Do inference ---------------------------------------------------------
        slog::info << "Starting inference (" << FLAGS_ni << " iterations)" << slog::endl;

        typedef std::chrono::high_resolution_clock Time;
        typedef std::chrono::duration<double, std::ratio<1, 1000>> ms;
        typedef std::chrono::duration<float> fsec;

        double total = 0.0;
        double sample[FLAGS_ni]; //compute STDev
        for (int iter = 0; iter < FLAGS_ni; ++iter){
          sample[FLAGS_ni] = 0.0;
        }

        /** warmup the inference engine **/
        for (int iter = 0; iter < 10; ++iter) {
            infer_request.Infer();
        }

        /** Start inference & calc performance **/
        for (int iter = 0; iter < FLAGS_ni; ++iter) {
            auto t0 = Time::now();
            infer_request.Infer();
            auto t1 = Time::now();
            fsec fs = t1 - t0;
            ms d = std::chrono::duration_cast<ms>(fs);
            total += d.count();
            sample[iter] = d.count();
        }

        /** Show performance results **/
        slog::info << "Average running time of one iteration: " << total / static_cast<double>(FLAGS_ni) << " ms" << slog::endl;

        if (FLAGS_pc) {
            printPerformanceCounts(infer_request, std::cout);
        }
        double avg_time = 0.0;
        double imgpersec = 0.0;
        avg_time = total/ static_cast<double>(FLAGS_ni);
        imgpersec = ( double(batch_size) / avg_time ) * 1000.0;
        //compute STDev
        double sqDiff = 0.0;
        double standard_deviation = 0.0;
        double variance = 0.0;
        if (FLAGS_ni>0){
          for (int iter = 0; iter < FLAGS_ni; ++iter)
          {
            std::cout << "    iteration: " << iter << " time(ms): " << sample[iter] << std::endl;
            sqDiff += pow(sample[iter]-avg_time, 2);
          }
          variance = sqDiff/FLAGS_ni;
          standard_deviation = sqrt(variance);
          std::cout << "    standard_deviation: " << standard_deviation << " variance: " << variance << std::endl;
        }


        std::cout << "Result: " << imgpersec << " images/sec" << std::endl;
        std::string const command = "python cpp_to_python_api.py "  + model_name +" " + std::to_string(batch_size) + " " + aarch + " " +\
                                    precision + " " + std::to_string(imgpersec) + " " + std::to_string(FLAGS_ni) + " " +\
                                    std::to_string(avg_time) + " " + std::to_string(standard_deviation) ;

        int ret_val = system(command.c_str());
        if (ret_val == 0)
        {
          std::cout << "Inference Successful" << std::endl;
        } else {
          std::cout << "Error Running inference" << std::endl;
          std::cout << "     "<< command << " RETURNS: " << std::to_string(ret_val) <<std::endl;
        }
        // -----------------------------------------------------------------------------------------------------

        // --------------------------- 8. Process output -------------------------------------------------------
        slog::info << "Processing output blobs" << slog::endl;

        const Blob::Ptr output_blob = infer_request.GetBlob(firstOutputName);
        auto output_data = output_blob->buffer().as<PrecisionTrait<Precision::FP32>::value_type*>();

        /** Validating -nt value **/
        const int resultsCnt = output_blob->size() / batchSize;
        if (FLAGS_nt > resultsCnt || FLAGS_nt < 1) {
            slog::warn << "-nt " << FLAGS_nt << " is not available for this network (-nt should be less than " \
                      << resultsCnt+1 << " and more than 0)\n            will be used maximal value : " << resultsCnt;
            FLAGS_nt = resultsCnt;
        }

        /** This vector stores id's of top N results **/
        std::vector<unsigned> results;
        TopResults(FLAGS_nt, *output_blob, results);

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
        for (int image_id = 0; image_id < batchSize; ++image_id) {
            std::cout << "Image " << imageNames[image_id] << std::endl << std::endl;
            for (size_t id = image_id * FLAGS_nt, cnt = 0; cnt < FLAGS_nt; ++cnt, ++id) {
                std::cout.precision(7);
                /** Getting probability for resulting class **/
                const auto result = output_data[results[id] + image_id*(output_blob->size() / batchSize)];
                std::cout << std::left << std::fixed << results[id] << " " << result;
                if (labelsEnabled) {
                    std::cout << " label " << labels[results[id]] << std::endl;
                } else {
                    std::cout << " label #" << results[id] << std::endl;
                }
            }
            std::cout << std::endl;
        }
        // -----------------------------------------------------------------------------------------------------
        std::cout << std::endl << "total inference time: " << total << std::endl;
        std::cout << "Average running time of one iteration: " << total / static_cast<double>(FLAGS_ni) << " ms" << std::endl;
        std::cout << std::endl << "Throughput: " << 1000 * static_cast<double>(FLAGS_ni) * batchSize / total << " FPS" << std::endl;
        std::cout << std::endl;

        /** Show performance results **/
        if (FLAGS_pc) {
            printPerformanceCounts(infer_request, std::cout);
        }
    }
    catch (const std::exception& error) {
        slog::err << "" << error.what() << slog::endl;
        return 1;
    }
    catch (...) {
        slog::err << "Unknown/internal exception happened." << slog::endl;
        return 1;
    }

    slog::info << "Execution successful" << slog::endl;
    return 0;
}

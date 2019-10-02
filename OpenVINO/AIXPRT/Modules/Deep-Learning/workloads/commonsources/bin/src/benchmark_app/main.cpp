/*
// Copyright (C) 2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
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

#include <algorithm>
#include <chrono>
#include <memory>
#include <map>
#include <string>
#include <vector>
#include <utility>

#include <inference_engine.hpp>
#include <ext_list.hpp>
#include <vpu/vpu_plugin_config.hpp>
#include <cldnn/cldnn_config.hpp>
#include <samples/common.hpp>
#include <samples/slog.hpp>
#include <samples/args_helper.hpp>

#include "benchmark_app.hpp"
#include "infer_request_wrap.hpp"
#include "progress_bar.hpp"
#include "statistics_report.hpp"
#include "inputs_filling.hpp"
#include "utils.hpp"

using namespace InferenceEngine;

static const size_t progressBarDefaultTotalCount = 1000;

uint64_t getDurationInMilliseconds(uint32_t duration) {
    return duration * 1000LL;
}

uint64_t getDurationInNanoseconds(uint32_t duration) {
    return duration * 1000000000LL;
}

double computePercentile( std::vector<double> arr, int perc){

   float loc = (float(perc)/100.0)*arr.size() - 1.0; // Index starts from zero
   int l = static_cast<int> (floor(loc));
   int h = static_cast<int> (ceil(loc));

   double lower = arr[l];
   double upper = arr[h];
   double value = lower + (upper - lower)*(loc - l);
   return value;

}

bool ParseAndCheckCommandLine(int argc, char *argv[]) {
    // ---------------------------Parsing and validating input arguments--------------------------------------
    slog::info << "Parsing input parameters" << slog::endl;
    gflags::ParseCommandLineNonHelpFlags(&argc, &argv, true);
    if (FLAGS_help || FLAGS_h) {
        showUsage();
        showAvailableDevices();
        return false;
    }

    if (FLAGS_m.empty()) {
        throw std::logic_error("Model is required but not set. Please set -m option.");
    }

    if (FLAGS_api != "async" && FLAGS_api != "sync") {
        throw std::logic_error("Incorrect API. Please set -api option to `sync` or `async` value.");
    }

    if (!FLAGS_report_type.empty() &&
         FLAGS_report_type != noCntReport && FLAGS_report_type != averageCntReport && FLAGS_report_type != detailedCntReport) {
        std::string err = "only " + std::string(noCntReport) + "/" + std::string(averageCntReport) + "/" + std::string(detailedCntReport) +
                " report types are supported (invalid -report_type option value)";
        throw std::logic_error(err);
    }

    // if (FLAGS_i.empty()) {
    //     throw std::logic_error("Parameter -i is not set");
    // }

    // if (FLAGS_niter < 1) {
    //     throw std::logic_error("Parameter -niter must be more than 0 ! (default 1)");
    // }

    // if (FLAGS_nireq < 1) {
    //     throw std::logic_error("Parameter -nireq must be more than 0 ! (default 1)");
    // }

    // if (FLAGS_niter < FLAGS_nireq) {
    //     throw std::logic_error("Number of iterations could not be less than requests quantity");
    // }

    return true;
}

static void next_step(const std::string additional_info = "") {
    static size_t step_id = 0;
    static const std::map<size_t, std::string> step_names = {
      { 1, "Parsing and validating input arguments" },
      { 2, "Loading Inference Engine" },
      { 3, "Reading the Intermediate Representation network" },
      { 4, "Resizing network to match image sizes and given batch" },
      { 5, "Configuring input of the model" },
      { 6, "Setting device configuration" },
      { 7, "Loading the model to the device" },
      { 8, "Setting optimal runtime parameters" },
      { 9, "Creating infer requests and filling input blobs with images" },
      { 10, "Measuring performance" },
      { 11, "Dumping statistics report" }
    };

    step_id++;
    if (step_names.count(step_id) == 0)
        THROW_IE_EXCEPTION << "Step ID " << step_id << " is out of total steps number " << step_names.size();

    std::cout << "[Step " << step_id << "/" << step_names.size() << "] " << step_names.at(step_id)
              << (additional_info.empty() ? "" : " (" + additional_info + ")") << std::endl;
}

/**
* @brief The entry point of the benchmark application
*/
int main(int argc, char *argv[]) {
    try {
        // ----------------- 1. Parsing and validating input arguments -------------------------------------------------
        next_step();

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
        std::vector<std::string> inputFiles;
        parseInputFilesArguments(inputFiles);

        // ----------------- 2. Loading the Inference Engine -----------------------------------------------------------
        next_step();

        // Get optimal runtime parameters for device
        std::string device_name = FLAGS_d;

        Core ie;

        if (FLAGS_d.find("CPU") != std::string::npos) {
            // Loading default CPU extensions
            ie.AddExtension(std::make_shared<Extensions::Cpu::CpuExtensions>(), "CPU");

            if (!FLAGS_l.empty()) {
                // CPU (MKLDNN) extensions is loaded as a shared library and passed as a pointer to base extension
                const auto extension_ptr = InferenceEngine::make_so_pointer<InferenceEngine::IExtension>(FLAGS_l);
                ie.AddExtension(extension_ptr, "CPU");
                slog::info << "CPU (MKLDNN) extensions is loaded " << FLAGS_l << slog::endl;
            }
        }

        if ((FLAGS_d.find("GPU") != std::string::npos) && !FLAGS_c.empty()) {
            // Load clDNN Extensions
            ie.SetConfig({ {CONFIG_KEY(CONFIG_FILE), FLAGS_c} });
            slog::info << "GPU extensions is loaded " << FLAGS_c << slog::endl;
        }

        if (FLAGS_d.find("MYRIAD") != std::string::npos){
            slog::info << " == MYRIAD == " << "iterations: " << FLAGS_niter << " request: " << FLAGS_nireq << " batch_size "<< batch_size << slog::endl;
        }

        if (FLAGS_d.find("HDDL") != std::string::npos){
            slog::info << " == HDDL == " << "iterations: " << FLAGS_niter << " request : " << FLAGS_nireq << " batch_size "<< batch_size << slog::endl;
        }

        slog::info << "InferenceEngine: " << GetInferenceEngineVersion() << slog::endl;
        slog::info << "Device info: " << slog::endl;
        std::cout << ie.GetVersions(device_name) << std::endl;

        // ----------------- 3. Reading the Intermediate Representation network ----------------------------------------
        next_step();

        slog::info << "Loading network files" << slog::endl;

        CNNNetReader netBuilder;
        netBuilder.ReadNetwork(FLAGS_m);
        const std::string binFileName = fileNameNoExt(FLAGS_m) + ".bin";
        netBuilder.ReadWeights(binFileName);

        CNNNetwork cnnNetwork = netBuilder.getNetwork();
        const InputsDataMap inputInfo(cnnNetwork.getInputsInfo());
        if (inputInfo.empty()) {
            throw std::logic_error("no inputs info is provided");
        }

        // ----------------- 4. Resizing network to match image sizes and given batch ----------------------------------
        next_step();

        if (FLAGS_b != 0) {
            ICNNNetwork::InputShapes shapes = cnnNetwork.getInputShapes();
            bool reshape = false;
            for (const InputsDataMap::value_type& item : inputInfo) {
                auto layout = item.second->getTensorDesc().getLayout();

                int batchIndex = -1;
                if ((layout == Layout::NCHW) || (layout == Layout::NCDHW) ||
                    (layout == Layout::NHWC) || (layout == Layout::NDHWC) ||
                    (layout == Layout::NC)) {
                    batchIndex = 0;
                } else if (layout == CN) {
                    batchIndex = 1;
                }
                if ((batchIndex != -1) && (shapes[item.first][batchIndex] != FLAGS_b)) {
                    shapes[item.first][batchIndex] = FLAGS_b;
                    reshape = true;
                }
            }
            if (reshape) {
                slog::info << "Resizing network to batch = " << FLAGS_b << slog::endl;
                cnnNetwork.reshape(shapes);
            }
        }

        const size_t batchSize = cnnNetwork.getBatchSize();
        slog::info << (FLAGS_b != 0 ? "Network batch size was changed to: " : "Network batch size: ") << batchSize <<
            ", precision: " << cnnNetwork.getPrecision() << slog::endl;

        // ----------------- 5. Configuring input ----------------------------------------------------------------------
        next_step();

        for (auto& item : inputInfo) {
            if (isImage(item.second)) {
                /** Set the precision of input data provided by the user, should be called before load of the network to the device **/
                item.second->setPrecision(Precision::U8);
            }
        }

        // ----------------- 6. Setting device configuration -----------------------------------------------------------
        next_step();

        bool perf_counts = (FLAGS_report_type == detailedCntReport ||
                            FLAGS_report_type == averageCntReport ||
                            FLAGS_pc);

        auto devices = parseDevices(device_name);
        std::map<std::string, uint32_t> device_nstreams = parseValuePerDevice(devices, FLAGS_nstreams);
        for (auto& device : devices) {
            if (device == "CPU") {  // CPU supports few special performance-oriented keys
                // limit threading for CPU portion of inference
                if (FLAGS_nthreads != 0)
                    ie.SetConfig({{ CONFIG_KEY(CPU_THREADS_NUM), std::to_string(FLAGS_nthreads) }}, device);

                if ((device_name.find("MULTI") != std::string::npos) &&
                    (device_name.find("GPU") != std::string::npos)) {
                    ie.SetConfig({{ CONFIG_KEY(CPU_BIND_THREAD), CONFIG_VALUE(NO) }}, device);
                } else {
                    // pin threads for CPU portion of inference
                    ie.SetConfig({{ CONFIG_KEY(CPU_BIND_THREAD), FLAGS_pin }}, device);
                }

                // for CPU execution, more throughput-oriented execution via streams
                if (FLAGS_api == "async")
                    ie.SetConfig({{ CONFIG_KEY(CPU_THROUGHPUT_STREAMS),
                                    (device_nstreams.count(device) > 0 ? std::to_string(device_nstreams.at(device)) :
                                                                         "CPU_THROUGHPUT_AUTO") }}, device);
                device_nstreams[device] = std::stoi(ie.GetConfig(device, CONFIG_KEY(CPU_THROUGHPUT_STREAMS)).as<std::string>());
            } else if (device == ("GPU")) {
                if (FLAGS_api == "async")
                    ie.SetConfig({{ CONFIG_KEY(GPU_THROUGHPUT_STREAMS),
                                    (device_nstreams.count(device) > 0 ? std::to_string(device_nstreams.at(device)) :
                                                                         "GPU_THROUGHPUT_AUTO") }}, device);
                device_nstreams[device] = std::stoi(ie.GetConfig(device, CONFIG_KEY(GPU_THROUGHPUT_STREAMS)).as<std::string>());

                if ((device_name.find("MULTI") != std::string::npos) &&
                    (device_name.find("CPU") != std::string::npos)) {
                    // multi-device execution with the CPU + GPU performs best with GPU trottling hint,
                    // which releases another CPU thread (that is otherwise used by the GPU driver for active polling)
                    ie.SetConfig({{ CLDNN_CONFIG_KEY(PLUGIN_THROTTLE), "1" }}, "GPU");
                }
            } else if ((device == "MYRIAD") || (device == "HDDL")){
                if (FLAGS_api == "async"){
                        ie.SetConfig({{ CONFIG_KEY(LOG_LEVEL), CONFIG_VALUE(LOG_NONE) },
                              { VPU_CONFIG_KEY(LOG_LEVEL), CONFIG_VALUE(LOG_WARNING) },
                              { VPU_CONFIG_KEY(HW_STAGES_OPTIMIZATION), CONFIG_VALUE(YES)},
                              { VPU_CONFIG_KEY(PRINT_RECEIVE_TENSOR_TIME), CONFIG_VALUE(YES)},
                              }, device);
                }
                else{
                        ie.SetConfig({{ CONFIG_KEY(LOG_LEVEL), CONFIG_VALUE(LOG_NONE) },
                              { VPU_CONFIG_KEY(LOG_LEVEL), CONFIG_VALUE(LOG_WARNING) }}, device);
                }
            }

        }

        // ----------------- 7. Loading the model to the device --------------------------------------------------------
        next_step();

        std::map<std::string, std::string> config = {{ CONFIG_KEY(PERF_COUNT), perf_counts ? CONFIG_VALUE(YES) :
                                                                                             CONFIG_VALUE(NO) }};
        ExecutableNetwork exeNetwork = ie.LoadNetwork(cnnNetwork, device_name, config);

        // ----------------- 8. Setting optimal runtime parameters -----------------------------------------------------
        next_step();

        // Number of requests
        uint32_t nireq = FLAGS_nireq;
        if (nireq == 0) {
            std::string key = METRIC_KEY(OPTIMAL_NUMBER_OF_INFER_REQUESTS);
            try {
                nireq = exeNetwork.GetMetric(key).as<unsigned int>();
            } catch (const std::exception ex) {
                nireq = deviceDefaultRequestsNumber(device_name);
                slog::warn << "Can't get " << key + " metric: " << ex.what()
                           << " Use default value " << nireq << " for device " << device_name << slog::endl;
            }
        }

        // Iteration limit
        uint32_t niter = FLAGS_niter;
        if ((niter > 0) && (FLAGS_api == "async")) {
            niter = ((niter + nireq - 1)/nireq)*nireq;
            if (FLAGS_niter != niter) {
                slog::warn << "Number of iterations was aligned by request number from "
                           << FLAGS_niter << " to " << niter << " using number of requests " << nireq << slog::endl;
            }
        }

        // Time limit
        uint32_t duration_seconds = 0;
        if (FLAGS_t != 0) {
            // time limit
            duration_seconds = FLAGS_t;
        } else if (FLAGS_niter == 0) {
            // default time limit
            duration_seconds = deviceDefaultDeviceDurationInSeconds(device_name);
        }
        uint64_t duration_nanoseconds = getDurationInNanoseconds(duration_seconds);

        // -------------------------------------------------------------------------------------------------------------------------------------
        // slog::info << " Initializing request times "<<slog::endl;
        //auto t_ = Time::now();
        //std::map < int, std::vector<decltype(t_)> > req_start_times;
        //std::map < int, std::vector<double> > req_total_times;
        //ns req_d;
        //std::chrono::duration<double> st_d;
        //for (unsigned int i=0; i < nireq; i++)
        //{
        //   std::vector<double> times_vec = {};
        //   std::vector<decltype(t_)> runtime_vec {Time::now()};
        //   req_start_times[i] = runtime_vec;
        //   req_total_times[i] = times_vec;
        //}

        // ----------------- 9. Creating infer requests and filling input blobs ----------------------------------------
        next_step();

        InferRequestsQueue inferRequestsQueue(exeNetwork, nireq);

        fillBlobs(inputFiles, batchSize, inputInfo, inferRequestsQueue.requests);

        // ----------------- 10. Measuring performance ------------------------------------------------------------------
        size_t progressCnt = 0;
        size_t progressBarTotalCount = progressBarDefaultTotalCount;
        size_t iteration = 0;

        std::stringstream ss;
        ss << "Start inference " << FLAGS_api << "ronously";
        if (FLAGS_api == "async") {
            if (!ss.str().empty()) {
                ss << ", ";
            }
            ss << nireq << " inference requests";
            std::stringstream device_ss;
            for (auto& nstreams : device_nstreams) {
                if (!device_ss.str().empty()) {
                    device_ss << ", ";
                }
                device_ss << nstreams.second << " streams for " << nstreams.first;
            }
            if (!device_ss.str().empty()) {
                ss << " using " << device_ss.str();
            }
        }
        ss << ", limits: ";
        if (duration_seconds > 0) {
            ss << getDurationInMilliseconds(duration_seconds) << " ms duration";
        }
        if (niter != 0) {
            if (duration_seconds == 0) {
                progressBarTotalCount = niter;
            }
            if (duration_seconds > 0) {
                ss << ", ";
            }
            ss << niter << " iterations";
        }
        next_step(ss.str());

        // warming up - out of scope
        auto inferRequest = inferRequestsQueue.getIdleRequest();
        if (!inferRequest) {
            THROW_IE_EXCEPTION << "No idle Infer Requests!";
        }

        if (FLAGS_api == "sync") {
            inferRequest->infer();
        } else {
            inferRequest->startAsync();
        }
        inferRequestsQueue.waitAll();
        inferRequestsQueue.resetTimes();

        const auto startTime = Time::now();
        auto execTime = std::chrono::duration_cast<ns>(Time::now() - startTime).count();

        /** Start inference & calculate performance **/
        /** to align number if iterations to guarantee that last infer requests are executed in the same conditions **/
        ProgressBar progressBar(progressBarTotalCount, FLAGS_stream_output, FLAGS_progress);

        while ((niter != 0LL && iteration < niter) ||
               (duration_nanoseconds != 0LL && (uint64_t)execTime < duration_nanoseconds) ||
               (FLAGS_api == "async" && iteration % nireq != 0)) {
            inferRequest = inferRequestsQueue.getIdleRequest();
            if (!inferRequest) {
                THROW_IE_EXCEPTION << "No idle Infer Requests!";
            }

            if (FLAGS_api == "sync") {
                inferRequest->infer();
            } else {
                inferRequest->startAsync();
            }
            iteration++;

            execTime = std::chrono::duration_cast<ns>(Time::now() - startTime).count();

            if (niter > 0) {
                progressBar.addProgress(1);
            } else {
                // calculate how many progress intervals are covered by current iteration.
                // depends on the current iteration time and time of each progress interval.
                // Previously covered progress intervals must be skipped.
                auto progressIntervalTime = duration_nanoseconds / progressBarTotalCount;
                size_t newProgress = execTime / progressIntervalTime - progressCnt;
                progressBar.addProgress(newProgress);
                progressCnt += newProgress;
            }
        }

        // wait the latest inference executions
        inferRequestsQueue.waitAll();

        StatisticsReport statistics({ FLAGS_d,
                                      FLAGS_api,
                                      batchSize,
                                      nireq,
                                      niter,
                                      getDurationInMilliseconds(duration_seconds),
                                      FLAGS_nthreads,
                                      device_nstreams,
                                      FLAGS_pin,
                                      FLAGS_report_type,
                                      FLAGS_report_folder
                                    });
        if (perf_counts) {
            for (auto& request : inferRequestsQueue.requests) {
                statistics.addPerfCounts(request->getPerformanceCounts());
            }
        }
        statistics.addLatencies(inferRequestsQueue.getLatencies());

        double totalDuration = inferRequestsQueue.getDurationInMilliseconds();
        double fps = (FLAGS_api == "sync") ? batchSize * 1000.0 * iteration / totalDuration :
                                             batchSize * 1000.0 * iteration / totalDuration;
        progressBar.finish();

        // -------------------------------------------------------------------------------------------------------------
        //print latencies per inference
        //ammend this values per iteration count
        std::vector<double> inference_times_array;
        inference_times_array = inferRequestsQueue.getLatencies();
        int idx = 0;
        for (auto i = inference_times_array.begin(); i != inference_times_array.end(); ++i){
            slog::info << "Request " << idx << " " << idx << " : " << *i << ' ' << slog::endl;
            idx++;
        }
        // -------------------------------------------------------------------------------------------------------------
        /** Compute Percentiles **/
        std::sort (inference_times_array.begin(), inference_times_array.end()); // Sort the times
        double max_time= inference_times_array.back();
        double perc_99 = computePercentile( inference_times_array, 99);
        double perc_95 = computePercentile( inference_times_array, 95);
        double perc_90 = computePercentile( inference_times_array, 90);
        double perc_50 = computePercentile( inference_times_array, 50);
        double min_time= inference_times_array[0];

        // -------------------------------------------------------------------------------------------------------------------------------------

        std::cout << std::endl;
        std::cout << "total: " << totalDuration << std::endl;
        double avg_time = totalDuration / static_cast<double>(FLAGS_niter);
        std::cout << "Throughput: " << fps << " FPS" << std::endl;
        double imgpersec =  fps;
        std::cout << "Result: " << imgpersec << " images/sec" << std::endl;
        std::cout << std::endl;

        //double standard_deviation = 0.0; // not possible to compute
        std::string standard_deviation = "Undefined";
        std::string const command = "python3 cpp_to_python_api.py "  + model_name +" " + std::to_string(batch_size) + " " + aarch +\
            " " + precision + " " + std::to_string(imgpersec) + " " + std::to_string(iteration) + " " + std::to_string(avg_time) +\
            " " + standard_deviation + " " + std::to_string(nireq) +\
            " " + std::to_string(perc_99) + " " + std::to_string(perc_95) + " " + std::to_string(perc_90) + " " + std::to_string(perc_50) +\
            " " + std::to_string(min_time) + " " + std::to_string(max_time);

        // -------------------------------------------------------------------------------------------------------------------------------------
        int ret_val = system(command.c_str());
        if (ret_val == 0) std::cout << "Inference Successful" << std::endl;
        else {
          std::cout << "Error Running inference" << std::endl;
          std::cout << "     " << command << " RETURNS: " << std::to_string(ret_val) <<std::endl;
        }
        // ----------------- 11. Dumping statistics report -------------------------------------------------------------
        next_step();

        statistics.dump(fps, iteration, totalDuration);

        if (!FLAGS_exec_graph_path.empty()) {
            try {
                CNNNetwork execGraphInfo = exeNetwork.GetExecGraphInfo();
                execGraphInfo.serialize(FLAGS_exec_graph_path);
                slog::info << "executable graph is stored to " << FLAGS_exec_graph_path << slog::endl;
            } catch (const std::exception & ex) {
                slog::err << "Can't get executable graph: " << ex.what() << slog::endl;
            }
        }

        if (FLAGS_pc) {
            for (size_t ireq = 0; ireq < nireq; ireq++) {
                slog::info << "Pefrormance counts for " << ireq << "-th infer request:" << slog::endl;
                printPerformanceCounts(inferRequestsQueue.requests[ireq]->getPerformanceCounts(), std::cout, getFullDeviceName(ie, FLAGS_d), false);
            }
        }

        std::cout << "Count:      " << iteration << " iterations" << std::endl;
        std::cout << "Duration:   " << totalDuration << " ms" << std::endl;
        if (device_name.find("MULTI") == std::string::npos)
            std::cout << "Latency:    " << statistics.getMedianLatency() << " ms" << std::endl;
        std::cout << "Throughput: " << fps << " FPS" << std::endl;
    } catch (const std::exception& ex) {
        slog::err << ex.what() << slog::endl;
        return 3;
    }

    return 0;
}

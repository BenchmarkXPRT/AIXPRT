import   sys
import   os
import   time
import   numpy as np
import   json
import   subprocess
from subprocess import Popen
import   os.path
from numpy import genfromtxt

sys.path.insert(1, os.path.join(os.environ['APP_HOME'], 'Harness'))
import resultsapi
import utils

batch_sizes=[1024]
model_name= "wide-deep"
dir_name= "wide-deep"





def writeBatchResults(path_list,batchsize,aarch,iterations,instances,total_requests, precision):
    #read timings from csv file and log results

      #read timings from csv file and log results

    for path in path_list:
        csv_data = genfromtxt(path, delimiter=',')
        # print("iteration "+str(iterations))
        if 'np_from_csv_data' in dir():
            np_from_csv_data = np.vstack((np.array(np_from_csv_data), csv_data))
        else:
            np_from_csv_data = csv_data
    async_timings = (np_from_csv_data[:,1] - np_from_csv_data[:,0])*1000

    if (np_from_csv_data.shape == (2,)):
        tend_max = np_from_csv_data[1]
        tstart_min = np_from_csv_data[0]
    else:
        tstart_max, tend_max = np_from_csv_data.max(axis=0)
        tstart_min, tend_min = np_from_csv_data.min(axis=0)

    speed_mean =  ( batchsize*np_from_csv_data.shape[0] )/ (tend_max - tstart_min)

    labelstr = "Batch "+ str(batchsize)
    additional_info_details = {}
    additional_info_details["total_requests"] = total_requests
    additional_info_details["concurrent_instances"] = instances
    additional_info_details["50_percentile_time"] = np.percentile(async_timings, 50)
    additional_info_details["90_percentile_time"] = np.percentile(async_timings, 90)
    additional_info_details["95_percentile_time"] = np.percentile(async_timings, 95)
    additional_info_details["99_percentile_time"] = np.percentile(async_timings, 99)
    additional_info_details["time_units"] = "milliseconds"
    accelerator_lib_details = {}

    if (aarch.lower()=="cpu"):
        accelerator_lib_details["cpu_accelerator_lib"] = ""
    else:
        accelerator_lib_details["gpu_accelerator_lib"] = ""
    workloadInput={
          "architecture":aarch,
          "precision":precision,
          "accelerator_lib": [accelerator_lib_details],
          "framework": "MXNet"
         }

    results=[
          {
          "label":labelstr,
          "system_throughput":speed_mean,
          "system_throughput_units":"samples/sec",
          "additional info":[additional_info_details]
          }
        ]
    resultsapi.createResultJson("wide-deep", workloadInput, results)


def get_params_from_json():
    # defaults
    aarch="CPU"
    precision="fp32"
    batch_size_number=[1,2,4,8]
    total_requests = 10
    concurrent_instances = 1
    framework_graphTransform = None
    instance_allocation = []
    workload_details = os.path.join(os.environ['APP_HOME'],"Modules","Deep-Learning","workloads","wide-deep","workload_details.json")
    with open(workload_details) as data_file:
        data = json.load(data_file)
    data_file.close()
    batch_size_number=[]
    workloadName = data["name"]
    workloadID = data["id"]
    if not (data.get("requested_config",None)==None):
        requested_config = data["requested_config"]
        if not (requested_config.get("hardware",None)==None):
            aarch = data["requested_config"]["hardware"].upper()
        if not (requested_config.get("precision",None)==None):
            precision = data["requested_config"]["precision"]
        if not (requested_config.get("batch_sizes",None)==None):
            batch_size_number = data["requested_config"]["batch_sizes"]
        if not (requested_config.get("total_requests",None)==None):
            total_requests = data["requested_config"]["total_requests"]
        if not (requested_config.get("concurrent_instances",None)==None):
            concurrent_instances = data["requested_config"]["concurrent_instances"]

        if not (requested_config.get("setNUMA",None)==None):
            setNUMA = data["requested_config"]["setNUMA"]
        if not (requested_config.get("env_variables",None)==None):
            env_variables = data["requested_config"]["env_variables"]

        if not (requested_config.get("instance_allocation",None)==None):
            instance_allocation = data["requested_config"]["instance_allocation"]
        if not (requested_config.get("runtype",None)==None):
            runtype = data["requested_config"]["runtype"]


    return(aarch, precision, batch_size_number, workloadName , workloadID,total_requests , concurrent_instances, setNUMA,env_variables,instance_allocation,runtype)


aarch, precision, batch_size_number, workloadName, workloadID, total_requests, concurrent_instances,setNUMA,env_variables,instance_allocation,runtype=get_params_from_json()

path_arg =  os.path.join(os.environ['APP_HOME'],"Modules","Deep-Learning","workloads","wide-deep","bin")

path = os.path.join(os.environ['APP_HOME'],"Modules","Deep-Learning","workloads","wide-deep","bin","dataScript.sh ")

fullPath = path + path_arg
if not (os.path.exists(os.path.join(os.environ['APP_HOME'], "Modules", "Deep-Learning", "workloads","commonsources", "recommendation","val_csr.pkl"))):
    subprocess.call(fullPath, shell=True)



# Set environment variable if provided
if(env_variables):
    for key, value in env_variables.items():
        os.environ[key] = value


if total_requests % concurrent_instances == 0:
    iterations = int(total_requests/concurrent_instances)
else:
    print("ERROR: total_requests should be a mutiple of concurrent_instances")
    sys.exit()


os.chdir(os.path.join(os.environ['APP_HOME'],"Modules","Deep-Learning","workloads","commonsources","recommendation"))

out = os.path.join(os.environ['APP_HOME'], "Modules", "Deep-Learning", "workloads", "commonsources", "recommendation",
                       "result","console_out_wide-deep.txt")
f = open(out, "w")



for batchSize in batch_size_number:
    commands = []
    path_list = []
    allocation = []


    if setNUMA:
        if  not len(instance_allocation) == concurrent_instances:
            print("Please add instance allocation to your config as the NUMA is set to true")
            sys.exit()
        else:
            for item in instance_allocation:
                cmd = ""
                for key ,value in item.items():
                    cmd+= "--"+key+"="+value+" "
                allocation.append(str(cmd))
    for ins in range(concurrent_instances):
        #For writing timings from all instances
        csv_file_path = os.path.join(os.environ['APP_HOME'],"Modules","Deep-Learning","workloads",
        "wide-deep",'result','output','wide-deep_batch_'+str(batchSize)+'_'+precision+'_concurrent_instance'+str(ins)+'.csv')
        if os.path.exists(csv_file_path):
            os.remove(csv_file_path)
        command = ""
        if(setNUMA):
            #instantiate numactl variables
            command = "numactl "+ allocation[ins]
        if precision == "fp32":
            command = command + " python3 inference.py --batch-size "+ str(batchSize) + " --csv-file-path "+ csv_file_path + " --aarch "  + aarch +" --iterations "+ str(iterations)
        if precision == "int8":
            command = command + " python3 inference.py --batch-size "+ str(batchSize) + " --csv-file-path "+ csv_file_path + " --aarch "  + aarch +" --iterations "+ str(iterations) + " --symbol-file=WD-quantized-162batches-naive-symbol.json --param-file=WD-quantized-0000.params"

        commands.append(command)
        path_list.append(csv_file_path)
    print(commands)
    processes = [Popen(cmd, shell=True) for cmd in commands]
    for p in processes: p.wait()
    writeBatchResults(path_list,batchSize,aarch,iterations,concurrent_instances, total_requests, precision)

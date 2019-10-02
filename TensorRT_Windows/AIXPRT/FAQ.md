#### Results json file shows all batchsize tags even though only batch size 1 is ran. <br />
     Yes, this is an expected behaviour. This is the default results schema set in place.
    If a workload tag has no value to it then it does not mean that the workload failed to run.
    
#### Do AIXPRT-OpenVINO workloads require installing OpenVINO for every update of AIXPRT build
    No, AIXPRT needs a version of OpenVINO to be installed on the system.
    Unless user wants to try a different version of OpenVINO, one doesnt have to install
    OpenVINO for every update of AIXPRT. However, users are required to build the AIXPRT 
    workloads for every version update or fresh install. 

#### Can there be multiple versions/types of tensorflow on the system ?
    We recommend to have only one version/type of Tensorflow installed on the test machine to
    avoid discrepancies in the performance. If user wants to run tensorflow-gpu and 
    tensorflow cpu on the same system , then we recommended uninstalling the existing version, reboot the system
    and install the other version of tensorflow to run the bencmark. 
    
#### How can I understand what each tag in the config file mean 

     Please check this [document] ( TODO : add url to  AIXPRT/EditConfig.md)
     
#### Pip Warning messages while running the workload ( You are using pip version 8.1.1, however version 18.1 is available.You should consider upgrading via the 'pip install --upgrade pip' command.) 

     AIXPRT uses pip list to know the version and type of tensorflow installed with a pip command .
     This is a warning from pip to let user know that there is a latest version. Users can safely ignore these messages. 

####  How many Neural Compute sticks/Movidious Devices can AIXPRT support in a single run ?

     AIXPRT only supports 1 device on a single run. Users are requiered not to insert more than 1 movidius devices
     onto to the host machine while running AIXPRT. 

####  Why do I see warnings below while running tensorrt workloads and will it impact performance?
WARNING:tensorflow:From /workspace/AIXPRT/models/research/object_detection/exporter.py:330: get_or_create_global_step (from tensorflow.contrib.framework.python.ops.variables) is deprecated and will be removed in a future version.
Instructions for updating:
Please switch to tf.train.get_or_create_global_step
WARNING:tensorflow:From /workspace/AIXPRT/models/research/object_detection/exporter.py:484: print_model_analysis (from tensorflow.contrib.tfprof.model_analyzer) is deprecated and will be removed after 2018-01-01.
Instructions for updating:
Use `tf.profiler.profile(graph, run_meta, op_log, cmd, options)`. Build `options` with `tf.profiler.ProfileOptionBuilder`. See README.md for details

    
     The warnings come from a tensorflow repository and do not impact performance.
 

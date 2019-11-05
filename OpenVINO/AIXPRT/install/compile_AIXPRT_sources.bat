@echo off

:: ============================= parse command line options and set AIXPRT install directory ===============================================
:: 							Step 1. Parse command line inputs and set up directories
:: =========================================================================================================================================
setlocal EnableDelayedExpansion
set "DASHES==========================================================================================="
:parse_input_arguments
echo %DASHES%
echo Setting AIXPRT Path
echo.
if not "%~dp1"=="" (
	if exist "%~dp1" (
		set AIXPRT_DIR="%~dp1"
	)
) else (
	 echo.
	 echo Please provide path to AIXPRT
	 echo.
	 goto :usage
)
echo %DASHES%
echo Setting OpenVINO Path
echo.
if not "%~dp2"=="" (
 	 set OPENVINO_DIR="%~dp2"
) else (
	 echo.
	 echo Please provide path to openvino installation
	 echo.
	 goto :usage
)

:: ================================================================
::                  Verify directories
:: ================================================================
echo %DASHES%
echo Verifying AIXPRT installation directories in %AIXPRT_DIR%
echo.
if exist "%AIXPRT_DIR%\" (
	if not exist "%AIXPRT_DIR%\Modules\" (
		echo.
		echo ERROR: Incorrect path to  AIXPRT directory. Should contain 'Modules' subdirectory
		echo.
		goto :usage
	)
)

echo %DASHES%
echo Verifing OpenVINO installation directories in %OPENVINO_DIR%\bin
echo.
: : Check if the openvino directory is custome dldt build
if not x%OPENVINO_DIR:\dldt\=%==x%OPENVINO_DIR% (
	echo Using the build of dldt
	call restructure-installation.bat %OPENVINO_DIR% %AIXPRT_DIR%
	set OPENVINO_DIR="C:\Program Files (x86)\IntelSWTools\ov_custom\"
) 
if not exist %OPENVINO_DIR%\bin\ (
	echo.
	echo The provided directory %OPENVINO_DIR% must contain 'bin' subdirectory. Please provide the correct install directory of OpenVINO
	echo.
	goto :usage
)

:: Set Openvino version
echo %DASHES%
echo Checking OpenVINO Version
echo.

if exist %OPENVINO_DIR%\deployment_tools\tools\ (
     set OPENVINO_VERSION="R1 and above"
)
if exist %OPENVINO_DIR%\deployment_tools\model_downloader\ (
	 :: previous version (2018 R3-R5) have similar directory structure
     set OPENVINO_VERSION="R5"
)
echo OpenVINO version is %OPENVINO_VERSION%
echo.

::================================================================
::                        Set Directories
::================================================================

set "AIXPRT_SOURCES=%AIXPRT_DIR%\Modules\Deep-Learning\workloads\commonsources\bin\src"
set "AIXPRT_MODELS=%AIXPRT_DIR%\Modules\Deep-Learning\packages\models"
set "AIXPRT_PLUGIN=%AIXPRT_DIR%\Modules\Deep-Learning\packages\plugin\"
set "AIXPRT_BIN=%AIXPRT_DIR%\Modules\Deep-Learning\workloads\commonsources\bin\"

set "CUR_PATH=%~dp0"

:: ========================= Install dependencies ================
::                      Step 2. Install Dependencies
::================================================================

echo %DASHES%
echo      Setting up OpenVINO environment
echo.

cd "%OPENVINO_DIR%"
if exist "bin\" (
	 echo Initializing OpenVINO variables in %OPENVINO_DIR%\bin\setupvars.bat
	 cd "bin\"
	 call setupvars.bat
) else (
	 echo Cannot find OpenVINO environment setup script. Please make sure OpenVINO is installed correctly
	 goto :usage
)
cd "%CUR_PATH%"
 
set "PYTHON_BINARY=python"
set "OPENVINO_IE_DIR="%INTEL_CVSDK_DIR%\deployment_tools\inference_engine\"
set "OPENVINO_MO_DIR=%INTEL_CVSDK_DIR%\deployment_tools\model_optimizer\"
set MO_PATH="%OPENVINO_MO_DIR%\mo.py"

if %OPENVINO_VERSION%=="R1 and above" (
	 if exist %INTEL_CVSDK_DIR%\deployment_tools\tools\model_downloader (
		set MD_PATH="%INTEL_CVSDK_DIR%\deployment_tools\tools\model_downloader\downloader.py" 
	 ) else (
	 set MD_PATH="%INTEL_CVSDK_DIR%\deployment_tools\tools\open_model_zoo\tools\downloader\downloader.py"
	 )
) else (
	 set MD_PATH="%INTEL_CVSDK_DIR%\deployment_tools\model_downloader\downloader.py"
)

echo %DASHES%
echo     Install Model Optimizer and Downloader dependencies
echo.
pip install --user pyyaml requests pillow numpy

cd "%OPENVINO_MO_DIR%/install_prerequisites"
call install_prerequisites_caffe.bat

cd %CUR_PATH%

::============= Download and Convert pre-trained models ==========
::        Step 3. Download and Convert pretrained Models
::================================================================

echo %DASHES%
echo      Generating IR files
echo.

set MODEL_DIR="%CUR_PATH%\caffe_models"

set MODEL_NAMES=(resnet-50,mobilenet-ssd)

set PRECISION_LIST=(FP16,FP32)

cd 
for %%G in %MODEL_NAMES% do (
  set MODEL_NAME="%%G"
  call echo !MODEL_NAME!
  ::call color 2
  call echo Generating IR files
  ::call color 1
  set IR_PATH="%CUR_PATH%\!MODEL_NAME!"
  if exist !IR_PATH! ( 
     rd /s /q !IR_PATH!
     if exist !IR_PATH! rd /s /q !IR_PATH!
  )
	 
  if !MODEL_NAME!=="mobilenet-ssd" (
    call echo Inside mobilenet-ssd
	call set "MODEL_DEST=ssd_mobilenet"
	call set "MODEL_PATH=%MODEL_DIR%\public\mobilenet-ssd\mobilenet-ssd.caffemodel"

	call set "MEAN_VALUES=data[127.5,127.5,127.5]"
	call set "SCALE_VALUES=data[127.50223128904757]"
	call set "MODEL_DEST=ssd_mobilenet"
	call set "INPUT_SHAPE=[1,3,300,300]"
	call echo Setting up parameters for ssd_mobilenet
  )
  
  if !MODEL_NAME!=="resnet-50" (
	call set "MODEL_NAME=resnet-50"
	call set "MODEL_DEST=resnet-50"
	call set "MODEL_PATH=%MODEL_DIR%\public\resnet-50\resnet-50.caffemodel"

	call set "MEAN_VALUES=data[104.0,117.0,123.0]"
	call set "SCALE_VALUES=data[1.0]"
	call set "INPUT_SHAPE=[1,3,224,224]"
	call echo Setting up parameters for resnet-50
  )
  call echo Model Destination is !MODEL_DEST!
  
  call %PYTHON_BINARY% %MD_PATH% --name "!MODEL_NAME!" --output_dir "%MODEL_DIR%"

  for %%P in %PRECISION_LIST% do (
	 call set PRECISION=%%P
	 if "!PRECISION!"=="FP16" call set prec=fp16
	 if "!PRECISION!"=="FP32" call set prec=fp32
	 
	 call set IR_MODEL_XML=!MODEL_DEST!_!PRECISION!.xml
	 call set IR_MODEL_BIN=!MODEL_DEST!_!PRECISION!.bin
	 call set IR_MODEL_MLxBench_XML=!MODEL_DEST!_!prec!.xml
	 call set IR_MODEL_MLxBench_BIN=!MODEL_DEST!_!prec!.bin
	 call echo Run !PYTHON_BINARY! !MO_PATH! --input_model !MODEL_PATH! --output_dir !IR_PATH! --model_name !MODEL_DEST! --data_type !PRECISION! --input_shape !INPUT_SHAPE! --mean_values !MEAN_VALUES! --scale_values !SCALE_VALUES!"
	 call %PYTHON_BINARY% !MO_PATH! --input_model !MODEL_PATH! --output_dir !IR_PATH! --model_name !MODEL_DEST!_!precision! --data_type !PRECISION! --input_shape !INPUT_SHAPE! --mean_values !MEAN_VALUES! --scale_values !SCALE_VALUES!
	 
	 call cd !IR_PATH!
     call ren !IR_MODEL_XML! !IR_MODEL_MLxBench_XML!
     call ren !IR_MODEL_BIN! !IR_MODEL_MLxBench_BIN!
	 call cd %CUR_PATH%
  )
  call xcopy !IR_PATH! !AIXPRT_MODELS!\!MODEL_DEST! /s /e /y /q
  
  rd /s /q !IR_PATH!
  if exist !IR_PATH! rd /s /q !IR_PATH!
)

::========== Build Classification and Detection binaries =========
::                   Step 4. Build samples
::================================================================

set "SOURCE_DIR=%CUR_PATH%\src\"
set "BUILD_FOLDER=%CUR_PATH%\build"

if exist "C:\Program Files (x86)\Microsoft Visual Studio\2019\BuildTools\MSBuild\Current\Bin\MSBuild.exe" (
   set "MSBUILD_BIN=C:\Program Files (x86)\Microsoft Visual Studio\2019\BuildTools\MSBuild\Current\Bin\MSBuild.exe"
   set "VS_VERSION=16 2019"
)
if exist "C:\Program Files (x86)\Microsoft Visual Studio\2019\Community\MSBuild\Current\Bin\MSBuild.exe" (
   set "MSBUILD_BIN=C:\Program Files (x86)\Microsoft Visual Studio\2019\Community\MSBuild\Current\Bin\MSBuild.exe"
   set "VS_VERSION=16 2019"
)
if exist "C:\Program Files (x86)\Microsoft Visual Studio\2019\Professional\MSBuild\Current\Bin\MSBuild.exe" (
   set "MSBUILD_BIN=C:\Program Files (x86)\Microsoft Visual Studio\2019\Profressional\MSBuild\Current\Bin\MSBuild.exe"
   set "VS_VERSION=16 2019"
)
if exist "C:\Program Files (x86)\MSBuild\14.0\Bin\MSBuild.exe" (
   set "MSBUILD_BIN=C:\Program Files (x86)\MSBuild\14.0\Bin\MSBuild.exe"
   set "VS_VERSION=14 2015 Win64"
)
if exist "C:\Program Files (x86)\Microsoft Visual Studio\2017\BuildTools\MSBuild\15.0\Bin\MSBuild.exe" (
   set "MSBUILD_BIN=C:\Program Files (x86)\Microsoft Visual Studio\2017\BuildTools\MSBuild\15.0\Bin\MSBuild.exe"
   set "VS_VERSION=15 2017 Win64"
)
if exist "C:\Program Files (x86)\Microsoft Visual Studio\2017\Professional\MSBuild\15.0\Bin\MSBuild.exe" (
   set "MSBUILD_BIN=C:\Program Files (x86)\Microsoft Visual Studio\2017\Professional\MSBuild\15.0\Bin\MSBuild.exe"
   set "VS_VERSION=15 2017 Win64"
)
if exist "C:\Program Files (x86)\Microsoft Visual Studio\2017\Community\MSBuild\15.0\Bin\MSBuild.exe" (
   set "MSBUILD_BIN=C:\Program Files (x86)\Microsoft Visual Studio\2017\Community\MSBuild\15.0\Bin\MSBuild.exe"
   set "VS_VERSION=15 2017 Win64"
)


if "%VS_VERSION%" == "" (
   echo Build tools for Visual Studio 2015, 2017 or 2019 cannot be found.
   exit /b 1
)

::================================================================
xcopy "%INTEL_CVSDK_DIR%\inference_engine\samples\thirdparty" "%SOURCE_DIR%\thirdparty\" /s /e /y /q
xcopy "%INTEL_CVSDK_DIR%\inference_engine\samples\common" "%SOURCE_DIR%\common\" /s /e /y /q

::================================================================
echo %DASHES%
echo Making builds
echo %DASHES%
xcopy "%AIXPRT_SOURCES%" "%SOURCE_DIR%" /s /e /y /q
cd "%SOURCE_DIR%"
echo Creating Visual Studio %VS_VERSION% (x64) files in "%BUILD_FOLDER%"... && ^
if exist "%BUILD_FOLDER%\CMakeCache.txt" del "%BUILD_FOLDER%\CMakeCache.txt"
cmake -E make_directory "%BUILD_FOLDER%"

cd "%BUILD_FOLDER%"
call cmake -G "Visual Studio %VS_VERSION%" "%SOURCE_DIR%"

::================================================================
echo %DASHES%
echo Building solution
echo.

"%MSBUILD_BIN%" Samples.sln /p:Configuration=Release /clp:ErrorsOnly /m

::================================================================
set "COMPILED_APP_DIR=%BUILD_FOLDER%\intel64\Release"
echo %DASHES%
echo Copying compiled binaries
echo.

xcopy %COMPILED_APP_DIR%\benchmark_app.exe %AIXPRT_BIN% /s /e /y /q
REM xcopy %COMPILED_APP_DIR%\image_classification_async.exe %AIXPRT_BIN% /s /e /y /q
REM xcopy %COMPILED_APP_DIR%\object_detection_ssd.exe %AIXPRT_BIN% /s /e /y /q
REM xcopy %COMPILED_APP_DIR%\object_detection_ssd_async.exe %AIXPRT_BIN% /s /e /y /q

:: these are found after compiling AIXPRT
xcopy %COMPILED_APP_DIR%\format_reader.dll %AIXPRT_PLUGIN% /s /e /y /q
xcopy %COMPILED_APP_DIR%\cpu_extension.dll %AIXPRT_PLUGIN% /s /e /y /q

::=============== Copy libraries =================================
::              Step 6. Copy libraries
::================================================================
echo %DASHES%
echo Copying OpenVINO libraries
echo.
set OPENVINO_IE_DIR=%OPENVINO_DIR%\inference_engine
if exist %OPENVINO_IE_DIR%\bin\intel64\Release\clDNN64.dll ( xcopy %OPENVINO_IE_DIR%\bin\intel64\Release\clDNN64.dll %AIXPRT_PLUGIN% /s /e /y /q )
if exist %OPENVINO_IE_DIR%\bin\intel64\Release\HDDLPlugin.dll ( xcopy %OPENVINO_IE_DIR%\bin\intel64\Release\HDDLPlugin.dll %AIXPRT_PLUGIN% /s /e /y /q )
if exist %OPENVINO_IE_DIR%\bin\intel64\Release\cpu_extension.dll ( xcopy %OPENVINO_IE_DIR%\bin\intel64\Release\cpu_extension.dll %AIXPRT_PLUGIN% /s /e /y /q )
if exist %OPENVINO_IE_DIR%\bin\intel64\Release\clDNNPlugin.dll ( xcopy %OPENVINO_IE_DIR%\bin\intel64\Release\clDNNPlugin.dll %AIXPRT_PLUGIN% /s /e /y /q )
if exist %OPENVINO_IE_DIR%\bin\intel64\Release\HeteroPlugin.dll ( xcopy %OPENVINO_IE_DIR%\bin\intel64\Release\HeteroPlugin.dll %AIXPRT_PLUGIN% /s /e /y /q )
if exist %OPENVINO_IE_DIR%\bin\intel64\Release\inference_engine.dll ( xcopy %OPENVINO_IE_DIR%\bin\intel64\Release\inference_engine.dll %AIXPRT_PLUGIN% /s /e /y /q )
if exist %OPENVINO_IE_DIR%\bin\intel64\Release\MKLDNNPlugin.dll ( xcopy %OPENVINO_IE_DIR%\bin\intel64\Release\MKLDNNPlugin.dll %AIXPRT_PLUGIN% /s /e /y /q )
if exist %OPENVINO_IE_DIR%\bin\intel64\Release\myriadPlugin.dll ( xcopy %OPENVINO_IE_DIR%\bin\intel64\Release\myriadPlugin.dll %AIXPRT_PLUGIN% /s /e /y /q )
if exist %OPENVINO_IE_DIR%\bin\intel64\Release\plugins.xml ( xcopy %OPENVINO_IE_DIR%\bin\intel64\Release\plugins.xml %AIXPRT_PLUGIN% /s /e /y /q )


if exist %OPENVINO_IE_DIR%\bin\intel64\Release\mkl_tiny_omp.dll (
    XCOPY %OPENVINO_IE_DIR%\bin\intel64\Release\mkl_tiny_omp.dll %AIXPRT_PLUGIN% /s /e /y /q
) else (
    if exist %OPENVINO_IE_DIR%\bin\intel64\Debug\mkl_tiny_omp.dll (
        XCOPY %OPENVINO_IE_DIR%\bin\intel64\Debug\mkl_tiny_omp.dll %AIXPRT_PLUGIN% /s /e /y /q
    ) else (
		echo Cannot find mkl_tiny_omp library shipped with OpenVINO. AIXPRT may not run on your system.
	)
)
if %OPENVINO_VERSION%=="R1 and above" (
	
	XCOPY %OPENVINO_IE_DIR%\bin\intel64\Release\tbb.dll %AIXPRT_PLUGIN% /s /e /y /q
	XCOPY %OPENVINO_IE_DIR%\bin\intel64\Release\tbbmalloc.dll %AIXPRT_PLUGIN% /s /e /y /q

	if exist %OPENVINO_IE_DIR%\bin\intel64\Release\mkl_tiny_seq.dll (
		XCOPY %OPENVINO_IE_DIR%\bin\intel64\Release\mkl_tiny_seq.dll %AIXPRT_PLUGIN% /s /e /y /q
	) else (
		if exist %OPENVINO_IE_DIR%\bin\intel64\Debug\mkl_tiny_seq.dll (
			XCOPY %OPENVINO_IE_DIR%\bin\intel64\Debug\mkl_tiny_seq.dll %AIXPRT_PLUGIN% /s /e /y /q
		)
	)

	if exist %OPENVINO_IE_DIR%\bin\intel64\Release\mkl_tiny_tbb.dll (
		XCOPY %OPENVINO_IE_DIR%\bin\intel64\Release\mkl_tiny_tbb.dll %AIXPRT_PLUGIN% /s /e /y /q
	) else (
		if exist %OPENVINO_IE_DIR%\bin\intel64\Debug\mkl_tiny_tbb.dll (
			XCOPY %OPENVINO_IE_DIR%\bin\intel64\Debug\mkl_tiny_tbb.dll %AIXPRT_PLUGIN% /s /e /y /q
		)
	)
	
	if exist %OPENVINO_DIR%\deployment_tools\inference_engine\external\omp\lib\libiomp5md.dll (
		XCOPY %OPENVINO_DIR%\deployment_tools\inference_engine\external\omp\lib\libiomp5md.dll %AIXPRT_PLUGIN% /s /e /y /q
	)
		

) else (
	 
	 if exist %OPENVINO_IE_DIR%\bin\intel64\Release\libiomp5md.dll (
		 xcopy %OPENVINO_IE_DIR%\bin\intel64\Release\libiomp5md.dll %AIXPRT_PLUGIN% /s /e /y /q
	 )
)

:: copy python interface

cd %OPENVINO_DIR%\python\python3.6\openvino\inference_engine\
for /f "tokens=*" %%G in ('dir /b *.* ^| find "ie_api"') do (
	call xcopy %%G "%AIXPRT_PLUGIN%" /s /e /y /q
)

:: Copy HDDL plugins
echo Copying HDDL plugin libraries
cd "%HDDL_INSTALL_DIR%\bin"
for /f "tokens=*" %%G in ('dir /b *.* ^| find ".dll"') do (
	call xcopy %%G "%AIXPRT_PLUGIN%" /s /e /y /q
)

:: Copy OpenCV files
echo Copying OpenCV libraries
cd "%OPENVINO_DIR%\opencv\bin"
for /f "tokens=*" %%G IN ('dir /b *.* ^| find ".dll"') DO (
	call xcopy %%G "%AIXPRT_PLUGIN%" /s /e /y /q
)

:: Copy libmmd.dll (Apparently some libraries from OpenVINO depends on it :(
echo Copying libmmd from openvx
cd "%OPENVINO_DIR%\openvx\bin"
xcopy libmmd.dll "%AIXPRT_PLUGIN%" /s /e /y /q

cd %CUR_PATH%

::================================================================
::            Clean up
::================================================================
echo Removing build files
if exist "%BUILD_FOLDER%" rd /s /q "%BUILD_FOLDER%"
if exist "%BUILD_FOLDER%" rd /s /q "%BUILD_FOLDER%"

echo Removing downloaded files
if exist "%MODEL_DIR%" rd /s /q "%MODEL_DIR%"
if exist "%MODEL_DIR%" rd /s /q "%MODEL_DIR%"

echo Removing temporary source files
if exist "%SOURCE_DIR%" rd /s /q "%SOURCE_DIR%"
if exist "%SOURCE_DIR%" rd /s /q "%SOURCE_DIR%"

::================================================================


goto :eof

:usage
echo Usage:
echo  	%~n0 ^</path/to/AIXPRT^> ^</path/to/openvino^>
echo  	Requirements:
echo    		--- you have installed OpenVINO 2019 R1 and above or 2018 R5
echo    		--- you have cloned AIXPRT
exit /b 0

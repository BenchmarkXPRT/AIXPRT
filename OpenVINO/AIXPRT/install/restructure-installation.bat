set DLDT_BUILD="%~dp1"
set AIXPRT_DIR="%~dp2"
set INSTALL_DIR="C:\Program Files (x86)\IntelSWTools\ov_custom\"
set OPENVINO_DIR="C:\Program Files (x86)\IntelSWTools\openvino"

if exist %INSTALL_DIR% (
    rd /s /q %INSTALL_DIR%
)

:: =============== Create necessary directories ===================

:: 1. deployment_tools directory
mkdir %INSTALL_DIR%
mkdir %INSTALL_DIR%\deployment_tools\
mkdir %INSTALL_DIR%\deployment_tools\inference_engine
mkdir %INSTALL_DIR%\deployment_tools\model_optimizer
mkdir %INSTALL_DIR%\deployment_tools\tools
mkdir %INSTALL_DIR%\deployment_tools\demo
xcopy /s/e %DLDT_BUILD%\model-optimizer %INSTALL_DIR%\deployment_tools\model_optimizer

mkdir %INSTALL_DIR%\bin
xcopy %AIXPRT_DIR%\install\setupvars.bat %INSTALL_DIR%\bin 

:: 1.1 inference_engine subdirectory
mkdir %INSTALL_DIR%\deployment_tools\inference_engine\external
mkdir %INSTALL_DIR%\deployment_tools\inference_engine\src
mkdir %INSTALL_DIR%\deployment_tools\inference_engine\lib\intel64
mkdir %INSTALL_DIR%\deployment_tools\inference_engine\share
mkdir %INSTALL_DIR%\deployment_tools\inference_engine\include
mkdir %INSTALL_DIR%\deployment_tools\inference_engine\src\extension
mkdir %INSTALL_DIR%\deployment_tools\inference_engine\samples
mkdir %INSTALL_DIR%\deployment_tools\inference_engine\lib\intel64\Release
mkdir %INSTALL_DIR%\deployment_tools\inference_engine\bin\intel64\Release

xcopy /s/e %OPENVINO_DIR%\deployment_tools\inference_engine\share %INSTALL_DIR%\deployment_tools\inference_engine\share
xcopy /s/e %DLDT_BUILD%\inference-engine\include  %INSTALL_DIR%\deployment_tools\inference_engine\include   
xcopy /s/e %DLDT_BUILD%\inference-engine\src\extension %INSTALL_DIR%\deployment_tools\inference_engine\src\extension
xcopy /s/e %DLDT_BUILD%\inference-engine\samples %INSTALL_DIR%\deployment_tools\inference_engine\samples
xcopy /s/e %DLDT_BUILD%\inference-engine\bin\intel64\Release %INSTALL_DIR%\deployment_tools\inference_engine\lib\intel64\Release
xcopy /s/e %DLDT_BUILD%\inference-engine\bin\intel64\Release %INSTALL_DIR%\deployment_tools\inference_engine\bin\intel64\Release

if exist %DLDT_BUILD%\inference-engine\temp\omp (
    mkdir %INSTALL_DIR%\deployment_tools\inference_engine\external\omp
    xcopy /s/e %DLDT_BUILD%\inference-engine\temp\omp  %INSTALL_DIR%\deployment_tools\inference_engine\external\omp
)
if exist %DLDT_BUILD%\inference-engine\temp\tbb (
    mkdir %INSTALL_DIR%\deployment_tools\inference_engine\external\tbb
    xcopy /s/e %DLDT_BUILD%\inference-engine\temp\tbb  %INSTALL_DIR%\deployment_tools\inference_engine\external\tbb
)

if exist %INSTALL_DIR%\inference_engine( rd /s /q %INSTALL_DIR%\inference_engine )
mklink /D %INSTALL_DIR%\inference_engine %INSTALL_DIR%\deployment_tools\inference_engine

:: 1.2 tools subdirectory

REM mkdir %INSTALL_DIR%\deployment_tools\tools\open_model_zoo

if exist %INSTALL_DIR%\deployment_tools\tools\open_model_zoo (
   rd /s /q %INSTALL_DIR%\deployment_tools\tools\open_model_zoo 
)

REM if exist %DLDT_BUILD%\open_model_zoo (
REM     REM pull and revert to Oct 28th commit e372d4173e50741a6828cda415d55c37320f89cd to ensure consistency
REM     pushd %DLDT_BUILD%\open_model_zoo && git pull
REM ) else (
REM     pushd %DLDT_BUILD%
REM     git clone https://github.com/opencv/open_model_zoo.git
REM )
pushd %INSTALL_DIR%\deployment_tools\tools\
git clone https://github.com/opencv/open_model_zoo.git
REM xcopy /s/e %DLDT_BUILD%\open_model_zoo %INSTALL_DIR%\deployment_tools\tools\open_model_zoo

:: 2. OpenCV
mkdir %INSTALL_DIR%\opencv
echo Copying opencv libs
xcopy /s/e %DLDT_BUILD%\inference-engine\temp\opencv_4.1.2 %INSTALL_DIR%\opencv

:: 3. Python
mkdir %INSTALL_DIR%\python
mkdir %INSTALL_DIR%\deployment_tools\inference_engine\samples\python_samples
xcopy /s/e %DLDT_BUILD%\inference-engine\bin\intel64\Release\python_api %INSTALL_DIR%\python
xcopy /s/e  %DLDT_BUILD%\inference-engine\ie_bridges\python\sample %INSTALL_DIR%\deployment_tools\inference_engine\samples\python_samples

:: 4. OpenVX
mkdir %INSTALL_DIR%\openvx\bin
xcopy /s/e %OPENVINO_DIR%\openvx\bin\libmmd.dll %INSTALL_DIR%\openvx\bin

set INTEL_SHARE_LIB="C:\Program Files (x86)\Common Files\Intel\Shared Libraries\redist\intel64\compiler"

	if exist %INTEL_SHARE_LIB%\svml_dispmd.dll (
		XCOPY %INTEL_SHARE_LIB%\svml_dispmd.dll %AIXPRT_DIR%\Modules\Deep-Learning\packages\plugin /s /e /y /q
	)
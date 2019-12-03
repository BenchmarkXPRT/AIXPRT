@echo off


set CUR_DIR="%cd%"
set DASHES="============================================================"
set INSTALL_DIR="%LocalAppData%\Programs\Python\Python36"

echo %DASHES%
echo installing VC Redistributable
echo %DASHES%
call "%CUR_DIR%\VC_redist.x64.exe" /quiet
if errorlevel 1 (
	echo VC-Redistributable could not be installed on your system. Try running the script from administrative prompt.
	exit /b 1
)

echo.
echo %DASHES%
echo Installing python3.6.5
echo %DASHES%
call "%CUR_DIR%\python-3.6.5-amd64.exe" /quiet InstallAllUsers=0 Include_launcher=0 Include_test=0 PrependPath=1 SimpleInstall=1 SimpleInstallDescription="Just for me, no test suite." TargetDir=%INSTALL_DIR%

if errorlevel 1 (
	echo Python could not be installed. Try running the script from administrative prompt.
	exit /b 1
)

echo.
echo %DASHES%
set "PATH=%PATH%;%INSTALL_DIR%;%INSTALL_DIR%\Scripts"
:: set "PYTHONPATH=%INSTALL_DIR%;%INSTALL_DIR%\Scripts;%PYTHONPATH%"
set PYTHON_BIN=%INSTALL_DIR%\python.exe

:: Check if Python is installed correctly
%PYTHON_BIN% --version 2>NUL
if errorlevel 1 (
   echo Error^: Python is not correctly installed. Try using the python installer from the install directory.
   exit /b 1
)

copy %INSTALL_DIR%\python.exe %INSTALL_DIR%\python3.exe


:: ====================== Install Pip packages ==========================
echo Installing pip packages
call %PYTHON_BIN% -m pip install pillow numpy pywin32 wmi opencv-python --user

:: ====================== Check Pip packages ==========================
echo "Checking installed packages..."
call :check_installed_pypi_package PIL
call :check_installed_pypi_package numpy
call :check_installed_pypi_package cv2
call :check_installed_pypi_package wmi
if errorlevel 2 (
	echo Error^: Setup Unsuccessful.
	echo Please check that the pip packages are installed correctly
	exit /b 2
)

echo %DASHES%
echo Setup completed sucessfully
echo %DASHES%
timeout -1

goto :eof

:check_installed_pypi_package
setlocal
:: Check if packages are installed
set PACKAGE=%~1
%PYTHON_BIN% -c "import %PACKAGE%" > NUL
if errorlevel 1 (
	echo Error^: pip could not install %PACKAGE%
	exit /b 2
) else (
	%PYTHON_BIN% -c "print('import %PACKAGE% --- check')"
)
goto :eof

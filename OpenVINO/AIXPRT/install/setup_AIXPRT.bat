@echo off


set CUR_DIR="%cd%"
set DASHES="============================================================"
set INSTALL_DIR="%LocalAppData%\Programs\Python\Python36"

echo %DASHES%
echo installing VC Redistributable
echo %DASHES%
call "%CUR_DIR%\VC_redist.x64.exe" /quiet

echo.
echo %DASHES%
echo Installing python3.6.5
echo %DASHES%
call "%CUR_DIR%\python-3.6.5-amd64.exe" /quiet InstallAllUsers=0 Include_launcher=0 Include_test=0 PrependPath=1 SimpleInstall=1 SimpleInstallDescription="Just for me, no test suite." TargetDir=%INSTALL_DIR%

echo.
echo %DASHES%
set "PATH=%PATH%;%INSTALL_DIR%;%INSTALL_DIR%\Scripts"

copy %INSTALL_DIR%\python.exe %INSTALL_DIR%\python3.exe

echo Installing pip packages
python3 -m pip install pillow numpy pywin32 wmi opencv-python --user

echo %DASHES%
echo              Setup completed sucessfully
echo %DASHES%
timeout -1
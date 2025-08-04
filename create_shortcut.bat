@echo off
REM Создание ярлыка ATB Trading System на рабочем столе

echo Создание ярлыка ATB Trading System...

REM Получение пути к рабочему столу
for /f "tokens=2*" %%a in ('reg query "HKEY_CURRENT_USER\Software\Microsoft\Windows\CurrentVersion\Explorer\Shell Folders" /v Desktop 2^>nul') do set "DESKTOP=%%b"

REM Создание VBS скрипта для создания ярлыка
echo Set oWS = WScript.CreateObject("WScript.Shell") > "%TEMP%\CreateShortcut.vbs"
echo sLinkFile = "%DESKTOP%\ATB Trading System.lnk" >> "%TEMP%\CreateShortcut.vbs"
echo Set oLink = oWS.CreateShortcut(sLinkFile) >> "%TEMP%\CreateShortcut.vbs"
echo oLink.TargetPath = "%~dp0start_atb_desktop.bat" >> "%TEMP%\CreateShortcut.vbs"
echo oLink.WorkingDirectory = "%~dp0" >> "%TEMP%\CreateShortcut.vbs"
echo oLink.Description = "ATB Trading System - Professional Edition" >> "%TEMP%\CreateShortcut.vbs"
echo oLink.IconLocation = "%~dp0icon.ico" >> "%TEMP%\CreateShortcut.vbs"
echo oLink.Save >> "%TEMP%\CreateShortcut.vbs"

REM Выполнение VBS скрипта
cscript //nologo "%TEMP%\CreateShortcut.vbs"

REM Удаление временного файла
del "%TEMP%\CreateShortcut.vbs"

echo.
echo Ярлык "ATB Trading System" создан на рабочем столе!
echo.
echo Теперь вы можете запускать приложение двойным щелчком по ярлыку.
echo.
pause 
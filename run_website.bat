@echo off
title CardioAI — Heart Disease Prediction System
cls
echo.
echo  ============================================================
echo   CardioAI - Heart Disease Prediction System
echo  ============================================================
echo.
echo  Using Python: E:\Everything\Anaconda\python.exe
echo.
echo  Checking Flask installation...
E:\Everything\Anaconda\python.exe -m pip install flask --quiet
echo.
echo  Starting server on http://localhost:5000
echo  Press Ctrl+C to stop.
echo.
start "" "http://localhost:5000"
E:\Everything\Anaconda\python.exe app.py
pause

@echo off
echo =========================================
echo  NutriAgent - AI Nutrition Assistant
echo  Powered by IBM watsonx.ai
echo =========================================
echo.
echo Starting server at http://localhost:8501
echo Press Ctrl+C to stop
echo.
"C:\Program Files\Python311\python.exe" -m streamlit run "%~dp0app.py"
pause

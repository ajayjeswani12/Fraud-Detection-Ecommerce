@echo off
echo Installing required packages...
py -m pip install --user pandas numpy matplotlib seaborn scikit-learn xgboost lightgbm catboost
echo.
echo Running fraud detection analysis...
py fraud_detection_model.py
pause


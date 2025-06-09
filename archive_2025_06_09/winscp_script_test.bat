@echo off
FOR /L %%A IN (1000,1000,10000) DO (
  "C:\Users\agcraig\AppData\Local\Programs\WinSCP\WinSCP.exe" /log="C:\Users\agcraig\Documents\GitHub\MachineLearningNeuralMassModel\IsingModel\WinSCP.log" /ini=nul /script="C:\Users\agcraig\Documents\GitHub\MachineLearningNeuralMassModel\IsingModel\model_file_upload_script.txt" /parameter %%A
  timeout /t 5400 /nobreak
)
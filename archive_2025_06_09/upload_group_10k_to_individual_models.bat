@echo off
FOR /L %%A IN (1000,1000,32000) DO (
  "C:\Users\agcraig\AppData\Local\Programs\WinSCP\WinSCP.exe" /log="C:\Users\agcraig\Documents\GitHub\MachineLearningNeuralMassModel\IsingModel\WinSCPGroup.log" /ini=nul /script="C:\Users\agcraig\Documents\GitHub\MachineLearningNeuralMassModel\IsingModel\upload_group_10k_to_individual_models_winscp.txt" /parameter %%A
  timeout /t 5400 /nobreak
)
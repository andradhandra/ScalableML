1) Navigate to your Code folder in ScalableML
2) Put Lab3.py into your Code folder
3) Replace 'your_username' with your own username
4) Navigate to your HPC folder in ScalableML
5) Put Lab3.sh into your HPC folder
6) Make sure you have an "Output" folder
7) Run 'qsub Lab3.sh' from the HPC folder

In case of errors from the bash file:
-Use file to make sure no CRLF line endings and use to dos2unix if there are CRLF line endings (as described in the end of Lab 1)
-Make sure you have an 'Output' folder as specified in line 7 of the bash script, the script won't make the folder itself
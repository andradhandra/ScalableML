1) Navigate to your Code folder in ScalableML
2) Put lab10.py into your Code folder
3) Replace 'your_username' with your own username
4) Navigate to your HPC folder in ScalableML
5) Put lab10.sh into your HPC folder
6) Replace ýour_email' with your own email (if you want to recieve an email when it finishes running)
7) Make sure you have an "Output" folder
8) Run 'qsub lab10.sh' from the HPC folder

In case of errors from the bash file:
-Use file to make sure no CRLF line endings and use to dos2unix if there are CRLF line endings (as described in the end of Lab 1)
-Make sure you have an 'Output' folder as specified in line 7 of the bash script, the script won't make the folder itself

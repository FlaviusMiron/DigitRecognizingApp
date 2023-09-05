"""
This script was used to buid the "config.exe" executable and it is used to install the required libraries on the user's machine
It should be run via the "Run as administrator" command
This script assumes that the user has Python installed on the machine
"""
import os

has_pip = not os.system("pip -V")

has_conda = not os.system("conda list")

if has_conda:
    print("Conda is installed on this machine")
    os.system("conda install -c anaconda tk")
    os.system("conda install -c anaconda customtkinter")
    os.system("conda install -c anaconda numpy")

elif has_pip:
    print("Pip is installed on this machine")
    os.system("pip install tk")
    os.system("pip install customtkinter")
    os.system("pip install numpy")

"""
You can run this script if you want the database to take less space on the computer, by zipping it and then deleting the "extended_training_data.pkl" file,
at the cost of waitning a bit more time: The "extended_minst_loader will have to unzip it every time you train the network."
"""

import gzip

with open("expanded_training_data.pkl","rb") as input_file, gzip.open("expanded_training_data.pkl.gz","wb") as output_file:
    output_file.writelines(input_file)

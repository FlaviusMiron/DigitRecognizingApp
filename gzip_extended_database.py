import gzip

with open("expanded_training_data.pkl","rb") as input_file, gzip.open("expanded_training_data.pkl.gz","wb") as output_file:
    output_file.writelines(input_file)
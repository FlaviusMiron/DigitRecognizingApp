"""Data loader used to load the extended data set."""

import minst_loader
import pickle
import gzip
import os

class MissingDatabaseError(BaseException):
    pass

def load_data_wrapper():
    """This method is not technically a wrapper but has been named like that to be consistent with the normal data loader"""
    training_data_none, validation_data, test_data = minst_loader.load_data_wrapper()
    # f = gzip.open('expanded_training_data.pkl.gz', 'rb')
    # training_data = pickle.load(f, encoding="latin1")
    if os.path.isfile("expanded_training_data.pkl"):
        file = open("expanded_training_data.pkl","rb")
        training_data = pickle.load(file)
        
    elif os.path.isfile("expanded_training_data.pkl.gz"):
        print('Un-zipping "expanded_training_data.pkl.gz", might take extra time. You can un-zip it to prevent this from happening.')
        f = gzip.open('expanded_training_data.pkl.gz', 'rb')
        training_data = pickle.load(f, encoding="latin1")
        
    else:
        raise MissingDatabaseError('Extended database missing. Run the "process_images.py" script to create it.')

    return (training_data, validation_data, test_data)

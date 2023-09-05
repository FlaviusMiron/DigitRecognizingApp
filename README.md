# DigitRecognizingApp
An application that lets you draw digits, and the recognizes them using machine learning. 
Was created while reading through the book of Michael Nielsen, link here:
http://neuralnetworksanddeeplearning.com/index.html
The book is an amazing resource on the topic, and most of the inspiration came from there. Also, the "minst_loader.py" data loader was taken from Michael Nielsen's github repos.

Note that it comes with a pretrained model, whom parameters are stored in the "parameters.pkl". This file will be over-written with other parameters as you train the network.

I have added the "extended_training_data.pkl.gz" file, that contains an extended database of the minst. I created this database with the "process_images.py" script, that is by no means optimal,
as that was not it's purpose. For every image in the original minst database i added a version of it that is rotate by 5 degrees to the left, to the right, translated by 4 pixels to the lest,
to the right and one that has added gaussian noise on it. The original 50.000 images database was extended to 300.000. I was hoping that by doing this the modell would generalize better digits
drawn by the user, as they might differ from the minst digits.

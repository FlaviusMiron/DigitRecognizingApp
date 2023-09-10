# DigitRecognizingApp
An application that lets you draw digits, and then recognizes them using machine learning. 
Was created while reading through the book of Michael Nielsen, link here:
http://neuralnetworksanddeeplearning.com/index.html
The book is an amazing resource on the topic, and most of the inspiration came from there. Also, the "minst_loader.py" data loader was taken from Michael Nielsen's github repos.

Note that it comes with a pretrained model, whose parameters are stored in the "parameters.pkl". This file will be over-written with other parameters as you train the network.

I have created an extended database with the "process_images.py" script, that is by no means optimal, as that was not its purpose. For every image in the original minst database I added a version of it that is rotated by 5 degrees to the left, to the right, translated by 4 pixels to the left, to the right and one that has added gaussian noise into it. The original 50.000 images database was extended to 300.000. I was hoping that by doing this the model would generalize better the digits drawn by the user, as they might differ from the minst digits. I couldn't upload the whole database here since it is too big, even archived. Therefore, if you want to use this database, you will need to generate the database on your machine by running the "process_images.py" script once.
Note that the program will not be able to recognize every digit you draw, and it is normal, as it was trained on a structured, special-formatted database. Every aspect could influence the recognition, like
the dimension of the digits, the positioning in the drawing space and even different ways of drawing a single digit (ways that maybe he hasn't seen in the training data). Therefore i will in the future adress this
issue by building a feedback loop and making it learn new drawn digits that he classified as wrong.

The network achieves over 98% accucary on unseen, new data, but can easily fail to classify custom drawn digits, for the reasons mentioned above.

The "config.exe" file will install the required modules on your computer, and should be run via the "rus as administrator" option.

Also, ghostscript has to be manually downloaded and added to the path envoirment variable in order for python's imaging library to work properly:
https://www.ghostscript.com/releases/index.html

![image](https://github.com/FlaviusMiron/DigitRecognizingApp/assets/100422650/19e6d8f6-afaa-4c15-b287-56a7590de022)


"""
This scipt is used to expand the images in the minst dataset, via rotation, translation and noise adding.
The puprpose is improving the generalization of the neural network, forcing it to learn patters specific to digits in general,
not just patterns in the training data.
The scirpt will generate a ".pkl" file. In order to use this database, read the docstring of the "model_training.py" script.

"""
import minst_loader
import numpy as np
from PIL import Image
import pickle

training_data, validation_data, test_data = minst_loader.load_data_wrapper()
expanded_training_data = []

for image_array, labels in training_data:

    pixel_map = image_array
    new_pixel_map = [(int(i[0] * 255),int(i[0] * 255),int(i[0] * 255)) for i in pixel_map]
    new_pixel_map_matrix = [new_pixel_map[k:k+28] for k in range(0,784,28)]
    pixel_array = np.array(new_pixel_map_matrix, dtype=np.uint8)
    new_image = Image.fromarray(pixel_array)


    rotated_image1 =new_image.rotate(10)
    rotated_image2 =new_image.rotate(-10)

    transalted_left = Image.new('RGB',(28,28))
    transalted_left.paste(new_image,(-4,0))

    transalted_right = Image.new('RGB',(28,28))
    transalted_right.paste(new_image,(4,0))

    image_array = np.array(new_image)
    gaussian_noise = np.random.normal(0, 0.8, image_array.shape[:2]).astype(np.uint)
    noisy_image_array = np.clip(image_array.astype(np.int16) + gaussian_noise[:, :, np.newaxis], 0, 255).astype(np.uint8)
    noisy_image = Image.fromarray(noisy_image_array)


    # new_image.save("original_image.png")
    # rotated_image1.save("rotated_image1.png")
    # rotated_image2.save("rotated_image2.png")
    # transalted_left.save("left.png")
    # transalted_right.save("right.png")
    # noisy_image.save("noise.png")

    #img = Image.open("original_image.png")
    pixel_map1 = list(new_image.getdata())
    pixel_map_for_network = np.array([[pixel[0]/255] for pixel in pixel_map1])
    expanded_training_data.append((pixel_map_for_network,labels))

    pixel_map_r1 = list(rotated_image1.getdata())
    pixel_map_for_network_r1 = np.array([[pixel[0]/255] for pixel in pixel_map_r1])
    expanded_training_data.append((pixel_map_for_network_r1,labels))

    pixel_map_r2 = list(rotated_image2.getdata())
    pixel_map_for_network_r2 = np.array([[pixel[0]/255] for pixel in pixel_map_r2])
    expanded_training_data.append((pixel_map_for_network_r2,labels))

    pixel_map_t1 = list(transalted_left.getdata())
    pixel_map_for_network_t1 = np.array([[pixel[0]/255] for pixel in pixel_map_t1])
    expanded_training_data.append((pixel_map_for_network_t1,labels))

    pixel_map_t2 = list(transalted_right.getdata())
    pixel_map_for_network_t2 = np.array([[pixel[0]/255] for pixel in pixel_map_t2])
    expanded_training_data.append((pixel_map_for_network_t2,labels))

    pixel_map_n = list(transalted_right.getdata())
    pixel_map_for_network_n = np.array([[pixel[0]/255] for pixel in pixel_map_n])
    expanded_training_data.append((pixel_map_for_network_n,labels))


f = open("expanded_training_data.pkl","wb")
pickle.dump(expanded_training_data,f)
f.close()

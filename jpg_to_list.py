"""
Script for converting an image in jpg format to a list of numbers recognizable by the network.
Other scripts will make use of this script.
"""

from PIL import Image

def convert(file_name):
    img = Image.open(file_name)

    pixel_map = list(img.getdata())

    pixel_map_for_network = [[pixel[0]/255] for pixel in pixel_map]

    return pixel_map_for_network
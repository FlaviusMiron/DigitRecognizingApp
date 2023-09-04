"""
Drawing application that recognizes the digit you draw using machine learning. It recognizes only digits from 0 to 9. Comes with a pretrained model,
but can also be trained locally on your machine. Uses the MNIST handwritten digit database to train. In order to train the model, run the 
"model_training.py" script and the best parameters from the training epochs will be associated to the app on the next run.
After drawing, clicking on "see prediction" will cause the script to take the pixel map of the digit you've drawn, format it and feed it to the
neural network. Afterwards, the results are showed.

I plan on imrpovint the network's training in the following time.
"""



import customtkinter as ctk
import tkinter as tk
import feed_forward_image
import numpy as np
import time

from PIL import Image, ImageOps

class DrawingApp():
    def __init__(self):
        self.root = ctk.CTk()
        self.root.geometry("1400x600") 
        self.drawing_width = 25
        self.root.title("Drawing Adventure")
        self.place_widgets()
        self.old_x = None
        self.ond_y = None
        np.set_printoptions(precision=8, suppress=True)


    def place_widgets(self):
        self.fb = ctk.CTkLabel(self.root, text = "Width size")
        self.fb.place(relx = 0.01,rely = 0.01)

        self.plus_button = ctk.CTkButton(self.root,text="+",command = self.increase_width)
        self.plus_button.place(relx = 0.06 , rely = 0.01, relheight = 0.04, relwidth = 0.04)

        self.minus_button = ctk.CTkButton(self.root,text="-",command = self.decrease_width)
        self.minus_button.place(relx = 0.11 , rely = 0.01, relheight = 0.04, relwidth = 0.04)

        self.size_label = ctk.CTkLabel(self.root, text=self.drawing_width)
        self.size_label.place(relx = 0.16 , rely = 0.01, relheight = 0.04, relwidth = 0.02)

        self.save_image_button = ctk.CTkButton(self.root, text="See prediction", command=self.save_image)
        self.save_image_button.place(relx = 0.01 , rely = 0.1)

        self.prediction_label = ctk.CTkLabel(self.root, text=None)
        self.prediction_label.place(relx = 0.12 , rely = 0.1, relheight = 0.04, relwidth = 0.02)

        self.clear_canvas_button = ctk.CTkButton(self.root, text="Clear canvas", command=self.clear_canvas)
        self.clear_canvas_button.place(relx = 0.01 , rely = 0.19)

        self.confidence_label = ctk.CTkLabel(self.root, text="Confidence levels:")
        self.confidence_label.place(relx = 0.01 , rely = 0.29, relheight = 0.04, relwidth = 0.10)

        self.confidence0_label = ctk.CTkLabel(self.root, text="0:")
        self.confidence0_label.place(relx = 0.11 , rely = 0.20, relheight = 0.04, relwidth = 0.06)

        self.confidence1_label = ctk.CTkLabel(self.root, text="1:")
        self.confidence1_label.place(relx = 0.11 , rely = 0.25, relheight = 0.04, relwidth = 0.06)

        self.confidence2_label = ctk.CTkLabel(self.root, text="2:")
        self.confidence2_label.place(relx = 0.11 , rely = 0.30, relheight = 0.04, relwidth = 0.06)

        self.confidence3_label = ctk.CTkLabel(self.root, text="3:")
        self.confidence3_label.place(relx = 0.11 , rely = 0.35, relheight = 0.04, relwidth = 0.06)

        self.confidence4_label = ctk.CTkLabel(self.root, text="4:")
        self.confidence4_label.place(relx = 0.11 , rely = 0.40, relheight = 0.04, relwidth = 0.06)

        self.confidence5_label = ctk.CTkLabel(self.root, text="5:")
        self.confidence5_label.place(relx = 0.11 , rely = 0.45, relheight = 0.04, relwidth = 0.06)

        self.confidence6_label = ctk.CTkLabel(self.root, text="6:")
        self.confidence6_label.place(relx = 0.11 , rely = 0.50, relheight = 0.04, relwidth = 0.06)

        self.confidence7_label = ctk.CTkLabel(self.root, text="7:")
        self.confidence7_label.place(relx = 0.11 , rely = 0.55, relheight = 0.04, relwidth = 0.06)

        self.confidence8_label = ctk.CTkLabel(self.root, text="8:")
        self.confidence8_label.place(relx = 0.11 , rely = 0.60, relheight = 0.04, relwidth = 0.06)

        self.confidence9_label = ctk.CTkLabel(self.root, text="9:")
        self.confidence9_label.place(relx = 0.11 , rely = 0.65, relheight = 0.04, relwidth = 0.06)



        self.confidence0_labelv = ctk.CTkLabel(self.root, text="0")
        self.confidence0_labelv.place(relx = 0.18 , rely = 0.20, relheight = 0.04, relwidth = 0.08)

        self.confidence1_labelv = ctk.CTkLabel(self.root, text="0")
        self.confidence1_labelv.place(relx = 0.18 , rely = 0.25, relheight = 0.04, relwidth = 0.08)

        self.confidence2_labelv = ctk.CTkLabel(self.root, text="0")
        self.confidence2_labelv.place(relx = 0.18 , rely = 0.30, relheight = 0.04, relwidth = 0.08)

        self.confidence3_labelv = ctk.CTkLabel(self.root, text="0")
        self.confidence3_labelv.place(relx = 0.18 , rely = 0.35, relheight = 0.04, relwidth = 0.08)

        self.confidence4_labelv = ctk.CTkLabel(self.root, text="0")
        self.confidence4_labelv.place(relx = 0.18 , rely = 0.40, relheight = 0.04, relwidth = 0.08)

        self.confidence5_labelv = ctk.CTkLabel(self.root, text="0")
        self.confidence5_labelv.place(relx = 0.18 , rely = 0.45, relheight = 0.04, relwidth = 0.08)

        self.confidence6_labelv = ctk.CTkLabel(self.root, text="0")
        self.confidence6_labelv.place(relx = 0.18 , rely = 0.50, relheight = 0.04, relwidth = 0.08)

        self.confidence7_labelv = ctk.CTkLabel(self.root, text="0")
        self.confidence7_labelv.place(relx = 0.18 , rely = 0.55, relheight = 0.04, relwidth = 0.08)

        self.confidence8_labelv = ctk.CTkLabel(self.root, text="0")
        self.confidence8_labelv.place(relx = 0.18 , rely = 0.60, relheight = 0.04, relwidth = 0.08)

        self.confidence9_labelv = ctk.CTkLabel(self.root, text="0")
        self.confidence9_labelv.place(relx = 0.18 , rely = 0.65, relheight = 0.04, relwidth = 0.08)    

    

        self.drawing_space = ctk.CTkCanvas(self.root)
        self.drawing_space.place(relx = 0.51, rely = 0.34 , relheight = 0.3, relwidth = 0.14)
        self.drawing_space.bind("<B1-Motion>", self.draw_line)
        self.drawing_space.bind("<Button-1>",self.draw_dot)
        self.drawing_space.bind("<ButtonRelease-1>",self.reset_coordinates)

    def draw_line(self,event):
        if self.old_x and self.old_y:
            self.drawing_space.create_line(self.old_x, self.old_y,event.x,event.y, width = self.drawing_width,capstyle=tk.ROUND,smooth=True,)

        self.old_x = event.x
        self.old_y = event.y

    def draw_dot(self,event):
        self.drawing_space.create_oval(event.x, event.y,event.x+1,event.y+1, width = self.drawing_width-1,)

    def reset_coordinates(self,event):
        self.old_x = None
        self.old_y = None

    def increase_width(self):
        if self.drawing_width < 100:
            self.drawing_width += 1
        self.update_width()

    def update_width(self):
        self.size_label.configure(text=self.drawing_width)
    def decrease_width(self):
        if self.drawing_width > 1:
            self.drawing_width -= 1
        self.update_width()

    def save_image(self):
        self.drawing_space.postscript(file='canvas.eps')
        img = Image.open('canvas.eps')

        resized_img = img.resize((28,28),Image.NEAREST)
        resized_inverted_img = ImageOps.invert(resized_img)
        resized_inverted_img.save("number_final.jpg")

        prediction = feed_forward_image.get_result("number_final.jpg")

        self.prediction_label.configure(text = str(prediction[0]))

        self.confidence0_labelv.configure(text = str(prediction[1][0]) )
        self.confidence1_labelv.configure(text = str(prediction[1][1]) )
        self.confidence2_labelv.configure(text = str(prediction[1][2]) )
        self.confidence3_labelv.configure(text = str(prediction[1][3]) )
        self.confidence4_labelv.configure(text = str(prediction[1][4]) )
        self.confidence5_labelv.configure(text = str(prediction[1][5]) )
        self.confidence6_labelv.configure(text = str(prediction[1][6]) )
        self.confidence7_labelv.configure(text = str(prediction[1][7]) )
        self.confidence8_labelv.configure(text = str(prediction[1][8]) )
        self.confidence9_labelv.configure(text = str(prediction[1][9]) )


    def clear_canvas(self):
        self.drawing_space.delete(tk.ALL)

if __name__ == "__main__":
    app = DrawingApp()
    app.root.mainloop()

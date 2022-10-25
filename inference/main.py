import tkinter as tk
from tkinter.colorchooser import askcolor
from PIL import ImageGrab, Image, ImageTk
import numpy as np


from inference import segmentation2space

app = tk.Tk()

def change_color():
    global color
    selected_color = askcolor(title="Tkinter Color Chooser")
    color = selected_color[1]
    print(color)

def get_size(event):
    global size
    sz = size_entry.get()
    if 10 <= sz <= 70:
        size = sz

def draw_smth(event):
    canvas.create_oval(
        event.x - size//2,
        event.y - size//2, 
        event.x + size//2, 
        event.y + size//2, 
        fill=color,
        width=0, 
    )

def save_canvas():
    x = tk.Canvas.winfo_rootx(canvas)
    y = tk.Canvas.winfo_rooty(canvas)
    w = tk.Canvas.winfo_width(canvas)
    h = tk.Canvas.winfo_height(canvas)
    ImageGrab.grab(((x, y, x+w, y+h))).save("spacegan_tmp.png")
    print('Save image in: spacegan_tmp.png')

    generated_output = segmentation2space("spacegan_tmp.png")
    image = Image.fromarray(generated_output.astype('uint8'), 'RGB').resize((500,500))
    image = ImageTk.PhotoImage(image)
    out_label.configure(image = image)
    out_label.image = image
    out_label.pack()    

        
canvas = tk.Canvas(app, bg='#000019')
canvas.config(width=500, height=500)
canvas.pack(side=tk.LEFT)
canvas.bind("<B1-Motion>", draw_smth)

color = '#00ffff'
size = 10

# Color selector:
tk.Button(
    app,
    text='Select a Color',
    command=change_color).pack(expand=False, side=tk.LEFT)


size_entry = tk.Scale(app, from_=10, to=70, command=get_size)
size_entry.pack(side=tk.LEFT)

tk.Button(
    app,
    text='Done',
    command=save_canvas).pack(expand=False,side=tk.LEFT)

black = np.zeros((500,500,3))
black = Image.fromarray(black.astype('uint8'), 'RGB')
black = ImageTk.PhotoImage(black)
out_label = tk.Label(app, image=black)
out_label.pack(side=tk.LEFT)

app.mainloop()
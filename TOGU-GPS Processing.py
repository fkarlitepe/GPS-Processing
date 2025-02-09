import tkinter as tk
from tkinter import filedialog
from tkinter.ttk import Combobox
from PIL import Image, ImageTk
import pandas as pd


def image_transparency(image_path, alpha):
    image = Image.open(image_path)
    image = image.convert("RGBA")
    alpha_image = image.split()[3].point(lambda i: i * alpha)
    image.putalpha(alpha_image)
    return ImageTk.PhotoImage(image)


def center_image(canvas, image, canvas_width, canvas_height):
    image_width = image.width()
    image_height = image.height()
    x = (canvas_width - image_width) // 2
    y = (canvas_height - image_height) // 2
    return x, y


def next_destroy():
    root1.destroy()


root1 = tk.Tk()
root1.geometry('500x400')
root1.title("Software Information")

canvas_width = 500
canvas_height = 250
canvas = tk.Canvas(root1, width=canvas_width, height=canvas_height)
canvas.pack(fill="both", expand=True)

image_path1 = "C:\\Users\\Furkan\\Desktop\\git\\phd\\GPS-Processing\\images.jpg"
bg_image1 = Image.open(image_path1)
tk_image1 = ImageTk.PhotoImage(bg_image1)

x1, y1 = center_image(canvas, tk_image1, canvas_width, canvas_height)
canvas.create_image(x1, y1, image=tk_image1, anchor="nw")

image_path2 = "C:\\Users\\Furkan\\Desktop\\git\\phd\\GPS-Processing\\images1.jpg"
transparency = 0.1
tk_image2 = image_transparency(image_path2, transparency)

x2, y2 = center_image(canvas, tk_image2, canvas_width, canvas_height)
canvas.create_image(x2, y2, image=tk_image2, anchor="nw")

info_label = tk.Label(root1, text='''The TOGU-GPS Processing software is supported by Tokat Gaziosmanpaşa University.

For detailed information about the program, you can contact Dr. Furkan KARLITEPE.

furkan.karlitepe@gop.edu.tr''')
info_label.pack()

info_button = tk.Button(root1, text="Next", command=next_destroy)
info_button.pack(pady=10)

root1.mainloop()


def browse_sp3_file(index):
    sp3_file_paths = ["", "", ""]
    sp3_file_path = filedialog.askopenfilename(filetypes=[("SP3 Files", "*.SP3")])
    sp3_files[index].delete(0, tk.END)
    sp3_files[index].insert(0, sp3_file_path)
    sp3_file_paths[index] = sp3_file_path


def browse_gps_file():
    rinex_file_path = filedialog.askopenfilename(filetypes=[("RINEX Files", "*.RNX")])
    gps_file.delete(0, tk.END)
    gps_file.insert(0, rinex_file_path)
    return rinex_file_path


def browse_ion_file():
    pass


def browse_trop_file():
    pass


def process_data(sp3_file_paths, gps_file_path):
    from produce_sp3 import combine_sp3_data
    combined_sp3_data = combine_sp3_data(*sp3_file_paths)
    combined_sp3_data.to_excel('combine_sp3.xlsx')

    from observation_rinex3 import process_gps_file
    result_gps = process_gps_file(gps_file_path)

    from sat_regression import regression_data
    result_gps, sat3_df = regression_data(combined_sp3_data, result_gps)

    from sat_ele import calculate_Cll
    cll_result = calculate_Cll(result_gps, sat3_df)

    from PPP_process import process_result_gps
    xyz_result, n3_amb, dx_1 = process_result_gps(result_gps, sat3_df, cll_result)

    xyz_result.to_excel('xyz_result.xlsx')
    pd.DataFrame(dx_1).to_excel('dx_1.xlsx')

    result_label.config(text="Processing completed. Results saved.")


def run_process():
    sp3_file_paths = [sp3_file.get() for sp3_file in sp3_files]
    gps_file_path = gps_file.get()
    process_data(sp3_file_paths, gps_file_path)


def run_process_SDBS():
    result_label.config(text="Processing SDBS data... (placeholder implementation)")


def run_process_Machine_Learning():
    result_label.config(text="Processing Machine Learning data... (placeholder implementation)")


def show_gps_frame():
    gps_frame.grid(row=6, column=1, columnspan=3, padx=5, pady=10, sticky='W')


def hide_gps_frame():
    gps_frame.grid_remove()


def show_PPP_AR_frame():
    PPP_AR_frame.grid(row=7, column=1, columnspan=3, padx=5, pady=10, sticky='W')


def hide_PPP_AR_frame():
    PPP_AR_frame.grid_remove()
    
    
def conditional_run_process():
    selected_value = kutu1.get().strip()
    if selected_value == 'PPP':
        run_process()
    elif selected_value == 'PPP-AR/SDBS':
        run_process_SDBS()
    elif selected_value == 'PPP-AR/Machine_Learning':
        run_process_Machine_Learning()
    else:
        result_label.config(text="Please select at least one 'Process Techniques' from the dropdown to run the process.")


def update_frames(event):
    selected_value = kutu1.get().strip()
    if selected_value == 'PPP':
        show_gps_frame()
        hide_PPP_AR_frame()
    elif selected_value == 'PPP-AR/SDBS':
        hide_gps_frame()
        show_PPP_AR_frame()
    else:
        hide_gps_frame()
        hide_PPP_AR_frame()


root = tk.Tk()
root.geometry('910x750')
root.title("Data Processing")

sp3_frame = tk.LabelFrame(root, text="SP3 Files", padx=5, pady=5)
sp3_frame.grid(row=1, column=1, columnspan=3, padx=5, pady=10, sticky='W')

info_label = tk.Label(sp3_frame, text='''    * Satellite orbital movements (15 minute intervals) of the observation day should be selected.
    * Satellite orbital motions should be Lagrange 9th order regression according to the observation time (in seconds or 15 seconds).
    * In order to apply Lagrange 9th order regression, one day before and one day after data is needed.''',
                      anchor='w', justify='left', fg='white', bg='gray')
info_label.grid(row=1, column=1, columnspan=3, padx=5, pady=5)

global sp3_files
sp3_files = []
labels = ['before day', 'obs. day', 'next day']
for i, label in enumerate(labels):
    tk.Label(sp3_frame, text=f"Select SP3 File ({label}):").grid(row=i+2, column=1, padx=5, pady=5, sticky='W')
    sp3_file_entry = tk.Entry(sp3_frame, width=50)
    sp3_file_entry.grid(row=i+2, column=2, padx=5, pady=5, sticky='W')
    sp3_files.append(sp3_file_entry)
    browse_sp3_button = tk.Button(sp3_frame, text="Browse", command=lambda idx=i: browse_sp3_file(idx))
    browse_sp3_button.grid(row=i+2, column=3, padx=5, pady=5, sticky='W')
    
#########################################################
#########################################################
#########################################################
atmospheric_frame = tk.LabelFrame(root, text="Atmospheric Files", padx=5, pady=5)
atmospheric_frame.grid(row=3, column=1, columnspan=3, padx=5, pady=10, sticky='W')

label_3 = tk.Label(atmospheric_frame, text='Ionosphere File').grid(row=1, column=1, padx=5, pady=5, sticky='W')
ion_file = tk.Entry(atmospheric_frame, width=50)
ion_file.grid(row=1, column=2, padx=5, pady=5, sticky='W')
browse_ion_button = tk.Button(atmospheric_frame, text="Browse", command=browse_ion_file)
browse_ion_button.grid(row=1, column=3, padx=5, pady=5, sticky='W')

label_4 = tk.Label(atmospheric_frame, text='Troposphere File').grid(row=2, column=1, padx=5, pady=5, sticky='W')
trop_file = tk.Entry(atmospheric_frame, width=50)
trop_file.grid(row=2, column=2, padx=5, pady=5, sticky='W')
browse_trop_button = tk.Button(atmospheric_frame, text="Browse", command=browse_trop_file)
browse_trop_button.grid(row=2, column=3, padx=5, pady=5, sticky='W')

#########################################################
#########################################################
#########################################################
Process_frame = tk.LabelFrame(root, text='Process')
Process_frame.grid(row=5, column=1, columnspan=3, padx=5, pady=10, sticky='W')
liste = ['PPP', 'PPP-AR/SDBS']
label_1 = tk.Label(Process_frame, text='Select Process Technique:')
label_1.grid(row=1, column=1, padx=5, pady=5)
kutu1 = Combobox(Process_frame, values=liste)
kutu1.grid(row=1, column=2, padx=5, pady=5, sticky='W')
kutu1.bind("<<ComboboxSelected>>", update_frames)

#########################################################
#########################################################
#########################################################
gps_frame = tk.LabelFrame(root, text="PPP Process", padx=5, pady=5)
hide_gps_frame()

liste1 = ['Rinex_2', 'Rinex_3']
label_2 = tk.Label(gps_frame, text='Select Observation Data Type:')
label_2.grid(row=1, column=1, padx=5, pady=5)
kutu2 = Combobox(gps_frame, values=liste1)
kutu2.grid(row=1, column=2, padx=5, pady=5, sticky='W')

tk.Label(gps_frame, text="Select GPS observation File:").grid(row=2, column=1, padx=5, pady=5, sticky='W')
gps_file = tk.Entry(gps_frame, width=50)
gps_file.grid(row=2, column=2, padx=5, pady=5, sticky='W')
browse_gps_button = tk.Button(gps_frame, text="Browse", command=browse_gps_file)
browse_gps_button.grid(row=2, column=3, padx=5, pady=5, sticky='W')

#########################################################
#########################################################
#########################################################
PPP_AR_frame = tk.LabelFrame(root, text="PPP-AR Frame", padx=5, pady=5)
hide_PPP_AR_frame()

liste1 = ['Rinex_2', 'Rinex_3']
label_3 = tk.Label(PPP_AR_frame, text='Select Observation Data Type:')
label_3.grid(row=1, column=1, padx=5, pady=5)
kutu2 = Combobox(PPP_AR_frame, values=liste1)
kutu2.grid(row=1, column=2, padx=5, pady=5, sticky='W')
#•
liste1 = ['SDBS', 'Machine_Learning']
label_3 = tk.Label(PPP_AR_frame, text='Select FCB estimation Type:')
label_3.grid(row=1, column=3, padx=5, pady=5)
kutu2 = Combobox(PPP_AR_frame, values=liste1)
kutu2.grid(row=1, column=4, padx=5, pady=5, sticky='W')
#

tk.Label(PPP_AR_frame, text="Select to process GPS observation File :").grid(row=2, column=1, padx=5, pady=5, sticky='W')
gps_file = tk.Entry(PPP_AR_frame, width=50)
gps_file.grid(row=2, column=2, padx=5, pady=5, sticky='W')
browse_gps_button = tk.Button(PPP_AR_frame, text="Browse", command=browse_gps_file)
browse_gps_button.grid(row=2, column=3, padx=5, pady=5, sticky='W')
###
tk.Label(PPP_AR_frame, text="Select GPS observation File to estimate FCB :").grid(row=3, column=1, padx=5, pady=5, sticky='W')
gps_file = tk.Entry(PPP_AR_frame, width=50)
gps_file.grid(row=3, column=2, padx=5, pady=5, sticky='W')
browse_gps_button = tk.Button(PPP_AR_frame, text="Browse", command=browse_gps_file)
browse_gps_button.grid(row=3, column=3, padx=5, pady=5, sticky='W')
###
tk.Label(PPP_AR_frame, text="Select GPS observation File to estimate FCB:").grid(row=4, column=1, padx=5, pady=5, sticky='W')
gps_file = tk.Entry(PPP_AR_frame, width=50)
gps_file.grid(row=4, column=2, padx=5, pady=5, sticky='W')
browse_gps_button = tk.Button(PPP_AR_frame, text="Browse", command=browse_gps_file)
browse_gps_button.grid(row=4, column=3, padx=5, pady=5, sticky='W')
###
tk.Label(PPP_AR_frame, text="Select GPS observation File to estimate FCB:").grid(row=5, column=1, padx=5, pady=5, sticky='W')
gps_file = tk.Entry(PPP_AR_frame, width=50)
gps_file.grid(row=5, column=2, padx=5, pady=5, sticky='W')
browse_gps_button = tk.Button(PPP_AR_frame, text="Browse", command=browse_gps_file)
browse_gps_button.grid(row=5, column=3, padx=5, pady=5, sticky='W')
###
tk.Label(PPP_AR_frame, text="Select GPS observation File to estimate FCB:").grid(row=6, column=1, padx=5, pady=5, sticky='W')
gps_file = tk.Entry(PPP_AR_frame, width=50)
gps_file.grid(row=6, column=2, padx=5, pady=5, sticky='W')
browse_gps_button = tk.Button(PPP_AR_frame, text="Browse", command=browse_gps_file)
browse_gps_button.grid(row=6, column=3, padx=5, pady=5, sticky='W')
#########################################################
#########################################################
#########################################################


#########################################################
#########################################################
#########################################################
run_button = tk.Button(root, text="Run Process", command=conditional_run_process, font=("Arial", 11, "bold"))
run_button.grid(row=9, column=2, padx=5, pady=5)

result_label = tk.Label(root, text="")
result_label.grid(row=10, column=1, columnspan=3, padx=5, pady=5)

root.mainloop()

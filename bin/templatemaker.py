import tkinter as tk
from tkinter import Canvas, messagebox, Entry, Label, StringVar, OptionMenu
import pandas as pd
import string
from tkinter import font as tkfont  # this module provides utilities to work with fonts


def list_fonts():
    root = tk.Tk()  # Create a root window
    available_fonts = sorted(tkfont.families())  # Get the list of font families
    for font in available_fonts:
        print(font)
    root.destroy()
    
class SquareSelector:
    def __init__(self, master, rows=16, cols=24, cell_size=60):
        
        self.default_font = ('utopia', 18) 
        
        self.master = master
        self.rows = rows
        self.cols = cols
        self.cell_size = cell_size
        self.selected_cells = set()

        self.canvas_width = (cols + 1) * cell_size
        self.canvas_height = (rows + 1) * cell_size

        self.canvas = Canvas(master, width=self.canvas_width, height=self.canvas_height)
        self.canvas.grid(row=6, column=0, columnspan=3, pady=10)

        self.draw_grid()
        self.canvas.bind("<B1-Motion>", self.select_square)

    def draw_grid(self):
        for row in range(self.rows + 1):
            for col in range(self.cols + 1):
                x1 = col * self.cell_size
                y1 = row * self.cell_size
                x2 = x1 + self.cell_size
                y2 = y1 + self.cell_size

                if row == 0 and col > 0:
                    self.canvas.create_text(x1 + self.cell_size // 2, y1 + self.cell_size // 2, text=str(col), font=self.default_font)
                elif col == 0 and row > 0:
                    self.canvas.create_text(x1 + self.cell_size // 2, y1 + self.cell_size // 2, text=string.ascii_uppercase[row - 1], font=self.default_font)
                else:
                    self.canvas.create_rectangle(x1, y1, x2, y2, outline="black")

    def select_square(self, event):
        col = event.x // self.cell_size
        row = event.y // self.cell_size
        if 0 < col <= self.cols and 0 < row <= self.rows:
            cell = (string.ascii_uppercase[row - 1], col)
            if cell not in self.selected_cells:
                self.selected_cells.add(cell)
                x1 = col * self.cell_size
                y1 = row * self.cell_size
                x2 = x1 + self.cell_size
                y2 = y1 + self.cell_size
                self.canvas.create_rectangle(x1, y1, x2, y2, fill="green", outline="black")

    def clear_selection(self):
        self.selected_cells.clear()
        self.canvas.delete("all")  # Clear all items on the canvas
        self.draw_grid()

    def get_selected_cells(self):
        return self.selected_cells

class ExperimentPage(tk.Frame):
    def __init__(self, master):
        super().__init__(master)
        # self.master = master
        self.master.title("Experiment Page")
        # self.fnt = ('utopia', 18) 
        print(self.data)
        
        labels = [
            "Experiment Name:",
            "Barcode:",
            "Author:",
            "Description:",
            "Plate:",
            "Well Count:",
            "Image Folder:",
            "PFS Per Tile:",
            "Well Spacing:",
            "Plate Height:",
            "Imaging Pattern:",
            "Email:"
        ]

        self.entry_vars = {}
        row = 0
        for label_text in labels:
            label = Label(self, text=label_text, font=self.fnt)
            label.grid(row=row, column=0, padx=10, pady=5, sticky="e")
            var = StringVar()
            if label_text == "Well Count:":
                # Create a dropdown for Well Count
                options = ["96", "384"]
                var.set("96")
                dropdown = OptionMenu(self, var, *options)
                dropdown.config(font=self.fnt)
                dropdown.grid(row=row, column=1, padx=10, pady=5)
            elif label_text == "Imaging Pattern:":
                options = ["epi_only", "epi_dmd_per_well"]
                var.set("epi_only")
                dropdown = OptionMenu(self, var, *options)
                dropdown.config(font=self.fnt)
                dropdown.grid(row=row, column=1, padx=10, pady=5)
            else:
                if label_text == "Image Folder:":
                    var.set(r"D:\Images")
                entry = Entry(self, textvariable=var, font=self.fnt)
                entry.grid(row=row, column=1, padx=10, pady=5)
            self.entry_vars[label_text] = var
            row += 1
        # Create a button to go to the new page
        label = Label(self, text='Go to: ', font=self.fnt)
        label.grid(row=row, column=0, padx=10, pady=5, sticky="e")
        self.go_to_prev_page_button = tk.Button(self, text="Microscope Page", font=self.fnt, command=self.show_microscope_page)
        self.go_to_prev_page_button.grid(row=row, column=1, padx=10, pady=5)
        self.go_to_next_page_button = tk.Button(self, text="Plate Page", font=self.fnt, command=self.show_plate_page)
        self.go_to_next_page_button.grid(row=row, column=2, padx=10, pady=5)
        
    def show_microscope_page(self):
        self.master.switch_frame(MicroscopePage)
    def show_plate_page(self):
        self.master.switch_frame(PlatePage)
        
    def save_entry(self):
        # Retrieve the values entered by the user
        experiment_name = self.entry_vars["Experiment Name:"].get()
        barcode = self.entry_vars["Barcode:"].get()
        author = self.entry_vars["Author:"].get()
        description = self.entry_vars["Description:"].get()
        plate = self.entry_vars["Plate:"].get()
        well_count = int(self.entry_vars["Well Count:"].get())  # Convert to integer
        image_folder = self.entry_vars["Image Folder:"].get()
        pfs_per_tile = self.entry_vars["PFS Per Tile:"].get()
        well_spacing = self.entry_vars["Well Spacing:"].get()
        plate_height = self.entry_vars["Plate Height:"].get()
        imaging_pattern = self.entry_vars["Imaging Pattern:"].get()
        email = self.entry_vars["Email:"].get()

        # Create an entry dictionary or DataFrame to store the values
        entry = {
            "Experiment Name": experiment_name,
            "Barcode": barcode,
            "Author": author,
            "Description": description,
            "Plate": plate,
            "Well Count": well_count,
            "Image Folder": image_folder,
            "PFS Per Tile": pfs_per_tile,
            "Well Spacing": well_spacing,
            "Plate Height": plate_height,
            "Imaging Pattern": imaging_pattern,
            "Email": email,
        }

class PlatePage(tk.Frame):
    def __init__(self, master):
        super().__init__(master)
        self.master = master
        self.master.title("Plate Page")

        # Create a button to go back to the main page
        self.go_to_prev_page_button = tk.Button(self, text="Experiment Page", command=self.show_experiment_page)
        self.go_to_prev_page_button.pack()
        self.go_to_next_page_button = tk.Button(self, text="Plate Map Page", command=self.show_platemap_page)
        self.go_to_next_page_button.pack()

    def show_experiment_page(self):
        self.master.switch_frame(ExperimentPage)
        
    def show_platemap_page(self):
        self.master.switch_frame(PlateMapPage)
        

class PlateMapPage(tk.Frame):
    def __init__(self, master):
        super().__init__(master)
        # self.master = master
        self.master.title("Plate Map Page")
        
        self.platemap_df = pd.DataFrame(columns=["well", "celltype", "condition", "name", "dosage", "units"])

        # Entry for Celltype
        Label(self, text="Celltype:", font=self.fnt).grid(row=0, column=0, sticky="e")
        self.celltype_var = StringVar()
        Entry(self, textvariable=self.celltype_var, font=self.fnt).grid(row=0, column=1)
        # Entry for Condition
        Label(self, text="Condition:", font=self.fnt).grid(row=1, column=0, sticky="e")
        self.condition_var = StringVar()
        Entry(self, textvariable=self.condition_var, font=self.fnt).grid(row=1, column=1)

        # Entry for Dosage Name
        Label(self, text="Dose Name:", font=self.fnt).grid(row=2, column=0, sticky="e")
        self.dosage_name_var = StringVar()
        Entry(self, textvariable=self.dosage_name_var, font=self.fnt).grid(row=2, column=1)
        
        # Entry for Dosage Type
        Label(self, text="Dose Type:", font=self.fnt).grid(row=3, column=0, sticky="e")
        self.dosage_type_var = StringVar()
        Entry(self, textvariable=self.dosage_type_var, font=self.fnt).grid(row=3, column=1)

        # Entry for Dosage (float)
        Label(self, text="Dosage:", font=self.fnt).grid(row=4, column=0, sticky="e")
        self.dosage_var = StringVar()
        Entry(self, textvariable=self.dosage_var, font=self.fnt).grid(row=4, column=1)

        # Entry for Units
        Label(self, text="Units:", font=self.fnt).grid(row=5, column=0, sticky="e")
        self.units_var = StringVar()
        Entry(self, textvariable=self.units_var, font=self.fnt).grid(row=5, column=1)

        # Square Selector (10x10 grid)
        self.selector = SquareSelector(self)


        # Update DataFrame Button
        self.wells_button = tk.Button(self, text="96 <-> 384 Wells", command=self.toggle_well_count, font=self.fnt)
        self.wells_button.grid(row=7, column=0, pady=10)
        
        # Clear Selection
        self.clear_button = tk.Button(self, text="Clear Selection", command=self.selector.clear_selection, font=self.fnt)
        self.clear_button.grid(row=7, column=1, pady=10)
        
        # Update DataFrame Button
        self.update_button = tk.Button(self, text="Update DataFrame", command=self.update_df, font=self.fnt)
        self.update_button.grid(row=7, column=2, pady=10)

        # Save Button
        self.save_button = tk.Button(self, text="Save to CSV", font=self.fnt,command=self.save)
        self.save_button.grid(row=7, column=3, pady=10)
        
    def toggle_well_count(self):
        if self.selector.rows==8:
            rows=16
            cols = 24
        else:
            rows = 8
            cols=12
        print(self.selector.rows)
        self.selector.selected_cells.clear()
        self.selector.canvas.delete("all")  # Clear all items on the canvas
        self.selector = SquareSelector(self, rows=rows, cols=cols)

    def update_df(self):
        celltype = self.celltype_var.get()
        condition = self.condition_var.get()
        dosage_name = self.dosage_name_var.get()
        dosage_type = self.dosage_type_var.get()
        dosage = self.dosage_var.get()
        units = self.units_var.get()
        updates = []
        try:
            float_dosage = float(dosage)
            for cell in self.selector.get_selected_cells():
                row = cell[0]
                col = str(cell[1])
                well = row + col
                updates.append({"well":well, "celltype": celltype, "condition":condition, "name": dosage_name, "type":dosage_type, "dosage": float_dosage, "units": units})
            update_df = pd.DataFrame(updates)
            self.platemap_df = pd.concat([self.platemap_df, update_df], ignore_index=True)
            self.selector.clear_selection()
            self.dosage_name_var.set("")
            self.dosage_var.set("")
            self.units_var.set("")
        except ValueError:
            messagebox.showerror("Error", "Please enter a valid number for Dosage.")

        self.go_to_prev_page_button = tk.Button(self, text="Plate Page", command=self.show_plate_page)
        self.go_to_prev_page_button.pack()
        self.go_to_next_page_button = tk.Button(self, text="Timepoint Page", command=self.show_timepoint_page)
        self.go_to_next_page_button.pack()

    def show_plate_page(self):
        self.master.switch_frame(PlatePage)
    def show_timepoint_page(self):
        self.master.switch_frame(TimepointPage)

class TimepointPage(tk.Frame):
    def __init__(self, master):
        super().__init__(master)
        self.master = master
        self.master.title("Timepoint Page")

        self.go_to_prev_page_button = tk.Button(self, text="Plate Map Page", command=self.show_platemap_page)
        self.go_to_prev_page_button.pack()
        self.go_to_next_page_button = tk.Button(self, text="Microscope Page", command=self.show_microscope_page)
        self.go_to_next_page_button.pack()
    def show_platemap_page(self):
        self.master.switch_frame(PlateMapPage)
    def show_microscope_page(self):
        self.master.switch_frame(MicroscopePage)
        
        
class MicroscopePage(tk.Frame):
    def __init__(self, master):
        super().__init__(master)
        self.master = master
        self.master.title("Microscope Page")

        self.go_to_prev_page_button = tk.Button(self, text="Plate Map Page", command=self.show_timepoint_page)
        self.go_to_prev_page_button.pack()
        self.go_to_next_page_button = tk.Button(self, text="Experiment Page", command=self.show_experiment_page)
        self.go_to_next_page_button.pack()
    def show_timepoint_page(self):
        self.master.switch_frame(TimepointPage)
    def show_experiment_page(self):
        self.master.switch_frame(ExperimentPage)
        

class App(tk.Tk):
    def __init__(self):
        tk.Tk.__init__(self)
        self.fnt = ('utopia', 18) 

        self.title("Template Maker")
        self.data = {'created':0}
        self.experiment_frame = ExperimentPage(self)
        self.plate_frame = PlatePage(self)
        self.platemap_frame = PlateMapPage(self)
        self.timepoint_frame = TimepointPage(self)
        self.microscope_frame = MicroscopePage(self)
        
        self.save_page = tk.Button(self, text="Save Page", command=self.save_page)
        self.save_page.pack(side="top", padx=10, pady=10, fill="x")

        self.current_frame = None
        self.show_frame(self.experiment_frame)

    def switch_frame(self, frame_class):
        new_frame = frame_class(self)

        if self.current_frame is not None:
            self.current_frame.destroy()

        self.current_frame = new_frame
        self.current_frame.pack()
    
    def show_frame(self, frame):
        frame.tkraise()
    
    def save_page(self):
        print("Global Button Clicked")
        
    def save(self):
        with pd.ExcelWriter("output.xlsx", engine="xlsxwriter") as writer:
            self.platemap_df.to_excel(writer, sheet_name="experiment", index=False)
            
            # Add other data to sheets as needed (you need to replace this with your own data)
            plate_data = pd.DataFrame({
                "Column1": [1, 2, 3],
                "Column2": ['A', 'B', 'C']
            })
            plate_data.to_excel(writer, sheet_name="plate", index=False)

            platemap_data = pd.DataFrame({
                "Column1": [4, 5, 6],
                "Column2": ['D', 'E', 'F']
            })
            self.platemap_df.to_excel(writer, sheet_name="platemap", index=False)

            Timepoint_data = pd.DataFrame({
                "Column1": [7, 8, 9],
                "Column2": ['G', 'H', 'I']
            })
            Timepoint_data.to_excel(writer, sheet_name="Timepoint", index=False)

            microscope_data = pd.DataFrame({
                "Column1": [10, 11, 12],
                "Column2": ['J', 'K', 'L']
            })
            microscope_data.to_excel(writer, sheet_name="microscope", index=False)


if __name__ == "__main__":
    app = App()
    app.mainloop()

import tkinter as tk
from tkinter import Canvas, messagebox, Entry, Label, StringVar
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

class App(tk.Tk):
    def __init__(self):
        tk.Tk.__init__(self)
        large_font = ('utopia', 18) 

        self.title("Square Selector")

        self.df = pd.DataFrame(columns=["well", "celltype", "condition", "name", "dosage", "units"])

        # Entry for Celltype
        Label(self, text="Celltype:", font=large_font).grid(row=0, column=0, sticky="e")
        self.celltype_var = StringVar()
        Entry(self, textvariable=self.celltype_var, font=large_font).grid(row=0, column=1)
        # Entry for Condition
        Label(self, text="Condition:", font=large_font).grid(row=1, column=0, sticky="e")
        self.condition_var = StringVar()
        Entry(self, textvariable=self.condition_var, font=large_font).grid(row=1, column=1)

        # Entry for Dosage Name
        Label(self, text="Dose Name:", font=large_font).grid(row=2, column=0, sticky="e")
        self.dosage_name_var = StringVar()
        Entry(self, textvariable=self.dosage_name_var, font=large_font).grid(row=2, column=1)
        
        # Entry for Dosage Type
        Label(self, text="Dose Type:", font=large_font).grid(row=3, column=0, sticky="e")
        self.dosage_type_var = StringVar()
        Entry(self, textvariable=self.dosage_type_var, font=large_font).grid(row=3, column=1)

        # Entry for Dosage (float)
        Label(self, text="Dosage:", font=large_font).grid(row=4, column=0, sticky="e")
        self.dosage_var = StringVar()
        Entry(self, textvariable=self.dosage_var, font=large_font).grid(row=4, column=1)

        # Entry for Units
        Label(self, text="Units:", font=large_font).grid(row=5, column=0, sticky="e")
        self.units_var = StringVar()
        Entry(self, textvariable=self.units_var, font=large_font).grid(row=5, column=1)

        # Square Selector (10x10 grid)
        self.selector = SquareSelector(self)


        # Update DataFrame Button
        self.wells_button = tk.Button(self, text="96 <-> 384 Wells", command=self.toggle_well_count, font=large_font)
        self.wells_button.grid(row=7, column=0, pady=10)
        
        # Clear Selection
        self.clear_button = tk.Button(self, text="Clear Selection", command=self.selector.clear_selection, font=large_font)
        self.clear_button.grid(row=7, column=1, pady=10)
        
        # Update DataFrame Button
        self.update_button = tk.Button(self, text="Update DataFrame", command=self.update_df, font=large_font)
        self.update_button.grid(row=7, column=2, pady=10)

        # Save Button
        self.save_button = tk.Button(self, text="Save to CSV", font=large_font,command=self.save)
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
                updates.append({"well":well, "celltype": celltype, "condition":condition, "name": dosage_name, "kind":dosage_type, "dosage": float_dosage, "units": units})
            update_df = pd.DataFrame(updates)
            self.df = pd.concat([self.df, update_df], ignore_index=True)
            self.selector.clear_selection()
            self.dosage_name_var.set("")
            self.dosage_var.set("")
            self.units_var.set("")
        except ValueError:
            messagebox.showerror("Error", "Please enter a valid number for Dosage.")

    def save(self):
        self.df.to_csv("platemap.csv", index=False)

if __name__ == "__main__":
    app = App()
    app.mainloop()

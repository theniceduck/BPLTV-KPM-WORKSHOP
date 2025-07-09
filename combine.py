import os
import shutil
import tkinter as tk
from tkinter import filedialog, messagebox


def combine_files(directory, log_text):
    for root, dirs, files in os.walk(directory):
        part_prefixes = set()
        for file_name in files:
            if ".part" in file_name:
                part_prefix = file_name.split(".part")[0]
                part_prefixes.add(part_prefix)

        for part_prefix in part_prefixes:
            part_files = sorted([f for f in files if f.startswith(part_prefix + ".part")])
            if len(part_files) > 1:  # Ensure there are multiple parts
                combined_file = os.path.join(root, part_prefix)
                with open(combined_file, "wb") as combined:
                    for part_file in part_files:
                        part_file_path = os.path.join(root, part_file)
                        with open(part_file_path, "rb") as pf:
                            shutil.copyfileobj(pf, combined)
                        os.remove(part_file_path)  # Optionally delete the part file
                log_text.insert(tk.END, f"Combined: {combined_file}\n")
            else:
                log_text.insert(tk.END, f"Not enough parts for: {part_prefix}\n")
    log_text.insert(tk.END, "\nFile combining completed!\n")


def browse_directory(entry_field):
    directory = filedialog.askdirectory()
    if directory:
        entry_field.delete(0, tk.END)
        entry_field.insert(0, directory)


def start_combining(entry_field, log_text):
    directory = entry_field.get()
    if not directory or not os.path.isdir(directory):
        messagebox.showerror("Error", "Please select a valid directory")
        return
    log_text.delete("1.0", tk.END)  # Clear previous log
    combine_files(directory, log_text)


# GUI setup
root = tk.Tk()
root.title("File Combiner")
root.geometry("500x400")

frame = tk.Frame(root)
frame.pack(pady=10)

entry_label = tk.Label(frame, text="Select Directory:")
entry_label.grid(row=0, column=0, padx=5, pady=5)

directory_entry = tk.Entry(frame, width=40)
directory_entry.grid(row=0, column=1, padx=5, pady=5)

browse_button = tk.Button(frame, text="Browse", command=lambda: browse_directory(directory_entry))
browse_button.grid(row=0, column=2, padx=5, pady=5)

combine_button = tk.Button(root, text="Combine Files", command=lambda: start_combining(directory_entry, log_text))
combine_button.pack(pady=10)

log_text = tk.Text(root, height=15, width=60)
log_text.pack(pady=5)

root.mainloop()

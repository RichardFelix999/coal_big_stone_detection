import os
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk

class Box:
    def __init__(self, x1, y1, x2, y2, class_id):
        self.x1, self.y1, self.x2, self.y2 = x1, y1, x2, y2
        self.class_id = class_id

    def to_yolo(self, img_w, img_h):
        x_center = ((self.x1 + self.x2) / 2) / img_w
        y_center = ((self.y1 + self.y2) / 2) / img_h
        width = abs(self.x2 - self.x1) / img_w
        height = abs(self.y2 - self.y1) / img_h
        return f"{self.class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}"

class YOLOAnnotator:
    def __init__(self, root):
        self.root = root
        self.root.title("YOLO Annotator")

        # --- Class management ---
        self.classes = ["person", "car", "dog", "cat", "other"]
        self.class_var = tk.StringVar(value=self.classes[0])
        self.edit_class_var = tk.StringVar(value=self.classes[0])

        # --- Image and annotation state ---
        self.image_list = []
        self.current_image = None
        self.image_path = None
        self.tk_image = None
        self.img_dir = None
        self.labels_dir = None
        self.img_idx = 0
        self.boxes = []
        self.selected_box = None
        self.drawing = False
        self.start_x = self.start_y = None
        self.rect = None

        # --- Layout Frames ---
        self.left_frame = tk.Frame(root)
        self.left_frame.pack(side=tk.LEFT, fill=tk.Y, padx=8, pady=8)

        # --- Project Controls ---
        project_frame = tk.LabelFrame(self.left_frame, text="Project", padx=4, pady=4)
        project_frame.pack(fill=tk.X, pady=4)
        self.import_btn = tk.Button(project_frame, text="Import Folder", command=self.import_folder)
        self.import_btn.pack(fill=tk.X, pady=2)
        self.export_btn = tk.Button(project_frame, text="Export Labels", command=self.export_label)
        self.export_btn.pack(fill=tk.X, pady=2)

        # --- Class Management ---
        class_frame = tk.LabelFrame(self.left_frame, text="Class Management", padx=4, pady=4)
        class_frame.pack(fill=tk.X, pady=4)
        self.class_listbox = tk.Listbox(class_frame, height=5)
        self.class_listbox.pack(fill=tk.X, pady=2)
        self.class_listbox.bind("<<ListboxSelect>>", self.on_class_listbox_select)
        self.class_entry = tk.Entry(class_frame)
        self.class_entry.pack(fill=tk.X, pady=2)
        self.add_class_btn = tk.Button(class_frame, text="Add Class", command=self.add_class)
        self.add_class_btn.pack(fill=tk.X, pady=2)
        self.edit_class_entry = tk.Entry(class_frame)
        self.edit_class_entry.pack(fill=tk.X, pady=2)
        self.rename_class_btn = tk.Button(class_frame, text="Rename Class", command=self.rename_class)
        self.rename_class_btn.pack(fill=tk.X, pady=2)
        self.delete_class_btn = tk.Button(class_frame, text="Delete Class", command=self.delete_class)
        self.delete_class_btn.pack(fill=tk.X, pady=2)

        # --- Annotation Tools ---
        annot_frame = tk.LabelFrame(self.left_frame, text="Annotation Tools", padx=4, pady=4)
        annot_frame.pack(fill=tk.X, pady=4)
        tk.Label(annot_frame, text="Select Class:").pack(fill=tk.X, pady=2)
        self.class_menu = tk.OptionMenu(annot_frame, self.class_var, *self.classes)
        self.class_menu.pack(fill=tk.X, pady=2)
        self.box_listbox = tk.Listbox(annot_frame)
        self.box_listbox.pack(fill=tk.BOTH, expand=True, pady=2)
        self.box_listbox.bind("<<ListboxSelect>>", self.on_box_select)
        tk.Label(annot_frame, text="Edit Selected Box Class:").pack(fill=tk.X, pady=2)
        self.edit_class_menu = tk.OptionMenu(annot_frame, self.edit_class_var, *self.classes)
        self.edit_class_menu.pack(fill=tk.X, pady=2)
        self.update_class_btn = tk.Button(annot_frame, text="Update Class", command=self.update_box_class)
        self.update_class_btn.pack(fill=tk.X, pady=2)
        self.delete_box_btn = tk.Button(annot_frame, text="Delete Box", command=self.delete_selected_box)
        self.delete_box_btn.pack(fill=tk.X, pady=2)

        # --- Image Navigation ---
        nav_frame = tk.LabelFrame(self.left_frame, text="Image Navigation", padx=4, pady=4)
        nav_frame.pack(fill=tk.X, pady=4)
        self.prev_btn = tk.Button(nav_frame, text="<< Prev", command=self.prev_image)
        self.prev_btn.pack(fill=tk.X, pady=2)
        self.next_btn = tk.Button(nav_frame, text="Next >>", command=self.next_image)
        self.next_btn.pack(fill=tk.X, pady=2)

        self.info_label = tk.Label(self.left_frame, text="Keys: d=delete, n=next, p=prev")
        self.info_label.pack(fill=tk.X, pady=8)

        # --- Canvas for Image ---
        self.canvas = tk.Canvas(root, width=640, height=640, bg='gray')
        self.canvas.pack(side=tk.RIGHT, padx=8, pady=8)
        self.canvas.bind("<ButtonPress-1>", self.on_mouse_down)
        self.canvas.bind("<B1-Motion>", self.on_mouse_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_mouse_up)
        self.canvas.bind("<Button-3>", self.on_right_click)

        # --- Keyboard shortcuts ---
        self.root.bind("<d>", lambda e: self.delete_selected_box())
        self.root.bind("<n>", lambda e: self.next_image())
        self.root.bind("<p>", lambda e: self.prev_image())

        # Now it's safe to update class list and menus!
        self.update_class_listbox()

    # --- Class management ---
    def update_class_listbox(self):
        self.class_listbox.delete(0, tk.END)
        for c in self.classes:
            self.class_listbox.insert(tk.END, c)
        self.update_class_menus()

    def update_class_menus(self):
        menu = self.class_menu["menu"]
        menu.delete(0, "end")
        for c in self.classes:
            menu.add_command(label=c, command=lambda v=c: self.class_var.set(v))
        menu2 = self.edit_class_menu["menu"]
        menu2.delete(0, "end")
        for c in self.classes:
            menu2.add_command(label=c, command=lambda v=c: self.edit_class_var.set(v))
        # Set current values if possible
        if self.class_var.get() not in self.classes:
            self.class_var.set(self.classes[0])
        if self.edit_class_var.get() not in self.classes:
            self.edit_class_var.set(self.classes[0])

    def add_class(self):
        new_class = self.class_entry.get().strip()
        if new_class and new_class not in self.classes:
            self.classes.append(new_class)
            self.update_class_listbox()
            self.class_entry.delete(0, tk.END)

    def on_class_listbox_select(self, event):
        idx = self.class_listbox.curselection()
        if idx:
            self.edit_class_entry.delete(0, tk.END)
            self.edit_class_entry.insert(0, self.classes[idx[0]])

    def rename_class(self):
        idx = self.class_listbox.curselection()
        if idx:
            new_name = self.edit_class_entry.get().strip()
            if new_name and new_name not in self.classes:
                old_name = self.classes[idx[0]]
                old_idx = idx[0]
                self.classes[old_idx] = new_name
                # Update all boxes that use this class
                for box in self.boxes:
                    if box.class_id == old_idx:
                        box.class_id = self.classes.index(new_name)
                self.update_class_listbox()
                self.update_box_listbox()
                self.update_canvas()

    def delete_class(self):
        idx = self.class_listbox.curselection()
        if idx:
            class_id = idx[0]
            # Check if in use
            if any(box.class_id == class_id for box in self.boxes):
                messagebox.showwarning("Cannot delete", "Class is in use by an annotation.")
                return
            del self.classes[class_id]
            self.update_class_listbox()

    # --- Image and annotation ---
    def import_folder(self):
        folder = filedialog.askdirectory(title="Select Image Folder")
        if not folder:
            return
        self.img_dir = folder
        parent_dir = os.path.dirname(self.img_dir)
        self.labels_dir = os.path.join(parent_dir, "labels")
        os.makedirs(self.labels_dir, exist_ok=True)
        self.image_list = [os.path.join(folder, f) for f in os.listdir(folder)
                           if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        self.image_list.sort()
        self.img_idx = 0
        if self.image_list:
            self.load_image(self.image_list[self.img_idx])

    def load_image(self, path):
        self.image_path = path
        self.current_image = Image.open(path).convert("RGB").resize((640, 640))
        self.tk_image = ImageTk.PhotoImage(self.current_image)
        self.boxes = self.load_boxes_for_image(path)
        self.selected_box = None
        self.update_canvas()
        self.update_box_listbox()
        self.root.title(f"YOLO Annotator - {os.path.basename(path)} [{self.img_idx+1}/{len(self.image_list)}]")

    def update_canvas(self):
        self.canvas.delete("all")
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.tk_image)
        for idx, box in enumerate(self.boxes):
            color = "red" if idx == self.selected_box else "yellow"
            class_name = self.classes[box.class_id] if box.class_id < len(self.classes) else "?"
            self.canvas.create_rectangle(box.x1, box.y1, box.x2, box.y2, outline=color, width=2)
            self.canvas.create_text(box.x1+4, box.y1+10, anchor=tk.NW, text=class_name, fill=color, font=("Arial", 10, "bold"))
        if self.drawing and self.rect:
            self.canvas.lift(self.rect)

    def update_box_listbox(self):
        self.box_listbox.delete(0, tk.END)
        for idx, box in enumerate(self.boxes):
            class_name = self.classes[box.class_id] if box.class_id < len(self.classes) else "?"
            self.box_listbox.insert(tk.END, f"{class_name}: ({int(box.x1)},{int(box.y1)})-({int(box.x2)},{int(box.y2)})")

    def load_boxes_for_image(self, img_path):
        label_path = os.path.join(self.labels_dir, os.path.splitext(os.path.basename(img_path))[0] + ".txt")
        boxes = []
        if os.path.exists(label_path):
            with open(label_path, "r") as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) != 5:
                        continue
                    class_id = int(parts[0])
                    x_center, y_center, width, height = map(float, parts[1:])
                    img_w, img_h = 640, 640
                    x1 = (x_center - width/2) * img_w
                    y1 = (y_center - height/2) * img_h
                    x2 = (x_center + width/2) * img_w
                    y2 = (y_center + height/2) * img_h
                    boxes.append(Box(x1, y1, x2, y2, class_id))
        return boxes

    def save_boxes_for_image(self):
        if not self.image_path or not self.labels_dir:
            return
        label_path = os.path.join(self.labels_dir, os.path.splitext(os.path.basename(self.image_path))[0] + ".txt")
        img_w, img_h = 640, 640
        with open(label_path, "w") as f:
            for box in self.boxes:
                f.write(box.to_yolo(img_w, img_h) + "\n")

    def export_label(self):
        self.save_boxes_for_image()
        messagebox.showinfo("Exported", f"Labels saved to {self.labels_dir}")

    def next_image(self):
        if not self.image_list:
            return
        self.save_boxes_for_image()
        self.img_idx = (self.img_idx + 1) % len(self.image_list)
        self.load_image(self.image_list[self.img_idx])

    def prev_image(self):
        if not self.image_list:
            return
        self.save_boxes_for_image()
        self.img_idx = (self.img_idx - 1) % len(self.image_list)
        self.load_image(self.image_list[self.img_idx])

    # --- Drawing boxes ---
    def on_mouse_down(self, event):
        self.drawing = True
        self.start_x, self.start_y = event.x, event.y
        self.rect = self.canvas.create_rectangle(self.start_x, self.start_y, self.start_x, self.start_y, outline='red', width=2)

    def on_mouse_drag(self, event):
        if self.drawing and self.rect:
            self.canvas.coords(self.rect, self.start_x, self.start_y, event.x, event.y)

    def on_mouse_up(self, event):
        if not self.drawing:
            return
        end_x, end_y = event.x, event.y
        x1, y1 = min(self.start_x, end_x), min(self.start_y, end_y)
        x2, y2 = max(self.start_x, end_x), max(self.start_y, end_y)
        if abs(x2-x1) > 10 and abs(y2-y1) > 10:
            class_id = self.classes.index(self.class_var.get())
            self.boxes.append(Box(x1, y1, x2, y2, class_id))
            self.update_box_listbox()
        self.drawing = False
        if self.rect:
            self.canvas.delete(self.rect)
            self.rect = None
        self.update_canvas()

    def on_box_select(self, event):
        idx = self.box_listbox.curselection()
        if idx:
            self.selected_box = idx[0]
            box = self.boxes[self.selected_box]
            class_name = self.classes[box.class_id] if box.class_id < len(self.classes) else self.classes[0]
            self.edit_class_var.set(class_name)
            self.update_canvas()

    def update_box_class(self):
        if self.selected_box is not None and 0 <= self.selected_box < len(self.boxes):
            new_class = self.edit_class_var.get()
            self.boxes[self.selected_box].class_id = self.classes.index(new_class)
            self.update_box_listbox()
            self.update_canvas()

    def delete_selected_box(self):
        if self.selected_box is not None and 0 <= self.selected_box < len(self.boxes):
            del self.boxes[self.selected_box]
            self.selected_box = None
            self.update_box_listbox()
            self.update_canvas()

    def on_right_click(self, event):
        # Select box under mouse for delete/edit
        for idx, box in enumerate(self.boxes):
            if box.x1 <= event.x <= box.x2 and box.y1 <= event.y <= box.y2:
                self.selected_box = idx
                class_name = self.classes[box.class_id] if box.class_id < len(self.classes) else self.classes[0]
                self.edit_class_var.set(class_name)
                self.update_canvas()
                if messagebox.askyesno("Delete", "Delete this box?"):
                    self.delete_selected_box()
                break

if __name__ == "__main__":
    root = tk.Tk()
    app = YOLOAnnotator(root)
    root.mainloop()

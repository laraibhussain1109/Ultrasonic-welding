"""Tkinter UI for operator/admin blower fan inspection."""

from __future__ import annotations

import queue
import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

import cv2
from PIL import Image, ImageTk

from .auth import AuthStore, User
from .camera import USBCamera, save_capture
from .config import ModelRegistry, PartModelConfig, ensure_model_folders
from .trainer import NormalTemplateTrainer


class LoginWindow(ttk.Frame):
    def __init__(self, master: tk.Tk, auth: AuthStore, on_login) -> None:
        super().__init__(master, padding=24)
        self.auth = auth
        self.on_login = on_login
        master.title("Blower Fan Inspection - Login")
        master.geometry("420x260")
        ttk.Label(self, text="Blower Fan Inspection", font=("Segoe UI", 18, "bold")).grid(row=0, column=0, columnspan=2, pady=(0, 18))
        ttk.Label(self, text="Username").grid(row=1, column=0, sticky="w")
        self.username = ttk.Entry(self)
        self.username.grid(row=1, column=1, sticky="ew", pady=4)
        ttk.Label(self, text="Password").grid(row=2, column=0, sticky="w")
        self.password = ttk.Entry(self, show="*")
        self.password.grid(row=2, column=1, sticky="ew", pady=4)
        ttk.Button(self, text="Login", command=self._login).grid(row=3, column=0, columnspan=2, sticky="ew", pady=16)
        ttk.Label(self, text="Default: admin/admin123 or operator/operator123", foreground="#666").grid(row=4, column=0, columnspan=2)
        self.columnconfigure(1, weight=1)
        self.pack(fill="both", expand=True)
        self.username.focus_set()
        master.bind("<Return>", lambda _event: self._login())

    def _login(self) -> None:
        user = self.auth.authenticate(self.username.get(), self.password.get())
        if user is None:
            messagebox.showerror("Login failed", "Invalid username or password")
            return
        self.on_login(user)


class InspectionApp(ttk.Frame):
    def __init__(self, master: tk.Tk, user: User) -> None:
        super().__init__(master, padding=12)
        self.user = user
        self.registry = ModelRegistry()
        ensure_model_folders(self.registry)
        self.trainer = NormalTemplateTrainer()
        self.camera = USBCamera()
        self.frame = None
        self.preview_photo = None
        self.events: queue.Queue[str] = queue.Queue()
        master.title(f"Blower Fan Inspection - {user.username} ({user.role})")
        master.geometry("1180x760")
        self._build()
        self.pack(fill="both", expand=True)
        self.after(250, self._poll_events)

    def _build(self) -> None:
        header = ttk.Frame(self)
        header.pack(fill="x")
        ttk.Label(header, text="Ultrasonic Weld Blower Fan Inspection", font=("Segoe UI", 16, "bold")).pack(side="left")
        ttk.Label(header, text=f"Logged in: {self.user.username} / {self.user.role}").pack(side="right")

        body = ttk.Frame(self)
        body.pack(fill="both", expand=True, pady=10)
        left = ttk.LabelFrame(body, text="Camera / Image", padding=10)
        left.pack(side="left", fill="both", expand=True)
        right = ttk.LabelFrame(body, text="Controls", padding=10)
        right.pack(side="right", fill="y", padx=(10, 0))

        self.preview = ttk.Label(left, text="Open camera or load an image", anchor="center")
        self.preview.pack(fill="both", expand=True)

        models = self.registry.all()
        self.model_by_label = {f"{m.id} - {m.name}": m for m in models}
        ttk.Label(right, text="Part model").pack(anchor="w")
        self.model_var = tk.StringVar(value=next(iter(self.model_by_label)))
        ttk.Combobox(right, textvariable=self.model_var, values=list(self.model_by_label), state="readonly", width=34).pack(fill="x", pady=(0, 10))

        ttk.Button(right, text="Open Camera", command=self._open_camera).pack(fill="x", pady=3)
        ttk.Button(right, text="Grab Frame", command=self._grab_frame).pack(fill="x", pady=3)
        ttk.Button(right, text="Load Image", command=self._load_image).pack(fill="x", pady=3)
        ttk.Separator(right).pack(fill="x", pady=8)
        ttk.Button(right, text="Inspect", command=self._inspect).pack(fill="x", pady=3)
        ttk.Button(right, text="Capture Normal Sample", command=self._capture_normal).pack(fill="x", pady=3)
        if self.user.is_admin:
            ttk.Button(right, text="Train New/Selected Model", command=self._train).pack(fill="x", pady=3)
        else:
            ttk.Label(right, text="Training is admin-only", foreground="#666").pack(anchor="w", pady=6)
        ttk.Separator(right).pack(fill="x", pady=8)
        ttk.Button(right, text="Open Training Folder", command=self._open_training_folder).pack(fill="x", pady=3)

        self.status = tk.StringVar(value="Ready")
        ttk.Label(right, textvariable=self.status, wraplength=280, foreground="#064").pack(anchor="w", pady=12)
        self.result_text = tk.Text(right, height=12, width=42)
        self.result_text.pack(fill="both", expand=True)

    def selected_model(self) -> PartModelConfig:
        return self.model_by_label[self.model_var.get()]

    def _open_camera(self) -> None:
        try:
            self.camera.open()
            self.status.set("Camera opened. Click Grab Frame to capture.")
        except Exception as exc:
            messagebox.showerror("Camera error", str(exc))

    def _grab_frame(self) -> None:
        try:
            self.frame = self.camera.read()
            self._show_frame(self.frame)
            self.status.set("Frame captured")
        except Exception as exc:
            messagebox.showerror("Capture error", str(exc))

    def _load_image(self) -> None:
        path = filedialog.askopenfilename(filetypes=[("Images", "*.png *.jpg *.jpeg *.bmp *.tif *.tiff")])
        if not path:
            return
        frame = cv2.imread(path)
        if frame is None:
            messagebox.showerror("Image error", f"Unable to read {path}")
            return
        self.frame = frame
        self._show_frame(frame)
        self.status.set(f"Loaded {Path(path).name}")

    def _show_frame(self, frame) -> None:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(rgb)
        image.thumbnail((780, 610))
        self.preview_photo = ImageTk.PhotoImage(image)
        self.preview.configure(image=self.preview_photo, text="")

    def _inspect(self) -> None:
        if self.frame is None:
            messagebox.showwarning("No image", "Capture or load an image first")
            return
        try:
            result = self.trainer.inspect(self.selected_model(), self.frame)
        except Exception as exc:
            messagebox.showerror("Inspection error", str(exc))
            return
        self.result_text.insert("end", f"{result.status}: score={result.anomaly_score:.2f}, area={result.defect_area_px}, bad sectors={result.bad_sectors}\n")
        self.result_text.see("end")
        self.status.set(f"Inspection {result.status}. Overlay: {result.overlay_path}")
        if result.overlay_path:
            overlay = cv2.imread(str(result.overlay_path))
            if overlay is not None:
                self._show_frame(overlay)

    def _capture_normal(self) -> None:
        if self.frame is None:
            messagebox.showwarning("No image", "Capture or load an image first")
            return
        path = save_capture(self.frame, self.selected_model().normal_image_dir, "normal")
        self.status.set(f"Saved normal sample: {path}")

    def _train(self) -> None:
        model = self.selected_model()
        self.status.set(f"Training {model.id} in background...")
        threading.Thread(target=self._train_worker, args=(model,), daemon=True).start()

    def _train_worker(self, model: PartModelConfig) -> None:
        try:
            output = self.trainer.train(model)
            self.events.put(f"Training complete for {model.id}: {output}")
        except Exception as exc:
            self.events.put(f"Training failed for {model.id}: {exc}")

    def _poll_events(self) -> None:
        while not self.events.empty():
            message = self.events.get_nowait()
            self.status.set(message)
            self.result_text.insert("end", message + "\n")
        self.after(250, self._poll_events)

    def _open_training_folder(self) -> None:
        path = self.selected_model().normal_image_dir
        path.mkdir(parents=True, exist_ok=True)
        messagebox.showinfo("Training folder", str(path.resolve()))


def main() -> None:
    root = tk.Tk()
    auth = AuthStore()

    def on_login(user: User) -> None:
        for child in root.winfo_children():
            child.destroy()
        InspectionApp(root, user)

    LoginWindow(root, auth, on_login)
    root.mainloop()


if __name__ == "__main__":
    main()

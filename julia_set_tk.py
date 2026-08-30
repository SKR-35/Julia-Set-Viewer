#!/usr/bin/env python3
"""Interactive Julia Set viewer with Tkinter.

Features: rectangle zoom, pan, smooth coloring, preset constants,
auto-iteration scaling, optional Numba acceleration, PNG export and 5-second GIF export.
"""
from __future__ import annotations

import math
import os
import time
import threading
import tkinter as tk
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from tkinter import filedialog, messagebox, ttk
from typing import Optional, Tuple

import numpy as np

try:
    from numba import njit, prange

    HAS_NUMBA = True
except Exception:
    HAS_NUMBA = False

try:
    from PIL import Image, ImageTk
except Exception as exc:
    raise SystemExit("Pillow is required. Install with: pip install pillow") from exc

try:
    import matplotlib

    HAVE_MPL = True
except Exception:
    HAVE_MPL = False


CMAPS = [
    "Jade",
    "Pearl",
    "plasma",
    "viridis",
    "magma",
    "inferno",
    "turbo",
    "cividis",
    "twilight",
    "SoftSunset",
    "EarthAndSky",
    "Seashore",
    "Forest",
    "HotAndCold",
    "Pastel",
    "Grayscale",
]
DEFAULT_CMAP = "plasma"
MIN_SCALE = 1e-18

CUSTOM_STOPS = {
    "Jade": ["#020b08", "#073b2a", "#0b6b4f", "#13a777", "#62d6a7", "#d8fff0"],
    "Pearl": ["#101214", "#34383d", "#6f747a", "#b8bcc1", "#ece9e2", "#fffdf7"],
    "SoftSunset": ["#2b1055", "#6a0572", "#ff6f91", "#ffc15e", "#ffe29a"],
    "EarthAndSky": ["#1a2a6c", "#28a0b0", "#84ffc9", "#f0f3bd", "#ffd166"],
    "Seashore": ["#001219", "#005f73", "#0a9396", "#94d2bd", "#e9d8a6"],
    "Forest": ["#0b3d0b", "#236e3c", "#4caf50", "#a8e6cf", "#f1f8e9"],
    "HotAndCold": [
        "#313695",
        "#4575b4",
        "#74add1",
        "#abd9e9",
        "#fee090",
        "#f46d43",
        "#d73027",
    ],
    "Pastel": ["#b3e5fc", "#c5cae9", "#e1bee7", "#f8bbd0", "#ffe0b2", "#dcedc8"],
    "Grayscale": ["#0a0a0a", "#2f2f2f", "#5e5e5e", "#9a9a9a", "#cccccc", "#f2f2f2"],
}

JULIA_PRESETS = {
    "Classic dendrite": complex(-0.8000, 0.1560),
    "Douady rabbit": complex(-0.1230, 0.7450),
    "Spiral": complex(-0.70176, -0.38420),
    "Seahorse": complex(-0.8350, -0.23210),
    "Feather": complex(-0.7269, 0.18890),
    "Galaxies": complex(-0.4000, 0.60000),
    "Fine branches": complex(0.2850, 0.01000),
}
DEFAULT_PRESET = "Classic dendrite"
DEFAULT_C = JULIA_PRESETS[DEFAULT_PRESET]


def _build_lut(stops: list[str], n: int = 1024) -> np.ndarray:
    import matplotlib.colors as mcolors

    cmap = mcolors.LinearSegmentedColormap.from_list("custom", stops, N=n)
    return (cmap(np.linspace(0, 1, n))[:, :3] * 255.0).astype(np.uint8)


_CUSTOM_LUTS = {name: _build_lut(stops) for name, stops in CUSTOM_STOPS.items()}


@dataclass
class View:
    cx: float = 0.0
    cy: float = 0.0
    scale: float = 1.5
    max_iter: int = 600

    def grid(self, width: int, height: int) -> np.ndarray:
        aspect = height / width
        x_min = self.cx - self.scale / aspect
        x_max = self.cx + self.scale / aspect
        y_min = self.cy - self.scale
        y_max = self.cy + self.scale
        xs = np.linspace(x_min, x_max, width, dtype=np.float64)
        ys = np.linspace(y_min, y_max, height, dtype=np.float64)
        x_grid, y_grid = np.meshgrid(xs, ys)
        return x_grid + 1j * y_grid


def julia_smooth(
    z0: np.ndarray,
    c: complex,
    max_iter: int,
    cancel_event: Optional[threading.Event] = None,
) -> Optional[np.ndarray]:
    if HAS_NUMBA:
        return _julia_smooth_numba(z0, c.real, c.imag, max_iter)
    return _julia_smooth_numpy(z0, c, max_iter, cancel_event)


def _julia_smooth_numpy(
    z0: np.ndarray,
    c: complex,
    max_iter: int,
    cancel_event: Optional[threading.Event] = None,
) -> Optional[np.ndarray]:
    z = z0.copy()
    iterations = np.zeros(z.shape, dtype=np.float64)
    active = np.ones(z.shape, dtype=bool)

    for n in range(max_iter):
        if cancel_event is not None and cancel_event.is_set():
            return None
        z[active] = z[active] * z[active] + c
        escaped = np.zeros(z.shape, dtype=bool)
        escaped[active] = np.abs(z[active]) > 4.0
        newly_escaped = escaped & active
        if np.any(newly_escaped):
            magnitude = np.abs(z[newly_escaped])
            iterations[newly_escaped] = n + 1 - np.log2(np.log(magnitude) + 1e-16)
        active &= ~newly_escaped
        if not active.any():
            break

    iterations[active] = float(max_iter)
    return iterations


if HAS_NUMBA:

    @njit(cache=True, fastmath=True, parallel=True)
    def _julia_smooth_numba(
        z0: np.ndarray, c_real: float, c_imag: float, max_iter: int
    ) -> np.ndarray:
        height, width = z0.shape
        output = np.zeros((height, width), dtype=np.float64)
        c = complex(c_real, c_imag)

        for row in prange(height):
            for col in range(width):
                z = z0[row, col]
                escaped = False
                for n in range(max_iter):
                    z = z * z + c
                    magnitude_squared = z.real * z.real + z.imag * z.imag
                    if magnitude_squared > 16.0:
                        magnitude = magnitude_squared**0.5
                        output[row, col] = n + 1.0 - np.log(
                            np.log(magnitude) + 1e-16
                        ) / np.log(2.0)
                        escaped = True
                        break
                if not escaped:
                    output[row, col] = float(max_iter)
        return output


def escaped_contrast(values: np.ndarray, max_iter: int) -> Tuple[float, float]:
    escaped = values < (max_iter - 1e-9)
    if np.any(escaped):
        low = float(np.percentile(values[escaped], 0.5))
        high = float(np.percentile(values[escaped], 99.5))
        if high <= low:
            low, high = float(values.min()), float(values.max())
    else:
        low, high = float(values.min()), float(values.max())
    return low, high


def map_to_rgb(
    values: np.ndarray,
    cmap_name: str,
    low: float,
    high: float,
    gamma: float = 1.35,
    smoothstep: bool = True,
) -> np.ndarray:
    normalized = np.clip((values - low) / max(high - low, 1e-12), 0.0, 1.0)
    if smoothstep:
        normalized = normalized * normalized * (3.0 - 2.0 * normalized)
    if gamma > 0:
        normalized = np.power(normalized, gamma)

    if cmap_name in _CUSTOM_LUTS:
        lut = _CUSTOM_LUTS[cmap_name]
        indexes = np.minimum(
            (normalized * (len(lut) - 1)).astype(np.int32), len(lut) - 1
        )
        return lut[indexes]

    if HAVE_MPL:
        cmap = matplotlib.colormaps.get_cmap(cmap_name)
        return (cmap(normalized)[..., :3] * 255.0).astype(np.uint8)

    red = (0.6 + 0.4 * normalized) * 255
    green = (0.0 + 0.9 * normalized) * 255
    blue = (0.6 - 0.5 * normalized) * 255
    return np.dstack([red, green, blue]).astype(np.uint8)


def auto_iters(scale: float) -> int:
    depth = max(0.0, -math.log10(max(scale, MIN_SCALE)))
    return int(260 + 180 * depth + 70 * depth * depth)


class TkJulia:
    def __init__(self, width: int = 1200, height: int = 800) -> None:
        self.root = tk.Tk()
        self.root.title("Julia Set Viewer — Tk (Final)")

        toolbar = ttk.Frame(self.root, padding=(6, 4, 6, 4))
        toolbar.pack(side=tk.TOP, fill=tk.X)

        self.mode = tk.StringVar(value="zoom")
        ttk.Button(toolbar, text="Zoom", command=lambda: self.mode.set("zoom")).pack(side=tk.LEFT)
        ttk.Button(toolbar, text="Pan", command=lambda: self.mode.set("pan")).pack(side=tk.LEFT)

        ttk.Separator(toolbar, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=6)

        self.auto_iter = tk.BooleanVar(value=True)
        ttk.Checkbutton(
            toolbar,
            text="AutoIter",
            variable=self.auto_iter,
            command=self._request_progressive_render,
        ).pack(side=tk.LEFT)
        ttk.Button(toolbar, text="Iter +", command=lambda: self._bump_iter(1.25)).pack(
            side=tk.LEFT, padx=(6, 2)
        )
        ttk.Button(toolbar, text="Iter −", command=lambda: self._bump_iter(1 / 1.25)).pack(
            side=tk.LEFT, padx=(2, 6)
        )

        ttk.Separator(toolbar, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=6)

        ttk.Label(toolbar, text="c =").pack(side=tk.LEFT)
        self.c_real_var = tk.StringVar(value=f"{DEFAULT_C.real:.6f}")
        self.c_imag_var = tk.StringVar(value=f"{DEFAULT_C.imag:.6f}")
        ttk.Entry(toolbar, width=9, textvariable=self.c_real_var).pack(side=tk.LEFT, padx=(3, 1))
        ttk.Label(toolbar, text="+").pack(side=tk.LEFT)
        ttk.Entry(toolbar, width=9, textvariable=self.c_imag_var).pack(side=tk.LEFT, padx=1)
        ttk.Label(toolbar, text="i").pack(side=tk.LEFT, padx=(1, 3))
        ttk.Button(toolbar, text="Apply", command=self._apply_constant).pack(side=tk.LEFT)

        self.preset_var = tk.StringVar(value=DEFAULT_PRESET)
        ttk.OptionMenu(
            toolbar,
            self.preset_var,
            DEFAULT_PRESET,
            *JULIA_PRESETS.keys(),
            command=self._apply_preset,
        ).pack(side=tk.LEFT, padx=(5, 0))

        ttk.Separator(toolbar, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=6)

        ttk.Button(toolbar, text="Save PNG", command=self._save_dialog).pack(side=tk.LEFT)
        ttk.Button(toolbar, text="Save GIF", command=self._save_gif_dialog).pack(
            side=tk.LEFT, padx=(6, 0)
        )
        ttk.Button(toolbar, text="Reset", command=self._reset).pack(side=tk.LEFT, padx=(6, 0))

        ttk.Separator(toolbar, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=6)

        ttk.Label(toolbar, text="Colormap:").pack(side=tk.LEFT, padx=(0, 4))
        self.cmap_var = tk.StringVar(value=DEFAULT_CMAP)
        ttk.OptionMenu(
            toolbar,
            self.cmap_var,
            DEFAULT_CMAP,
            *CMAPS,
            command=lambda _=None: self._request_progressive_render(),
        ).pack(side=tk.LEFT)

        ttk.Separator(toolbar, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=6)
        ttk.Label(toolbar, text="Gamma:").pack(side=tk.LEFT, padx=(0, 4))
        self.gamma_var = tk.DoubleVar(value=1.35)
        ttk.Scale(
            toolbar,
            from_=0.6,
            to=2.2,
            variable=self.gamma_var,
            command=lambda _=None: self._request_progressive_render(),
            length=100,
        ).pack(side=tk.LEFT)

        # Live Julia-constant controls. Moving either slider updates c and
        # renders a low-resolution preview immediately, then a full image.
        parameter_bar = ttk.Frame(self.root, padding=(8, 2, 8, 4))
        parameter_bar.pack(side=tk.TOP, fill=tk.X)

        self.c_real_slider_var = tk.DoubleVar(value=DEFAULT_C.real)
        self.c_imag_slider_var = tk.DoubleVar(value=DEFAULT_C.imag)
        self.c_real_value = ttk.Label(parameter_bar, width=10, anchor="e")
        self.c_imag_value = ttk.Label(parameter_bar, width=10, anchor="e")

        ttk.Label(parameter_bar, text="Real c:").pack(side=tk.LEFT)
        ttk.Scale(
            parameter_bar,
            from_=-1.5,
            to=1.5,
            variable=self.c_real_slider_var,
            command=self._on_c_slider,
            length=280,
        ).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(4, 2))
        self.c_real_value.pack(side=tk.LEFT, padx=(0, 10))

        ttk.Label(parameter_bar, text="Imag c:").pack(side=tk.LEFT)
        ttk.Scale(
            parameter_bar,
            from_=-1.5,
            to=1.5,
            variable=self.c_imag_slider_var,
            command=self._on_c_slider,
            length=280,
        ).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(4, 2))
        self.c_imag_value.pack(side=tk.LEFT, padx=(0, 10))

        self.animate_c = tk.BooleanVar(value=False)
        ttk.Checkbutton(
            parameter_bar,
            text="Animate c",
            variable=self.animate_c,
            command=self._toggle_c_animation,
        ).pack(side=tk.LEFT)

        self.canvas = tk.Canvas(
            self.root,
            width=width,
            height=height,
            highlightthickness=0,
            bg="#ffffff",
            cursor="crosshair",
        )
        self.canvas.pack(fill=tk.BOTH, expand=True)

        self.status = ttk.Label(self.root, anchor="w", font=("Consolas", 10))
        self.status.pack(fill=tk.X)

        self.w, self.h = width, height
        self.view = View()
        self.julia_c = DEFAULT_C
        self._imgtk: Optional[ImageTk.PhotoImage] = None

        self._rb_start: Optional[Tuple[int, int]] = None
        self._rb_rect_id: Optional[int] = None
        self._pan_start_px: Optional[Tuple[int, int]] = None
        self._pan_start_center: Optional[Tuple[float, float]] = None
        self._resize_job: Optional[str] = None
        self._render_job: Optional[str] = None
        self._refine_job: Optional[str] = None
        self._drag_preview_job: Optional[str] = None
        self._render_generation = 0
        self._render_cancel = threading.Event()
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="julia-render")
        self._image_item: Optional[int] = None
        self._slider_job: Optional[str] = None
        self._animation_job: Optional[str] = None
        self._animation_phase = 0.0
        self._animation_center = DEFAULT_C
        self._animation_radius = 0.06
        self._gif_exporting = False
        self._gif_progress_window: Optional[tk.Toplevel] = None
        self._gif_progress_var = tk.DoubleVar(value=0.0)
        self._gif_progress_label_var = tk.StringVar(value="Preparing GIF…")
        self._update_c_labels()

        self.canvas.bind("<Configure>", self._on_resize)
        self.canvas.bind("<MouseWheel>", self._on_wheel)
        self.canvas.bind("<Button-4>", lambda event: self._zoom_at(0.85, event.x, event.y))
        self.canvas.bind("<Button-5>", lambda event: self._zoom_at(1 / 0.85, event.x, event.y))
        self.canvas.bind("<ButtonPress-1>", self._on_press_left)
        self.canvas.bind("<B1-Motion>", self._on_drag_left)
        self.canvas.bind("<ButtonRelease-1>", self._on_release_left)
        self.canvas.bind("<Button-3>", lambda _event: self._reset())

        self.root.bind("<KeyPress-z>", lambda _event: self._zoom_at(0.85, self.w / 2, self.h / 2))
        self.root.bind("<KeyPress-x>", lambda _event: self._zoom_at(1 / 0.85, self.w / 2, self.h / 2))
        self.root.bind("<KeyPress-plus>", lambda _event: self._bump_iter(1.25))
        self.root.bind("<KeyPress-equal>", lambda _event: self._bump_iter(1.25))
        self.root.bind("<KeyPress-minus>", lambda _event: self._bump_iter(1 / 1.25))
        self.root.bind("<KeyPress-c>", lambda _event: self._cycle_cmap())
        self.root.bind("<KeyPress-i>", lambda _event: self._toggle_autoiter())
        self.root.bind("<KeyPress-r>", lambda _event: self._reset())
        self.root.bind("<KeyPress-s>", lambda _event: self._save_dialog())
        self.root.bind("<KeyPress-g>", lambda _event: self._save_gif_dialog())
        self.root.bind("<Return>", lambda _event: self._apply_constant())
        self.root.bind("<Escape>", lambda _event: self._on_close())
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

        # Wait until Tk has completed geometry/layout before the first render.
        # This prevents the initial image from being calculated at a tiny 1x1-like size.
        self.root.after_idle(self._initial_render)

    def _status_text(self) -> str:
        return (
            f"c=({self.julia_c.real:+.6f}{self.julia_c.imag:+.6f}i)   "
            f"Center=({self.view.cx:.6f}, {self.view.cy:.6f})   "
            f"Scale={self.view.scale:.6e}   Iter={self.view.max_iter}   "
            f"AutoIter={'ON' if self.auto_iter.get() else 'OFF'}   "
            f"Size={self.w}x{self.h}   Numba={'ON' if HAS_NUMBA else 'OFF'}"
        )

    def _set_status(self) -> None:
        self.status.config(text=self._status_text())

    def _initial_render(self) -> None:
        """Render only after Tk has established the real canvas dimensions."""
        self.root.update_idletasks()
        self.w = max(100, self.canvas.winfo_width())
        self.h = max(100, self.canvas.winfo_height())
        self._request_progressive_render(refine_delay=220)

    def _on_resize(self, event: tk.Event) -> None:
        """Debounce canvas resizing and render once the window settles."""
        if event.widget is not self.canvas:
            return

        new_w = max(100, int(event.width))
        new_h = max(100, int(event.height))
        if new_w == self.w and new_h == self.h:
            return

        self.w, self.h = new_w, new_h
        if self._resize_job is not None:
            try:
                self.root.after_cancel(self._resize_job)
            except tk.TclError:
                pass

        # Maximizing/resizing emits many Configure events. Wait for 300 ms of
        # silence, then request one preview followed by one full render.
        self._resize_job = self.root.after(300, self._finish_resize)

    def _finish_resize(self) -> None:
        self._resize_job = None
        self._request_progressive_render(refine_delay=120)

    def _cycle_cmap(self) -> None:
        index = CMAPS.index(self.cmap_var.get())
        self.cmap_var.set(CMAPS[(index + 1) % len(CMAPS)])
        self._request_progressive_render()

    def _toggle_autoiter(self) -> None:
        self.auto_iter.set(not self.auto_iter.get())
        self._request_progressive_render()

    def _update_c_labels(self) -> None:
        real = float(self.c_real_slider_var.get())
        imag = float(self.c_imag_slider_var.get())
        self.c_real_value.config(text=f"{real:+.6f}")
        self.c_imag_value.config(text=f"{imag:+.6f}")

    def _sync_c_controls(self) -> None:
        self.c_real_slider_var.set(self.julia_c.real)
        self.c_imag_slider_var.set(self.julia_c.imag)
        self.c_real_var.set(f"{self.julia_c.real:.6f}")
        self.c_imag_var.set(f"{self.julia_c.imag:.6f}")
        self._update_c_labels()

    def _on_c_slider(self, _value: str = "") -> None:
        """Update c live while keeping the UI responsive."""
        real = float(self.c_real_slider_var.get())
        imag = float(self.c_imag_slider_var.get())
        self.julia_c = complex(real, imag)
        self.c_real_var.set(f"{real:.6f}")
        self.c_imag_var.set(f"{imag:.6f}")
        self.preset_var.set("Custom")
        self._update_c_labels()

        if self._slider_job is not None:
            try:
                self.root.after_cancel(self._slider_job)
            except tk.TclError:
                pass
        self._slider_job = self.root.after(35, self._slider_render)

    def _slider_render(self) -> None:
        self._slider_job = None
        self._request_progressive_render(refine_delay=420)

    def _toggle_c_animation(self) -> None:
        """Move c around a small circle to morph the Julia set continuously."""
        if self.animate_c.get():
            self._animation_center = self.julia_c
            self._animation_phase = 0.0
            self._animate_c_step()
        elif self._animation_job is not None:
            try:
                self.root.after_cancel(self._animation_job)
            except tk.TclError:
                pass
            self._animation_job = None
            self._request_progressive_render(refine_delay=80)

    def _animate_c_step(self) -> None:
        if not self.animate_c.get():
            return
        self._animation_phase += 0.10
        real = self._animation_center.real + self._animation_radius * math.cos(self._animation_phase)
        imag = self._animation_center.imag + self._animation_radius * math.sin(self._animation_phase)
        self.julia_c = complex(real, imag)
        self.c_real_slider_var.set(real)
        self.c_imag_slider_var.set(imag)
        self.c_real_var.set(f"{real:.6f}")
        self.c_imag_var.set(f"{imag:.6f}")
        self.preset_var.set("Custom")
        self._update_c_labels()
        self._render(quality="preview")
        self._animation_job = self.root.after(90, self._animate_c_step)

    def _apply_constant(self) -> None:
        try:
            real = float(self.c_real_var.get().strip())
            imag = float(self.c_imag_var.get().strip())
        except ValueError:
            messagebox.showerror("Invalid Julia constant", "Enter numeric values for c.real and c.imag.")
            return

        self.julia_c = complex(real, imag)
        self.preset_var.set("Custom")
        self._sync_c_controls()
        self.view = View()
        self._request_progressive_render()

    def _apply_preset(self, preset_name: str) -> None:
        if preset_name not in JULIA_PRESETS:
            return
        self.julia_c = JULIA_PRESETS[preset_name]
        self._sync_c_controls()
        self.view = View()
        self._request_progressive_render()

    def _on_press_left(self, event: tk.Event) -> None:
        if self.mode.get() == "pan":
            self._pan_start_px = (event.x, event.y)
            self._pan_start_center = (self.view.cx, self.view.cy)
        else:
            self._rb_start = (event.x, event.y)
            if self._rb_rect_id:
                self.canvas.delete(self._rb_rect_id)
                self._rb_rect_id = None

    def _on_drag_left(self, event: tk.Event) -> None:
        if self.mode.get() == "pan" and self._pan_start_px and self._pan_start_center:
            start_x, start_y = self._pan_start_px
            delta_x, delta_y = event.x - start_x, event.y - start_y
            aspect = self.h / self.w
            span_x = 2 * self.view.scale / aspect
            span_y = 2 * self.view.scale
            center_x, center_y = self._pan_start_center
            self.view.cx = center_x - delta_x / self.w * span_x
            self.view.cy = center_y + delta_y / self.h * span_y
            if self._image_item is not None:
                self.canvas.coords(self._image_item, delta_x, delta_y)
            self._schedule_drag_preview()
        elif self.mode.get() == "zoom" and self._rb_start:
            x0, y0 = self._rb_start
            if self._rb_rect_id:
                self.canvas.coords(self._rb_rect_id, x0, y0, event.x, event.y)
            else:
                self._rb_rect_id = self.canvas.create_rectangle(
                    x0,
                    y0,
                    event.x,
                    event.y,
                    outline="#00e0ff",
                    width=2,
                    dash=(4, 2),
                )

    def _on_release_left(self, event: tk.Event) -> None:
        if self.mode.get() == "pan":
            self._pan_start_px = None
            self._pan_start_center = None
            if self._drag_preview_job is not None:
                try:
                    self.root.after_cancel(self._drag_preview_job)
                except tk.TclError:
                    pass
                self._drag_preview_job = None
            self._request_progressive_render(refine_delay=70)
            return
        if not self._rb_start:
            return

        x0, y0 = self._rb_start
        x1, y1 = event.x, event.y
        self._rb_start = None

        if self._rb_rect_id:
            self.canvas.delete(self._rb_rect_id)
            self._rb_rect_id = None

        if abs(x1 - x0) < 6 or abs(y1 - y0) < 6:
            self._zoom_at(0.7, x1, y1)
            return

        left, right = sorted((x0, x1))
        screen_top, screen_bottom = sorted((y0, y1))
        aspect = self.h / self.w
        x_min = self.view.cx - self.view.scale / aspect
        x_max = self.view.cx + self.view.scale / aspect
        y_min = self.view.cy - self.view.scale
        y_max = self.view.cy + self.view.scale

        rect_x_min = x_min + left / self.w * (x_max - x_min)
        rect_x_max = x_min + right / self.w * (x_max - x_min)
        rect_y_min = y_min + (self.h - screen_bottom) / self.h * (y_max - y_min)
        rect_y_max = y_min + (self.h - screen_top) / self.h * (y_max - y_min)

        rect_width = rect_x_max - rect_x_min
        rect_height = rect_y_max - rect_y_min
        target_aspect = self.h / self.w

        self.view.cx = (rect_x_min + rect_x_max) / 2.0
        self.view.cy = (rect_y_min + rect_y_max) / 2.0
        if rect_height / rect_width > target_aspect:
            rect_width = rect_height / target_aspect
        else:
            rect_height = rect_width * target_aspect
        self.view.scale = max(MIN_SCALE, rect_height / 2.0)
        self._request_progressive_render(refine_delay=90)

    def _on_wheel(self, event: tk.Event) -> None:
        steps = int(event.delta / 120) if event.delta else 0
        if steps > 0:
            factor = 0.85**steps
        elif steps < 0:
            factor = (1 / 0.85) ** (-steps)
        else:
            return
        self._zoom_at(factor, event.x, event.y)

    def _zoom_at(self, factor: float, px: float, py: float) -> None:
        aspect = self.h / self.w
        x_min = self.view.cx - self.view.scale / aspect
        x_max = self.view.cx + self.view.scale / aspect
        y_min = self.view.cy - self.view.scale
        y_max = self.view.cy + self.view.scale
        target_x = x_min + px / self.w * (x_max - x_min)
        target_y = y_min + (self.h - py) / self.h * (y_max - y_min)
        self.view.cx = target_x + (self.view.cx - target_x) * factor
        self.view.cy = target_y + (self.view.cy - target_y) * factor
        self.view.scale = max(MIN_SCALE, self.view.scale * factor)
        self._request_progressive_render(refine_delay=140)

    def _bump_iter(self, multiplier: float) -> None:
        self.view.max_iter = max(20, int(self.view.max_iter * multiplier + 1))
        self._request_progressive_render()

    def _reset(self) -> None:
        self.view = View()
        self._request_progressive_render()

    def _schedule_drag_preview(self) -> None:
        """Render a tiny preview while panning without flooding the worker."""
        if self._drag_preview_job is not None:
            try:
                self.root.after_cancel(self._drag_preview_job)
            except tk.TclError:
                pass
        self._drag_preview_job = self.root.after(70, lambda: self._render(quality="preview"))

    def _request_progressive_render(self, refine_delay: int = 180) -> None:
        """Show a 320x200-class preview first, then refine at full canvas resolution."""
        self._render(quality="preview")
        if self._refine_job is not None:
            try:
                self.root.after_cancel(self._refine_job)
            except tk.TclError:
                pass
        self._refine_job = self.root.after(refine_delay, lambda: self._render(quality="full"))

    def _render(self, quality: str = "full") -> None:
        """Schedule a non-blocking render. New requests cancel stale NumPy jobs."""
        self._resize_job = None
        if self.w < 100 or self.h < 100:
            return

        if self._render_job is not None:
            try:
                self.root.after_cancel(self._render_job)
            except tk.TclError:
                pass
        delay = 15 if quality == "preview" else 35
        self._render_job = self.root.after(delay, lambda: self._start_render(quality))

    def _start_render(self, quality: str = "full") -> None:
        self._render_job = None
        self._render_generation += 1
        generation = self._render_generation

        self._render_cancel.set()
        self._render_cancel = threading.Event()
        cancel_event = self._render_cancel

        if self.auto_iter.get():
            self.view.max_iter = max(self.view.max_iter, auto_iters(self.view.scale))

        # During interaction use a roughly 320x200 preview. Once interaction
        # pauses, refine at the full canvas resolution. PNG export remains 2x.
        if quality == "preview":
            preview_w = min(320, self.w)
            preview_h = max(120, int(preview_w * self.h / max(1, self.w)))
            if preview_h > 240:
                preview_h = 240
                preview_w = max(160, int(preview_h * self.w / max(1, self.h)))
            render_w, render_h = preview_w, preview_h
        else:
            render_w, render_h = self.w, self.h

        snapshot = View(self.view.cx, self.view.cy, self.view.scale, self.view.max_iter)
        julia_c = self.julia_c
        cmap_name = self.cmap_var.get()
        gamma = float(self.gamma_var.get())
        label = "Preview…" if quality == "preview" else "Refining…"
        self.status.config(text=self._status_text() + f"   {label}")

        future = self._executor.submit(
            self._render_worker,
            generation,
            cancel_event,
            snapshot,
            julia_c,
            cmap_name,
            gamma,
            render_w,
            render_h,
            self.w,
            self.h,
        )
        future.add_done_callback(lambda done: self.root.after(0, self._finish_render, done))

    @staticmethod
    def _render_worker(
        generation: int,
        cancel_event: threading.Event,
        view: View,
        julia_c: complex,
        cmap_name: str,
        gamma: float,
        render_w: int,
        render_h: int,
        canvas_w: int,
        canvas_h: int,
    ):
        grid = view.grid(render_w, render_h)
        values = julia_smooth(grid, julia_c, view.max_iter, cancel_event)
        if values is None or cancel_event.is_set():
            return None
        low, high = escaped_contrast(values, view.max_iter)
        rgb = map_to_rgb(values, cmap_name, low, high, gamma=gamma, smoothstep=True)
        image = Image.fromarray(rgb, mode="RGB")
        if (render_w, render_h) != (canvas_w, canvas_h):
            image = image.resize((canvas_w, canvas_h), Image.Resampling.BILINEAR)
        return generation, image

    def _finish_render(self, future) -> None:
        try:
            result = future.result()
        except Exception as exc:
            self.status.config(text=f"Render failed: {exc}")
            return
        if result is None:
            return
        generation, image = result
        if generation != self._render_generation:
            return

        self._imgtk = ImageTk.PhotoImage(image)
        self.canvas.delete("fractal")
        self._image_item = self.canvas.create_image(
            0, 0, image=self._imgtk, anchor="nw", tags=("fractal",)
        )
        self.canvas.tag_lower(self._image_item)
        self._set_status()

    def _on_close(self) -> None:
        self._render_cancel.set()
        for job in (
            self._render_job,
            self._refine_job,
            self._drag_preview_job,
            self._resize_job,
            self._slider_job,
            self._animation_job,
        ):
            if job is not None:
                try:
                    self.root.after_cancel(job)
                except tk.TclError:
                    pass
        self._executor.shutdown(wait=False, cancel_futures=True)
        self.root.destroy()

    def _save_dialog(self) -> None:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        real = f"{self.julia_c.real:+.4f}".replace("+", "p").replace("-", "m")
        imag = f"{self.julia_c.imag:+.4f}".replace("+", "p").replace("-", "m")
        default_name = f"julia_c_{real}_{imag}_{timestamp}.png"
        path = filedialog.asksaveasfilename(
            title="Save PNG",
            defaultextension=".png",
            filetypes=[("PNG image", "*.png")],
            initialfile=default_name,
        )
        if not path:
            return
        try:
            self._save_png(path)
        except Exception as exc:
            messagebox.showerror("Save failed", str(exc))

    def _save_png(self, path: str) -> None:
        width, height = self.w * 2, self.h * 2
        grid = self.view.grid(width, height)
        save_iter = max(self.view.max_iter, int(self.view.max_iter * 1.3))
        values = julia_smooth(grid, self.julia_c, save_iter)
        low, high = escaped_contrast(values, save_iter)
        rgb = map_to_rgb(
            values,
            self.cmap_var.get(),
            low,
            high,
            gamma=self.gamma_var.get(),
            smoothstep=True,
        )
        Image.fromarray(rgb, mode="RGB").save(path, optimize=True)
        print(f"Saved: {os.path.abspath(path)}")

    def _save_gif_dialog(self) -> None:
        """Export a five-second animation of c moving around its current centre."""
        if self._gif_exporting:
            messagebox.showinfo("GIF export", "A GIF export is already running.")
            return

        timestamp = time.strftime("%Y%m%d_%H%M%S")
        real = f"{self.julia_c.real:+.4f}".replace("+", "p").replace("-", "m")
        imag = f"{self.julia_c.imag:+.4f}".replace("+", "p").replace("-", "m")
        default_name = f"julia_animation_c_{real}_{imag}_{timestamp}.gif"
        path = filedialog.asksaveasfilename(
            title="Save 5-second GIF",
            defaultextension=".gif",
            filetypes=[("Animated GIF", "*.gif")],
            initialfile=default_name,
        )
        if not path:
            return

        # Freeze the animation's centre at the current c. The live animation
        # may continue visually, but the exported orbit is deterministic.
        export_center = self.julia_c
        snapshot = View(self.view.cx, self.view.cy, self.view.scale, self.view.max_iter)
        cmap_name = self.cmap_var.get()
        gamma = float(self.gamma_var.get())

        self._gif_exporting = True
        self._show_gif_progress()
        thread = threading.Thread(
            target=self._save_gif_worker,
            args=(path, export_center, snapshot, cmap_name, gamma),
            daemon=True,
            name="julia-gif-export",
        )
        thread.start()

    def _show_gif_progress(self) -> None:
        window = tk.Toplevel(self.root)
        window.title("Rendering GIF")
        window.resizable(False, False)
        window.transient(self.root)
        window.protocol("WM_DELETE_WINDOW", lambda: None)

        frame = ttk.Frame(window, padding=14)
        frame.pack(fill=tk.BOTH, expand=True)
        ttk.Label(frame, textvariable=self._gif_progress_label_var).pack(anchor="w")
        ttk.Progressbar(
            frame,
            maximum=100.0,
            variable=self._gif_progress_var,
            length=360,
            mode="determinate",
        ).pack(fill=tk.X, pady=(8, 2))
        ttk.Label(frame, text="5 seconds · 15 FPS · 800×450 maximum").pack(anchor="w")

        self._gif_progress_var.set(0.0)
        self._gif_progress_label_var.set("Preparing GIF…")
        self._gif_progress_window = window
        window.update_idletasks()
        x = self.root.winfo_rootx() + max(0, (self.root.winfo_width() - window.winfo_width()) // 2)
        y = self.root.winfo_rooty() + max(0, (self.root.winfo_height() - window.winfo_height()) // 2)
        window.geometry(f"+{x}+{y}")
        window.grab_set()

    def _update_gif_progress(self, completed: int, total: int) -> None:
        if not self._gif_exporting:
            return
        percent = completed / max(1, total) * 100.0
        self._gif_progress_var.set(percent)
        self._gif_progress_label_var.set(f"Rendering frame {completed} / {total}")

    def _save_gif_worker(
        self,
        path: str,
        center: complex,
        view: View,
        cmap_name: str,
        gamma: float,
    ) -> None:
        """Render GIF frames away from Tk's UI thread."""
        fps = 15
        duration_seconds = 5
        frame_count = fps * duration_seconds
        frame_duration_ms = round(1000 / fps)

        # Keep GIFs practical in size while preserving the current aspect ratio.
        max_width, max_height = 800, 450
        ratio = min(max_width / max(1, self.w), max_height / max(1, self.h), 1.0)
        width = max(320, int(self.w * ratio))
        height = max(180, int(self.h * ratio))

        frames: list[Image.Image] = []
        gif_iter = min(max(180, view.max_iter), 700)
        radius = self._animation_radius

        try:
            for index in range(frame_count):
                angle = 2.0 * math.pi * index / frame_count
                frame_c = center + radius * complex(math.cos(angle), math.sin(angle))
                grid = view.grid(width, height)
                values = julia_smooth(grid, frame_c, gif_iter)
                if values is None:
                    raise RuntimeError("GIF rendering was interrupted.")
                low, high = escaped_contrast(values, gif_iter)
                rgb = map_to_rgb(
                    values,
                    cmap_name,
                    low,
                    high,
                    gamma=gamma,
                    smoothstep=True,
                )
                # Adaptive palette conversion keeps the animation compact and
                # avoids relying on a single frame's palette for all frames.
                frame = Image.fromarray(rgb, mode="RGB").convert(
                    "P", palette=Image.Palette.ADAPTIVE, colors=256
                )
                frames.append(frame)
                self.root.after(0, self._update_gif_progress, index + 1, frame_count)

            if not frames:
                raise RuntimeError("No GIF frames were rendered.")
            frames[0].save(
                path,
                save_all=True,
                append_images=frames[1:],
                duration=frame_duration_ms,
                loop=0,
                optimize=False,
                disposal=2,
            )
            size_mb = os.path.getsize(path) / (1024 * 1024)
            self.root.after(
                0,
                self._finish_gif_export,
                None,
                path,
                frame_count,
                fps,
                size_mb,
            )
        except Exception as exc:
            self.root.after(0, self._finish_gif_export, exc, path, 0, fps, 0.0)

    def _finish_gif_export(
        self,
        error: Optional[Exception],
        path: str,
        frame_count: int,
        fps: int,
        size_mb: float,
    ) -> None:
        self._gif_exporting = False
        if self._gif_progress_window is not None:
            try:
                self._gif_progress_window.grab_release()
                self._gif_progress_window.destroy()
            except tk.TclError:
                pass
            self._gif_progress_window = None

        if error is not None:
            messagebox.showerror("GIF export failed", str(error))
            return

        messagebox.showinfo(
            "GIF exported",
            f"Saved successfully.\n\n"
            f"{frame_count} frames · {fps} FPS · 5 seconds\n"
            f"File size: {size_mb:.1f} MB\n\n{os.path.abspath(path)}",
        )
        print(f"Saved GIF: {os.path.abspath(path)}")

    def run(self) -> None:
        self.root.mainloop()


if __name__ == "__main__":
    TkJulia().run()
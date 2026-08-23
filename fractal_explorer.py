"""
Real-time fractal explorer: Julia / Mandelbrot sets with user-defined
constants (c, exponent, iteration count) and live color mapping.

Run: python fractal_explorer.py
"""
import tkinter as tk
from tkinter import ttk, filedialog
import numpy as np
from PIL import Image, ImageTk

try:
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # silence cupy's harmless CUDA_PATH notice
        import cupy as cp
    cp.cuda.runtime.getDeviceCount()  # raises if no usable GPU/driver
    GPU_AVAILABLE = True
except Exception:
    cp = None
    GPU_AVAILABLE = False

_RAW_KERNELS = {}


def _get_raw_kernel(name, source):
    """Compile (once) and cache a CuPy RawKernel. Returns None if compilation
    fails for any reason, so callers can fall back to plain CuPy ops."""
    if name in _RAW_KERNELS:
        return _RAW_KERNELS[name]
    try:
        kernel = cp.RawKernel(source, name)
        _RAW_KERNELS[name] = kernel
    except Exception:
        _RAW_KERNELS[name] = None
    return _RAW_KERNELS[name]


_ESCAPE_KERNEL_SRC = r"""
extern "C" __global__
void escape_kernel(const double* re_line, const double* im_line,
                    int width, int height,
                    double c_re, double c_im, int p, int max_iter,
                    int mode, int formula, double bailout_sq,
                    double* smooth_iter, bool* escaped)
{
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int total = width * height;
    if (idx >= total) return;
    int x = idx % width;
    int y = idx / width;

    double zr, zi, cr, ci;
    if (mode == 0) {           // julia: c fixed, z sweeps the view
        zr = re_line[x];
        zi = im_line[y];
        cr = c_re;
        ci = c_im;
    } else {                   // mandelbrot: z starts at 0, c sweeps the view
        zr = 0.0;
        zi = 0.0;
        cr = re_line[x];
        ci = im_line[y];
    }

    bool esc = false;
    double result_iter = (double)max_iter;
    double log_p = log((double)p);

    for (int n = 0; n < max_iter; n++) {
        double tr = zr;
        double ti = zi;
        if (formula == 1) {         // burning ship
            tr = fabs(zr);
            ti = fabs(zi);
        } else if (formula == 2) {  // tricorn
            ti = -zi;
        }
        double pr = 1.0, pi = 0.0;
        for (int k = 0; k < p; k++) {
            double nr = pr * tr - pi * ti;
            double ni = pr * ti + pi * tr;
            pr = nr;
            pi = ni;
        }
        double zr_next = pr + cr;
        double zi_next = pi + ci;
        double mag_sq = zr_next * zr_next + zi_next * zi_next;
        if (mag_sq > bailout_sq) {
            double mod = sqrt(mag_sq);
            result_iter = (double)(n + 1) - log(log(mod)) / log_p;
            esc = true;
            break;
        }
        zr = zr_next;
        zi = zi_next;
    }
    smooth_iter[idx] = result_iter;
    escaped[idx] = esc;
}
"""

_FORMULA_CODE = {"standard": 0, "burning_ship": 1, "tricorn": 2}


def _compute_escape_gpu_kernel(width, height, center, half_width, c_val, p, max_iter,
                                mode, formula):
    """Single fused CUDA kernel: the whole per-pixel escape-time loop runs on
    the GPU in one launch, with zero Python-level calls per iteration (the
    plain-CuPy path below still issues ~10 small kernel launches per
    iteration, which dominates runtime at interactive resolutions)."""
    kernel = _get_raw_kernel("escape_kernel", _ESCAPE_KERNEL_SRC)
    if kernel is None:
        raise RuntimeError("raw escape kernel unavailable")

    half_height = half_width * height / width
    re_line = cp.linspace(center.real - half_width, center.real + half_width,
                           width, dtype=cp.float64)
    im_line = cp.linspace(center.imag - half_height, center.imag + half_height,
                           height, dtype=cp.float64)

    smooth_iter = cp.empty((height, width), dtype=cp.float64)
    escaped = cp.empty((height, width), dtype=cp.bool_)

    total = width * height
    threads = 256
    blocks = (total + threads - 1) // threads

    kernel((blocks,), (threads,), (
        re_line, im_line, np.int32(width), np.int32(height),
        np.float64(c_val.real), np.float64(c_val.imag), np.int32(p), np.int32(max_iter),
        np.int32(0 if mode == "julia" else 1), np.int32(_FORMULA_CODE[formula]),
        np.float64(1e12), smooth_iter, escaped,
    ))
    return cp.asnumpy(smooth_iter), cp.asnumpy(escaped)


_NEWTON_KERNEL_SRC = r"""
extern "C" __global__
void newton_kernel(const double* re_line, const double* im_line,
                    int width, int height,
                    double c_re, double c_im, int p, int max_iter,
                    const double* root_re, const double* root_im,
                    double tol_sq,
                    double* iter_count, bool* converged, int* root_index)
{
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int total = width * height;
    if (idx >= total) return;
    int x = idx % width;
    int y = idx / width;

    double zr = re_line[x];
    double zi = im_line[y];

    bool conv = false;
    double iters = (double)max_iter;

    for (int n = 0; n < max_iter; n++) {
        // z^(p-1) via repeated multiplication (p is small: 2..6)
        double zr_pm1 = 1.0, zi_pm1 = 0.0;
        for (int k = 0; k < p - 1; k++) {
            double nr = zr_pm1 * zr - zi_pm1 * zi;
            double ni = zr_pm1 * zi + zi_pm1 * zr;
            zr_pm1 = nr;
            zi_pm1 = ni;
        }
        double zpr = zr_pm1 * zr - zi_pm1 * zi;  // z^p
        double zpi = zr_pm1 * zi + zi_pm1 * zr;

        double fr = zpr - c_re;                  // f(z) = z^p - c
        double fi = zpi - c_im;
        double fpr = p * zr_pm1;                 // f'(z) = p * z^(p-1)
        double fpi = p * zi_pm1;

        double denom = fpr * fpr + fpi * fpi;
        if (denom < 1e-24) denom = 1e-24;
        double step_r = (fr * fpr + fi * fpi) / denom;
        double step_i = (fi * fpr - fr * fpi) / denom;

        double zr_next = zr - step_r;
        double zi_next = zi - step_i;

        if (step_r * step_r + step_i * step_i < tol_sq) {
            conv = true;
            iters = (double)n;
            zr = zr_next;
            zi = zi_next;
            break;
        }
        zr = zr_next;
        zi = zi_next;
    }

    int best_k = 0;
    double best_dist = 1e300;
    for (int k = 0; k < p; k++) {
        double dr = zr - root_re[k];
        double di = zi - root_im[k];
        double d = dr * dr + di * di;
        if (d < best_dist) {
            best_dist = d;
            best_k = k;
        }
    }

    iter_count[idx] = iters;
    converged[idx] = conv;
    root_index[idx] = best_k;
}
"""


def _compute_newton_gpu_kernel(width, height, center, half_width, c_val, p, max_iter):
    """Fused single-launch CUDA kernel version of compute_newton (see
    _compute_escape_gpu_kernel for why this matters)."""
    kernel = _get_raw_kernel("newton_kernel", _NEWTON_KERNEL_SRC)
    if kernel is None:
        raise RuntimeError("raw newton kernel unavailable")

    half_height = half_width * height / width
    re_line = cp.linspace(center.real - half_width, center.real + half_width,
                           width, dtype=cp.float64)
    im_line = cp.linspace(center.imag - half_height, center.imag + half_height,
                           height, dtype=cp.float64)

    roots = [c_val ** (1.0 / p) * np.exp(2j * np.pi * k / p) for k in range(p)]
    root_re = cp.asarray([r.real for r in roots], dtype=cp.float64)
    root_im = cp.asarray([r.imag for r in roots], dtype=cp.float64)

    iter_count = cp.empty((height, width), dtype=cp.float64)
    converged = cp.empty((height, width), dtype=cp.bool_)
    root_index = cp.empty((height, width), dtype=cp.int32)

    total = width * height
    threads = 256
    blocks = (total + threads - 1) // threads

    kernel((blocks,), (threads,), (
        re_line, im_line, np.int32(width), np.int32(height),
        np.float64(c_val.real), np.float64(c_val.imag), np.int32(p), np.int32(max_iter),
        root_re, root_im, np.float64(NEWTON_TOL ** 2),
        iter_count, converged, root_index,
    ))
    return cp.asnumpy(root_index), cp.asnumpy(iter_count), cp.asnumpy(converged)


WIDTH, HEIGHT = 480, 480
DEBOUNCE_MS = 30
RESIZE_DEBOUNCE_MS = 150
EXPORT_MAX_SIDE = 1920
COLOR_CYCLE = 24.0  # escape-count band width; keeps color visible far from max_iter

DEFAULT_MODE = "julia"
DEFAULT_FORMULA = "standard"
DEFAULT_C_RE = -0.70
DEFAULT_C_IM = 0.27
DEFAULT_EXPONENT = 2
DEFAULT_MAX_ITER = 200
DEFAULT_ZOOM = 1.0
DEFAULT_CENTER = complex(0.0, 0.0)

# ---------------------------------------------------------------------------
# Colormaps
# ---------------------------------------------------------------------------

def _custom_colormaps():
    """Hand-written fallback colormaps, each t in [0,1] -> (r,g,b) in [0,255]."""
    def grayscale(t):
        v = (t * 255).astype(np.uint8)
        return np.stack([v, v, v], axis=-1)

    def fire(t):
        r = np.clip(t * 3.0, 0, 1)
        g = np.clip(t * 3.0 - 1.0, 0, 1)
        b = np.clip(t * 3.0 - 2.0, 0, 1)
        return (np.stack([r, g, b], axis=-1) * 255).astype(np.uint8)

    def ocean(t):
        r = np.clip(t * 2.0 - 1.0, 0, 1)
        g = np.clip(t * 1.5, 0, 1)
        b = np.clip(0.4 + t * 0.6, 0, 1)
        return (np.stack([r, g, b], axis=-1) * 255).astype(np.uint8)

    def rainbow_hsv(t):
        h = t
        s = np.ones_like(t)
        v = np.ones_like(t)
        i = (h * 6.0).astype(int) % 6
        f = (h * 6.0) - np.floor(h * 6.0)
        p = v * (1 - s)
        q = v * (1 - f * s)
        tt = v * (1 - (1 - f) * s)
        r = np.select(
            [i == 0, i == 1, i == 2, i == 3, i == 4, i == 5],
            [v, q, p, p, tt, v],
        )
        g = np.select(
            [i == 0, i == 1, i == 2, i == 3, i == 4, i == 5],
            [tt, v, v, q, p, p],
        )
        b = np.select(
            [i == 0, i == 1, i == 2, i == 3, i == 4, i == 5],
            [p, p, tt, v, v, q],
        )
        return (np.stack([r, g, b], axis=-1) * 255).astype(np.uint8)

    return {
        "grayscale": grayscale,
        "fire": fire,
        "ocean": ocean,
        "rainbow": rainbow_hsv,
    }


def _matplotlib_colormaps():
    try:
        import matplotlib
    except ImportError:
        return {}
    names = ["inferno", "twilight", "viridis", "turbo", "hsv"]
    out = {}
    for name in names:
        try:
            mpl_cmap = matplotlib.colormaps[name]
        except (KeyError, AttributeError):
            continue

        def make(mpl_cmap=mpl_cmap):
            def fn(t):
                rgba = mpl_cmap(t)
                return (rgba[..., :3] * 255).astype(np.uint8)
            return fn

        out[name] = make()
    return out


COLORMAPS = {}
COLORMAPS.update(_matplotlib_colormaps())
COLORMAPS.update(_custom_colormaps())  # always keep fallbacks available too


# ---------------------------------------------------------------------------
# Fractal math
# ---------------------------------------------------------------------------

def compute_escape(width, height, center, half_width, c_val, p, max_iter, mode,
                    use_gpu=False, formula="standard"):
    """Vectorized escape-time computation.

    `mode` ("julia"/"mandelbrot") selects whether c is fixed and z sweeps the
    view (Julia) or z starts at 0 and c sweeps the view (Mandelbrot).
    `formula` selects the per-iteration transform applied before z**p + c:
      - "standard":     z_{n+1} = z_n^p + c
      - "burning_ship": z_{n+1} = (|Re z_n| + i|Im z_n|)^p + c
      - "tricorn":      z_{n+1} = conj(z_n)^p + c

    Returns (smooth_iter, escaped): smooth_iter is a (height, width) float
    array of continuous escape counts (unbounded), escaped is a bool mask of
    which points left the bailout radius before max_iter. Runs on the GPU via
    CuPy when use_gpu=True and CuPy is available; always returns host (NumPy)
    arrays regardless of backend.
    """
    xp = cp if (use_gpu and GPU_AVAILABLE) else np
    log_p = float(np.log(p))

    if xp is np:
        # CPU: shrink the working set every iteration via boolean fancy-
        # indexing. Most pixels escape within the first few iterations, so
        # the active array collapses fast and later iterations (which only
        # matter for a thin band near the fractal boundary) stay cheap.
        # There's no host<->device sync cost on CPU, so checking every
        # iteration is free.
        half_height = half_width * height / width
        re = np.linspace(center.real - half_width, center.real + half_width, width)
        im = np.linspace(center.imag - half_height, center.imag + half_height, height)
        re_grid, im_grid = np.meshgrid(re, im)
        grid = re_grid + 1j * im_grid
        if mode == "julia":
            z = grid
            c = np.full_like(z, c_val)
        else:
            z = np.zeros_like(grid)
            c = grid

        escaped = np.zeros((height, width), dtype=bool)
        smooth_iter = np.zeros((height, width), dtype=np.float64)
        with np.errstate(over="ignore", invalid="ignore"):
            for n in range(max_iter):
                mask = ~escaped
                zm = z[mask]
                if formula == "burning_ship":
                    zm = np.abs(zm.real) + 1j * np.abs(zm.imag)
                elif formula == "tricorn":
                    zm = np.conj(zm)
                z[mask] = zm ** p + c[mask]
                newly_escaped = (~escaped) & (np.abs(z) > 1e6)
                if newly_escaped.any():
                    mod = np.abs(z[newly_escaped])
                    smooth_iter[newly_escaped] = n + 1 - np.log(np.log(mod)) / log_p
                    escaped |= newly_escaped
                if escaped.all():
                    break
        return smooth_iter, escaped

    # GPU, first choice: single fused kernel launch (see
    # _compute_escape_gpu_kernel), falls through to the dense CuPy loop
    # below if kernel compilation/execution fails for any reason.
    try:
        return _compute_escape_gpu_kernel(width, height, center, half_width, c_val,
                                           p, max_iter, mode, formula)
    except Exception:
        pass

    half_height = half_width * height / width
    re = xp.linspace(center.real - half_width, center.real + half_width, width)
    im = xp.linspace(center.imag - half_height, center.imag + half_height, height)
    re_grid, im_grid = xp.meshgrid(re, im)
    grid = re_grid + 1j * im_grid

    if mode == "julia":
        z = grid
        c = xp.full_like(z, c_val)
    else:  # mandelbrot
        z = xp.zeros_like(grid)
        c = grid

    escaped = xp.zeros((height, width), dtype=bool)
    smooth_iter = xp.zeros((height, width), dtype=xp.float64)
    bailout_sq = 1e12  # (1e6)**2; compare squared magnitude, skip sqrt until needed

    # GPU fallback: dense, every pixel recomputed every iteration (no fancy-indexing/
    # compaction), with escaped pixels frozen via xp.where instead of being
    # sliced out. This trades some redundant FLOPs (which the GPU has to
    # spare) for avoiding per-iteration host<->device syncs and irregular-
    # memory gather/scatter, which is what actually dominates GPU runtime at
    # these resolutions.
    check_every = 32  # how often to sync and check for early exit
    with np.errstate(over="ignore", invalid="ignore"):
        for n in range(max_iter):
            if formula == "burning_ship":
                zt = xp.abs(z.real) + 1j * xp.abs(z.imag)
            elif formula == "tricorn":
                zt = xp.conj(z)
            else:
                zt = z
            z_next = zt ** p + c

            mag_sq = z_next.real * z_next.real + z_next.imag * z_next.imag
            newly = (~escaped) & (mag_sq > bailout_sq)
            mod = xp.sqrt(mag_sq)
            # smooth escape count: n + 1 - log(log|z|)/log(p)
            smooth_candidate = (n + 1) - xp.log(xp.log(mod)) / log_p
            smooth_iter = xp.where(newly, smooth_candidate, smooth_iter)
            escaped = escaped | newly
            z = xp.where(escaped, z, z_next)

            if (n + 1) % check_every == 0 and bool(escaped.all()):
                break

    return cp.asnumpy(smooth_iter), cp.asnumpy(escaped)


def colorize(smooth_iter, escaped, cmap_name):
    """Cyclic band coloring: escaped points cycle through the colormap every
    COLOR_CYCLE escape-steps (so detail stays visible regardless of how far a
    point is from the fractal boundary), interior (non-escaping) points are
    fixed black."""
    cmap = COLORMAPS.get(cmap_name) or next(iter(COLORMAPS.values()))
    t = (smooth_iter / COLOR_CYCLE) % 1.0
    rgb = cmap(t)
    rgb[~escaped] = 0
    return rgb


NEWTON_TOL = 1e-6


def compute_newton(width, height, center, half_width, c_val, p, max_iter, use_gpu=False):
    """Newton's method fractal: find the roots of f(z) = z^p - c by iterating
    z_{n+1} = z_n - f(z_n)/f'(z_n) from every pixel as a starting z_0, then
    color each pixel by which of the p roots it converged to (and how fast).

    Returns (root_index, iter_count, converged), all (height, width) host
    (NumPy) arrays: root_index is which root (0..p-1) the point settled on,
    iter_count is how many steps it took, converged marks points that
    actually reached a root within max_iter.
    """
    xp = cp if (use_gpu and GPU_AVAILABLE) else np

    if xp is np:
        # CPU: shrink the working set as points converge (see compute_escape).
        half_height = half_width * height / width
        re = np.linspace(center.real - half_width, center.real + half_width, width)
        im = np.linspace(center.imag - half_height, center.imag + half_height, height)
        re_grid, im_grid = np.meshgrid(re, im)
        z = re_grid + 1j * im_grid

        converged = np.zeros((height, width), dtype=bool)
        iter_count = np.full((height, width), max_iter, dtype=np.float64)
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            for n in range(max_iter):
                mask = ~converged
                zm = z[mask]
                fprime = p * zm ** (p - 1)
                fprime = np.where(np.abs(fprime) < 1e-12, 1e-12 + 0j, fprime)
                step = (zm ** p - c_val) / fprime
                z[mask] = zm - step
                newly = np.zeros_like(converged)
                newly[mask] = np.abs(step) < NEWTON_TOL
                iter_count[newly] = n
                converged |= newly
                if converged.all():
                    break

        roots = [c_val ** (1.0 / p) * (np.exp(2j * np.pi * k / p)) for k in range(p)]
        best_dist = np.full((height, width), np.inf)
        root_index = np.zeros((height, width), dtype=np.int32)
        for k, root in enumerate(roots):
            dist = np.abs(z - root)
            better = dist < best_dist
            best_dist = np.where(better, dist, best_dist)
            root_index = np.where(better, k, root_index)
        return root_index, iter_count, converged

    # GPU, first choice: single fused kernel launch (see
    # _compute_newton_gpu_kernel), falls through to the dense CuPy loop
    # below if kernel compilation/execution fails for any reason.
    try:
        return _compute_newton_gpu_kernel(width, height, center, half_width, c_val,
                                           p, max_iter)
    except Exception:
        pass

    half_height = half_width * height / width
    re = xp.linspace(center.real - half_width, center.real + half_width, width)
    im = xp.linspace(center.imag - half_height, center.imag + half_height, height)
    re_grid, im_grid = xp.meshgrid(re, im)
    z = re_grid + 1j * im_grid

    converged = xp.zeros((height, width), dtype=bool)
    iter_count = xp.full((height, width), max_iter, dtype=xp.float64)

    # GPU fallback: dense, no per-iteration sync (see compute_escape).
    check_every = 32
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        for n in range(max_iter):
            fprime = p * z ** (p - 1)
            fprime = xp.where(xp.abs(fprime) < 1e-12, 1e-12 + 0j, fprime)
            step = (z ** p - c_val) / fprime
            z_next = z - step
            newly = (~converged) & (xp.abs(step) < NEWTON_TOL)
            iter_count = xp.where(newly, n, iter_count)
            converged = converged | newly
            z = xp.where(converged, z, z_next)
            if (n + 1) % check_every == 0 and bool(converged.all()):
                break

    # nth roots of c_val, assign each pixel to whichever root it landed nearest
    roots = [c_val ** (1.0 / p) * (np.exp(2j * np.pi * k / p)) for k in range(p)]
    best_dist = xp.full((height, width), xp.inf)
    root_index = xp.zeros((height, width), dtype=xp.int32)
    for k, root in enumerate(roots):
        dist = xp.abs(z - root)
        better = dist < best_dist
        best_dist = xp.where(better, dist, best_dist)
        root_index = xp.where(better, k, root_index)

    return cp.asnumpy(root_index), cp.asnumpy(iter_count), cp.asnumpy(converged)


def _hsv_to_rgb(h, s, v):
    """Vectorized HSV -> RGB, h/s/v each in [0,1], returns uint8 (..., 3) array."""
    i = (h * 6.0).astype(int) % 6
    f = (h * 6.0) - np.floor(h * 6.0)
    p_ = v * (1 - s)
    q = v * (1 - f * s)
    t = v * (1 - (1 - f) * s)
    r = np.select([i == 0, i == 1, i == 2, i == 3, i == 4, i == 5], [v, q, p_, p_, t, v])
    g = np.select([i == 0, i == 1, i == 2, i == 3, i == 4, i == 5], [t, v, v, q, p_, p_])
    b = np.select([i == 0, i == 1, i == 2, i == 3, i == 4, i == 5], [p_, p_, t, v, v, q])
    return (np.stack([r, g, b], axis=-1) * 255).astype(np.uint8)


def colorize_newton(root_index, iter_count, converged, p, max_iter):
    """One hue per root (evenly spaced around the color wheel), shaded darker
    the longer a point took to converge; points that never converge are
    black."""
    hue = root_index.astype(np.float64) / max(p, 1)
    shade = 1.0 - np.clip(iter_count / max_iter, 0.0, 1.0)
    value = 0.35 + 0.65 * shade
    rgb = _hsv_to_rgb(hue, np.ones_like(hue), value)
    rgb[~converged] = 0
    return rgb


# ---------------------------------------------------------------------------
# GUI
# ---------------------------------------------------------------------------

class FractalApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Real-Time Fractal Explorer")
        self.resizable(True, True)
        self.geometry(f"{WIDTH + 260}x{HEIGHT + 40}")
        self.is_fullscreen = False
        self.bind("<F11>", self._toggle_fullscreen)
        self.bind("<Escape>", self._exit_fullscreen)

        self.canvas_width = WIDTH
        self.canvas_height = HEIGHT

        self.default_cmap = "inferno" if "inferno" in COLORMAPS else "fire"

        self.mode = tk.StringVar(value=DEFAULT_MODE)
        self.formula = tk.StringVar(value=DEFAULT_FORMULA)
        self.c_re = tk.DoubleVar(value=DEFAULT_C_RE)
        self.c_im = tk.DoubleVar(value=DEFAULT_C_IM)
        self.exponent = tk.IntVar(value=DEFAULT_EXPONENT)
        self.max_iter = tk.IntVar(value=DEFAULT_MAX_ITER)
        self.zoom = tk.DoubleVar(value=DEFAULT_ZOOM)  # half-width = 2.0 / zoom
        self.cmap_name = tk.StringVar(value=self.default_cmap)
        self.use_gpu = tk.BooleanVar(value=GPU_AVAILABLE)

        self.center = DEFAULT_CENTER
        self._redraw_job = None
        self._resize_job = None
        self._drag_start = None
        self._photo = None

        self._build_ui()
        self._update_mode_controls_state()
        self.schedule_redraw()

    # -- UI construction ----------------------------------------------------
    def _build_ui(self):
        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(0, weight=1)

        main = ttk.Frame(self, padding=8)
        main.grid(row=0, column=0, sticky="nsew")
        main.grid_rowconfigure(0, weight=1)
        main.grid_columnconfigure(0, weight=1)

        self.canvas = tk.Canvas(main, width=WIDTH, height=HEIGHT, bg="black",
                                 highlightthickness=1, highlightbackground="#444")
        self.canvas.grid(row=0, column=0, rowspan=20, sticky="nsew", padx=(0, 10))
        self.canvas.bind("<ButtonPress-1>", self._on_press)
        self.canvas.bind("<B1-Motion>", self._on_drag)
        self.canvas.bind("<ButtonRelease-1>", self._on_release)
        self.canvas.bind("<MouseWheel>", self._on_wheel)       # Windows
        self.canvas.bind("<Button-4>", self._on_wheel)          # Linux scroll up
        self.canvas.bind("<Button-5>", self._on_wheel)          # Linux scroll down
        self.canvas.bind("<Configure>", self._on_canvas_resize)

        panel = ttk.Frame(main)
        panel.grid(row=0, column=1, sticky="n")

        row = 0
        ttk.Label(panel, text="Mode").grid(row=row, column=0, sticky="w")
        row += 1
        mode_frame = ttk.Frame(panel)
        mode_frame.grid(row=row, column=0, sticky="w", pady=(0, 8))
        self.julia_radio = ttk.Radiobutton(mode_frame, text="Julia", variable=self.mode,
                                            value="julia", command=self.schedule_redraw)
        self.julia_radio.pack(side="left")
        self.mandelbrot_radio = ttk.Radiobutton(mode_frame, text="Mandelbrot",
                                                 variable=self.mode, value="mandelbrot",
                                                 command=self.schedule_redraw)
        self.mandelbrot_radio.pack(side="left")
        row += 1

        ttk.Label(panel, text="Formula").grid(row=row, column=0, sticky="w")
        row += 1
        formula_frame = ttk.Frame(panel)
        formula_frame.grid(row=row, column=0, sticky="w", pady=(0, 8))
        ttk.Radiobutton(formula_frame, text="Standard", variable=self.formula,
                         value="standard", command=self._on_formula_change).pack(side="left")
        ttk.Radiobutton(formula_frame, text="Burning Ship", variable=self.formula,
                         value="burning_ship",
                         command=self._on_formula_change).pack(side="left")
        ttk.Radiobutton(formula_frame, text="Tricorn", variable=self.formula,
                         value="tricorn", command=self._on_formula_change).pack(side="left")
        ttk.Radiobutton(formula_frame, text="Newton", variable=self.formula,
                         value="newton", command=self._on_formula_change).pack(side="left")
        row += 1

        self.eq_label = ttk.Label(panel, text="", font=("Consolas", 11, "bold"))
        self.eq_label.grid(row=row, column=0, columnspan=2, sticky="w", pady=(0, 10))
        row += 1

        row = self._add_slider(panel, row, "Re(c)", self.c_re, -2.0, 2.0, decimals=2)
        row = self._add_slider(panel, row, "Im(c)", self.c_im, -2.0, 2.0, decimals=2)
        row = self._add_slider(panel, row, "Exponent n", self.exponent, 2, 6,
                                integer=True)
        row = self._add_slider(panel, row, "Max iterations", self.max_iter, 50, 400,
                                integer=True)

        ttk.Label(panel, text="Use mouse wheel to zoom in/out",
                  foreground="#888", font=("Segoe UI", 9, "italic")).grid(
            row=row, column=0, columnspan=2, sticky="w", pady=(0, 8))
        row += 1

        ttk.Label(panel, text="Colormap").grid(row=row, column=0, sticky="w", pady=(8, 0))
        row += 1
        cmap_choice = ttk.Combobox(panel, textvariable=self.cmap_name,
                                    values=sorted(COLORMAPS.keys()), state="readonly",
                                    width=18)
        cmap_choice.grid(row=row, column=0, sticky="w")
        cmap_choice.bind("<<ComboboxSelected>>", lambda e: self.schedule_redraw())
        row += 1

        gpu_label = "Use GPU (CuPy)" if GPU_AVAILABLE else "Use GPU (no GPU/CuPy found)"
        gpu_check = ttk.Checkbutton(panel, text=gpu_label, variable=self.use_gpu,
                                     command=self.schedule_redraw)
        gpu_check.grid(row=row, column=0, sticky="w", pady=(8, 0))
        if not GPU_AVAILABLE:
            gpu_check.state(["disabled"])
        row += 1

        ttk.Button(panel, text="Reset view", command=self._reset_view).grid(
            row=row, column=0, sticky="w", pady=(12, 0))
        row += 1

        ttk.Button(panel, text="Save image...", command=self._save_image).grid(
            row=row, column=0, sticky="w", pady=(4, 0))
        row += 1

        ttk.Button(panel, text="Toggle fullscreen (F11)",
                   command=self._toggle_fullscreen).grid(
            row=row, column=0, sticky="w", pady=(4, 0))
        row += 1

        self.status = ttk.Label(panel, text="", foreground="#666")
        self.status.grid(row=row, column=0, sticky="w", pady=(12, 0))

    def _add_slider(self, parent, row, label, var, lo, hi, integer=False, log=False,
                     decimals=None, step=None):
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky="w")
        row += 1
        value_label = ttk.Label(parent, text="", width=20)

        def on_move(_evt=None):
            v = var.get()
            if decimals is not None:
                rounded = round(v, decimals)
                if rounded != v:
                    var.set(rounded)
                    v = rounded
                value_label.config(text=f"{v:.{decimals}f}")
            else:
                value_label.config(text=f"{v:.4g}")
            self.schedule_redraw()

        slider_row = ttk.Frame(parent)
        slider_row.grid(row=row, column=0, sticky="w")

        def nudge(sign):
            if step is None:
                s = 1 if integer else (hi - lo) / 100
            else:
                s = step
            if log:
                # multiplicative step for log-scaled sliders (e.g. zoom)
                new_v = var.get() * (s if sign > 0 else 1 / s)
            else:
                new_v = var.get() + sign * s
            var.set(min(max(new_v, lo), hi))
            on_move()

        ttk.Button(slider_row, text="◀", width=2,
                   command=lambda: nudge(-1)).pack(side="left")

        def jump_to_click(event):
            width = scale.winfo_width()
            handle_pad = 12  # approx half the ttk slider-handle width in pixels
            usable = max(width - 2 * handle_pad, 1)
            frac = (event.x - handle_pad) / usable
            frac = min(max(frac, 0.0), 1.0)
            if log:
                new_v = lo * (hi / lo) ** frac
            else:
                new_v = lo + frac * (hi - lo)
            var.set(new_v)
            on_move()
            return "break"

        scale = ttk.Scale(slider_row, from_=lo, to=hi, orient="horizontal",
                           variable=var, command=lambda _v: on_move(), length=180)
        scale.pack(side="left", padx=2)
        scale.bind("<Button-1>", jump_to_click)

        ttk.Button(slider_row, text="▶", width=2,
                   command=lambda: nudge(1)).pack(side="left")

        value_label.grid(row=row, column=1, sticky="e", padx=(6, 0))
        on_move()
        row += 1
        return row

    # -- Interaction ---------------------------------------------------------
    def _on_press(self, event):
        if self.mode.get() == "mandelbrot":
            # click picks this point as the new Julia constant and switches mode
            c = self._pixel_to_complex(event.x, event.y)
            self.c_re.set(round(c.real, 6))
            self.c_im.set(round(c.imag, 6))
            self.mode.set("julia")
            self.schedule_redraw()
        self._drag_start = (event.x, event.y, self.center)

    def _on_drag(self, event):
        if self._drag_start is None or self.mode.get() == "mandelbrot":
            return
        sx, sy, start_center = self._drag_start
        half_width = 2.0 / self.zoom.get()
        half_height = half_width * self.canvas_height / self.canvas_width
        dx = (event.x - sx) / self.canvas_width * (2 * half_width)
        dy = (event.y - sy) / self.canvas_height * (2 * half_height)
        self.center = start_center - complex(dx, dy)
        self.schedule_redraw()

    def _on_release(self, _event):
        self._drag_start = None

    def _on_wheel(self, event):
        delta = event.delta if hasattr(event, "delta") and event.delta else (
            120 if getattr(event, "num", None) == 4 else -120)
        factor = 1.5 if delta > 0 else (1 / 1.5)
        old_zoom = self.zoom.get()
        new_zoom = min(max(old_zoom * factor, 0.2), 1e12)
        if new_zoom == old_zoom:
            return

        # keep the point under the cursor fixed on screen while zooming
        c_cursor = self._pixel_to_complex(event.x, event.y)
        self.zoom.set(new_zoom)
        half_width = 2.0 / new_zoom
        half_height = half_width * self.canvas_height / self.canvas_width
        w, h = self.canvas_width, self.canvas_height
        new_re = c_cursor.real + half_width * (1 - 2 * event.x / w)
        new_im = c_cursor.imag + half_height * (1 - 2 * event.y / h)
        self.center = complex(new_re, new_im)
        self.schedule_redraw()

    def _pixel_to_complex(self, x, y):
        half_width = 2.0 / self.zoom.get()
        half_height = half_width * self.canvas_height / self.canvas_width
        re = self.center.real - half_width + (x / self.canvas_width) * (2 * half_width)
        im = self.center.imag - half_height + (y / self.canvas_height) * (2 * half_height)
        return complex(re, im)

    def _reset_view(self):
        self.center = DEFAULT_CENTER
        self.mode.set(DEFAULT_MODE)
        self.formula.set(DEFAULT_FORMULA)
        self._update_mode_controls_state()
        self.c_re.set(DEFAULT_C_RE)
        self.c_im.set(DEFAULT_C_IM)
        self.exponent.set(DEFAULT_EXPONENT)
        self.max_iter.set(DEFAULT_MAX_ITER)
        self.zoom.set(DEFAULT_ZOOM)
        self.cmap_name.set(self.default_cmap)
        self.schedule_redraw()

    def _on_formula_change(self):
        if self.formula.get() == "newton":
            self.mode.set("julia")  # Newton always sweeps starting z, like Julia mode
        self._update_mode_controls_state()
        self.schedule_redraw()

    def _update_mode_controls_state(self):
        is_newton = self.formula.get() == "newton"
        state = ["disabled"] if is_newton else ["!disabled"]
        self.julia_radio.state(state)
        self.mandelbrot_radio.state(state)

    def _on_canvas_resize(self, event):
        if event.width < 10 or event.height < 10:
            return
        if event.width == self.canvas_width and event.height == self.canvas_height:
            return
        self.canvas_width = event.width
        self.canvas_height = event.height
        if self._resize_job is not None:
            self.after_cancel(self._resize_job)
        self._resize_job = self.after(RESIZE_DEBOUNCE_MS, self.redraw)

    def _toggle_fullscreen(self, _event=None):
        self.is_fullscreen = not self.is_fullscreen
        self.attributes("-fullscreen", self.is_fullscreen)

    def _exit_fullscreen(self, _event=None):
        if self.is_fullscreen:
            self.is_fullscreen = False
            self.attributes("-fullscreen", False)

    def _save_image(self):
        path = filedialog.asksaveasfilename(
            title="Save fractal image",
            defaultextension=".png",
            filetypes=[("PNG image", "*.png"), ("JPEG image", "*.jpg"),
                       ("All files", "*.*")],
        )
        if not path:
            return

        aspect = self.canvas_width / self.canvas_height
        if aspect >= 1:
            export_w = EXPORT_MAX_SIDE
            export_h = int(EXPORT_MAX_SIDE / aspect)
        else:
            export_h = EXPORT_MAX_SIDE
            export_w = int(EXPORT_MAX_SIDE * aspect)

        rgb, _c_val, _p, _formula = self._render_rgb(export_w, export_h)
        img = Image.fromarray(rgb, mode="RGB")
        img.save(path)
        self.status.config(text=f"Saved {export_w}x{export_h} -> {path}")

    # -- Rendering ------------------------------------------------------------
    def schedule_redraw(self):
        if self._redraw_job is not None:
            self.after_cancel(self._redraw_job)
        self._redraw_job = self.after(DEBOUNCE_MS, self.redraw)

    def _render_rgb(self, w, h):
        """Compute the colored fractal image at the given resolution using
        current UI parameters. Returns (rgb, c_val, p, formula)."""
        c_val = complex(self.c_re.get(), self.c_im.get())
        p = self.exponent.get()
        max_iter = self.max_iter.get()
        half_width = 2.0 / self.zoom.get()
        formula = self.formula.get()
        use_gpu = self.use_gpu.get()

        if formula == "newton":
            try:
                root_index, iter_count, converged = compute_newton(
                    w, h, self.center, half_width, c_val, p, max_iter, use_gpu=use_gpu)
            except Exception as exc:
                self.use_gpu.set(False)
                self.status.config(text=f"GPU render failed, falling back to CPU: {exc}")
                root_index, iter_count, converged = compute_newton(
                    w, h, self.center, half_width, c_val, p, max_iter, use_gpu=False)
            rgb = colorize_newton(root_index, iter_count, converged, p, max_iter)
        else:
            try:
                smooth_iter, escaped = compute_escape(w, h, self.center, half_width, c_val, p,
                                                       max_iter, self.mode.get(),
                                                       use_gpu=use_gpu, formula=formula)
            except Exception as exc:
                self.use_gpu.set(False)
                self.status.config(text=f"GPU render failed, falling back to CPU: {exc}")
                smooth_iter, escaped = compute_escape(w, h, self.center, half_width, c_val, p,
                                                       max_iter, self.mode.get(), use_gpu=False,
                                                       formula=formula)
            rgb = colorize(smooth_iter, escaped, self.cmap_name.get())

        if formula != "newton" and self.mode.get() == "mandelbrot":
            self._draw_marker(rgb, c_val, half_width)

        return rgb, c_val, p, formula

    def redraw(self):
        self._redraw_job = None
        self._resize_job = None
        w, h = self.canvas_width, self.canvas_height

        rgb, c_val, p, formula = self._render_rgb(w, h)

        img = Image.fromarray(rgb, mode="RGB")
        self._photo = ImageTk.PhotoImage(img)
        self.canvas.delete("all")
        self.canvas.create_image(0, 0, image=self._photo, anchor="nw")
        self._update_equation_label(c_val, p)

    def _update_equation_label(self, c_val, p):
        formula = self.formula.get()
        if formula == "newton":
            text = (f"zₙ₊₁ = zₙ - (zₙ^{p} - c)/({p}·zₙ^{p - 1}),   "
                    f"c = {c_val.real:.2f}{'+' if c_val.imag >= 0 else '-'}"
                    f"{abs(c_val.imag):.2f}i  (roots of zⁿ = c)")
            self.eq_label.config(text=text)
            return

        if formula == "burning_ship":
            zterm = f"(|Re zₙ| + i|Im zₙ|)^{p}"
        elif formula == "tricorn":
            zterm = f"conj(zₙ)^{p}"
        else:
            zterm = f"zₙ^{p}"

        sign = "+" if c_val.imag >= 0 else "-"
        if self.mode.get() == "julia":
            text = f"zₙ₊₁ = {zterm} + c,   c = {c_val.real:.2f} {sign} {abs(c_val.imag):.2f}i"
        else:
            text = f"zₙ₊₁ = {zterm} + c,   z₀ = 0,  c = pixel position"
        self.eq_label.config(text=text)

    def _draw_marker(self, rgb, c_val, half_width):
        w, h = self.canvas_width, self.canvas_height
        half_height = half_width * h / w
        x = int((c_val.real - (self.center.real - half_width)) / (2 * half_width) * w)
        y = int((c_val.imag - (self.center.imag - half_height)) / (2 * half_height) * h)
        size = 4
        x0, x1 = max(x - size, 0), min(x + size + 1, w)
        y0, y1 = max(y - size, 0), min(y + size + 1, h)
        if 0 <= x < w and 0 <= y < h:
            rgb[max(y - 1, 0):min(y + 2, h), x0:x1] = [255, 255, 255]
            rgb[y0:y1, max(x - 1, 0):min(x + 2, w)] = [255, 255, 255]


if __name__ == "__main__":
    app = FractalApp()
    app.mainloop()

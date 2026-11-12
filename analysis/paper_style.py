"""Colours and plot style shared by the figure notebooks (the paper's common style file).

Values are the RGB colours the figure code has always used. The Illustrator files are CMYK documents, so the same
colours read back slightly differently there (task-1 pink #EC008C -> #EB098D, task-2 green #006838 -> #0E6938,
'red' -> #ED1C24, 'blue' -> #2E3192): those are the CMYK round trips of the values below, not different colours.

Usage in a notebook (after the setup script, so these win over anything the setup defined):
    from paper_style import *
"""
import os, sys
import matplotlib.pyplot as plt
from matplotlib import rcParams

# ── tasks: task 1 (mouse visual / monkey saccade) pink, task 2 (mouse audio / monkey pursuit) green ───────────
TASK1, TASK2 = '#EC008C', '#006838'
TASK1_LIGHT, TASK2_LIGHT = '#F27DBF', '#8FCFAF'
TASK_COLORS = {'Visual': TASK1, 'Audio': TASK2, 'Saccade': TASK1, 'Pursuit': TASK2,
               'V': TASK1, 'A': TASK2, 'S': TASK1, 'P': TASK2, 'task1': TASK1, 'task2': TASK2}
CONFLICT = '#F26522'                         # audio-visual conflict task (Figure S5)

# ── choices ───────────────────────────────────────────────────────────────────────────────────────────────────────
C_LEFT, C_RIGHT = 'blue', 'red'              # matplotlib named colours, as in Figures 2 and 3
C_LEFT_LIGHT, C_RIGHT_LIGHT = '#9FA4D2', '#F1A485'
SIDE_COLS, SIDE_NAMES = (C_LEFT, C_RIGHT), ('Left', 'Right')

# ── neuron classes (Figures 4-6, S4, S5) ─────────────────────────────────────────────────────────────────────────
C_TI, C_TD, C_INT = '#1E86B5', '#111111', '#BBBBBB'
C_RANDOM = '#D9B449'
CLS_COL = {'TI': C_TI, 'TD': C_TD, 'Int': C_INT}
POP_C = {'TI': C_TI, 'TD': C_TD, 'random': C_RANDOM}
LES_C = {'all': '#777777', 'random': C_RANDOM, 'TI': '#2186B5', 'Int': '#8C8C8C', 'TD': '#121313'}

# ── coupling bars (Figure 6): same-side dark, opposite-side light, control yellow ────────────────────────────────
COUP_DARK, COUP_LIGHT, COUP_CTL = '#333333', '#A0A0A0', C_RANDOM

# ── GLM lesion classes (Figure S3 g heat maps and class decoding) ─────────────────────────────────────────────────
GLM_CLASS_COL = {'Choice': '#E8412C', 'Direction': '#E8412C', 'Kinematics': '#7F7F7F', 'Visual': TASK1,
                 'Audio': TASK2, 'Outcome': '#2CA02C', 'Movement': '#BCBE32', 'Target': '#27AAE1',
                 'Context': '#9268AD', 'Reward': '#2CA02C', 'Mixed': '#C8C7C7'}

MK_MARK = {'S': 'o', 'Y': 's'}               # monkey S circles, monkey Y squares
MOUSE_MARK = {'IS-2-1L': 'o', 'KO-5-1L': 's', 'KO-6-00': '^'}

# ── type scale (pt, at the printed size: panels are drawn at their .ai size and placed 1:1) ────────────────────────
FS_TICK = 5                                  # tick labels
FS_LABEL = 6                                 # axis labels
FS_TITLE = 6                                 # panel / column titles
FS_LEGEND = 5                                # legend entries
FS_ANNOT = 5                                 # n, p, slope and other text inside the axes
FS_MIN = 5                                   # nothing smaller (check_figures.py flags it)

# ── axes lines (pt) ───────────────────────────────────────────────────────────────────────────────────────────────
SPINE_W = 0.6
TICK_LEN, TICK_W = 2.5, 0.6
REF_LW = 0.6                                 # chance / reference lines

# ── rcParams (Arial, open spines, text kept as text in the SVG, math text in Arial too) ───────────────────────────
STYLE = {'font.family': 'Arial', 'font.size': FS_LABEL, 'axes.labelsize': FS_LABEL, 'axes.titlesize': FS_TITLE,
         'xtick.labelsize': FS_TICK, 'ytick.labelsize': FS_TICK, 'legend.fontsize': FS_LEGEND,
         'axes.linewidth': SPINE_W, 'xtick.major.size': TICK_LEN, 'ytick.major.size': TICK_LEN,
         'xtick.major.width': TICK_W, 'ytick.major.width': TICK_W, 'xtick.minor.size': TICK_LEN * .6,
         'ytick.minor.size': TICK_LEN * .6, 'xtick.minor.width': TICK_W, 'ytick.minor.width': TICK_W,
         'xtick.major.pad': 2, 'ytick.major.pad': 2, 'axes.labelpad': 2, 'axes.titlepad': 3,
         'axes.spines.right': False, 'axes.spines.top': False, 'figure.dpi': 150, 'savefig.dpi': 800,
         'svg.fonttype': 'none', 'legend.frameon': False, 'mathtext.fontset': 'custom', 'mathtext.rm': 'Arial',
         'mathtext.it': 'Arial:italic', 'mathtext.bf': 'Arial:bold', 'mathtext.default': 'regular'}


def apply_style():
    """Re-apply STYLE (the setup scripts set their own rcParams after this module is imported). Also draws math-text
    sub- and superscripts at FS_MIN / FS_LABEL of their label (matplotlib's default 0.7 makes a 6-pt label's
    subscript 4.2 pt; there is no rcParam for it)."""
    rcParams.update(STYLE)
    try:
        import matplotlib._mathtext as _mt
        _mt.SHRINK_FACTOR = FS_MIN / FS_LABEL
    except (ImportError, AttributeError):
        pass


apply_style()

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))   # repo root
if os.path.join(ROOT, 'src') not in sys.path: sys.path.insert(0, os.path.join(ROOT, 'src'))   # mtrnn importable without installing


def figure_dir(name):
    """analysis/figures/<name>/ as an absolute string ending in '/', created if missing."""
    d = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'figures', name, '')
    os.makedirs(d, exist_ok=True)
    return d


def panel_slot(figure, letter, part=None):
    """(canvas, axes) of a panel in layouts/<figure>.json, page pt: canvas [x, y, w, h] is the SVG's extent on the page,
    axes a list of [x, y, w, h] plot areas measured from the .ai page (layouts/extract_layouts.py --axes). For a panel
    made of several SVGs, `part` picks one: its index in the panel's parts or a piece of its file name."""
    import json
    with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'layouts', figure + '.json')) as f:
        P = json.load(f)['panels'][letter]
    if part is not None:
        parts = P.get('parts', [])
        hit = [p for k, p in enumerate(parts) if k == part or (isinstance(part, str) and part in p['file'])]
        if len(hit) != 1:
            raise KeyError(f'{figure} {letter}: part {part!r} matches {len(hit)} of {[p["file"] for p in parts]}')
        P = hit[0]
    if 'canvas' not in P:
        raise KeyError(f'{figure} {letter}{"" if part is None else " " + str(part)}: no canvas in the layout; '
                       f'run layouts/extract_layouts.py --axes {figure}')
    return P['canvas'], P['axes']


def panel_fig(figure, letter):
    """Figure at the panel's .ai size: the canvas is the panel's slot on the page and each axes sits exactly where the .ai
    draws its plot area. Returns (fig, ax), or (fig, [ax, ...]) for a panel with several plot areas. Save it with a fixed
    canvas (no bbox_inches='tight') so figure_layout.assemble places it 1:1."""
    apply_style()
    (cx, cy, cw, ch), rects = panel_slot(figure, letter)
    fig = plt.figure(figsize=(cw / 72, ch / 72))
    axes = [fig.add_axes([(x - cx) / cw, 1 - (y - cy + h) / ch, w / cw, h / ch]) for x, y, w, h in rects]
    return fig, axes[0] if len(axes) == 1 else axes


def _reading_order(rects, tol=3.0):
    """Indices of [x, y, w, h] rects (y down) sorted top to bottom in rows (tops within tol pt), left to right."""
    idx = sorted(range(len(rects)), key=lambda i: rects[i][1])
    rows, cur = [], []
    for i in idx:
        if cur and rects[i][1] - rects[cur[0]][1] > tol:
            rows.append(cur); cur = []
        cur.append(i)
    if cur:
        rows.append(cur)
    return [i for r in rows for i in sorted(r, key=lambda i: rects[i][0])]


def fit_panel(fig, figure, letter, part=None):
    """Resize an already drawn figure to the panel's .ai canvas and move its axes onto the .ai plot areas
    (layouts/<figure>.json; `part` as in panel_slot). Axes with a white background are matched to the plot areas in
    reading order (rows top to bottom, then left to right); any other axes (colour bars, legend-only axes) keep their
    place relative to the matched ones. Call it just before savefig, and save without bbox_inches='tight'."""
    from matplotlib.colors import to_hex
    (cx, cy, cw, ch), rects = panel_slot(figure, letter, part)
    W, H = fig.get_size_inches() * 72
    if fig.get_layout_engine() is not None:
        fig.canvas.draw()                                  # let tight / constrained layout place the axes once, then freeze
        fig.set_layout_engine('none')
    pos = {ax: ax.get_position() for ax in fig.axes}
    old = {ax: [p.x0 * W, (1 - p.y1) * H, p.width * W, p.height * H] for ax, p in pos.items()}   # pt, y down
    main = [ax for ax in fig.axes if ax.axison and ax.get_frame_on() and ax.patch.get_visible()
            and to_hex(ax.get_facecolor()) == '#ffffff']
    if len(main) != len(rects):                            # axes drawn with a hidden background count too
        main = [ax for ax in fig.axes if ax.axison]
    if len(main) != len(rects):
        raise ValueError(f'{figure} {letter}: {len(main)} white-background axes in the figure, {len(rects)} plot areas in '
                         'the layout (run layouts/extract_layouts.py --axes on the current SVG, or fix the panel)')
    om, nm = _reading_order([old[a] for a in main]), _reading_order(rects)
    target = {main[i]: rects[j] for i, j in zip(om, nm)}
    # other axes: the linear map between the bounding boxes of the matched axes, old -> new
    ob = [min(old[a][0] for a in main), min(old[a][1] for a in main),
          max(old[a][0] + old[a][2] for a in main), max(old[a][1] + old[a][3] for a in main)]
    nb = [min(r[0] for r in rects), min(r[1] for r in rects), max(r[0] + r[2] for r in rects), max(r[1] + r[3] for r in rects)]
    sx = (nb[2] - nb[0]) / max(ob[2] - ob[0], 1e-9); sy = (nb[3] - nb[1]) / max(ob[3] - ob[1], 1e-9)
    for ax in fig.axes:
        if ax not in target:
            x, y, w, h = old[ax]
            target[ax] = [nb[0] + (x - ob[0]) * sx, nb[1] + (y - ob[1]) * sy, w * sx, h * sy]
    # figure-level text and lines (fig.text, fig.add_artist) are in figure fractions: move them with the same map
    def remap(fx, fy):
        x, y = fx * W, (1 - fy) * H
        return (nb[0] + (x - ob[0]) * sx - cx) / cw, 1 - (nb[1] + (y - ob[1]) * sy - cy) / ch
    from matplotlib.text import Text
    from matplotlib.lines import Line2D
    for a in list(fig.texts) + list(fig.lines) + list(fig.artists):
        if a.get_transform() != fig.transFigure:
            continue
        if isinstance(a, Text):
            a.set_position(remap(*a.get_position()))
        elif isinstance(a, Line2D):
            xy = [remap(px, py) for px, py in zip(a.get_xdata(), a.get_ydata())]
            a.set_data([p[0] for p in xy], [p[1] for p in xy])
    fig.set_size_inches(cw / 72, ch / 72)
    for ax, (x, y, w, h) in target.items():
        ax.set_position([(x - cx) / cw, 1 - (y - cy + h) / ch, w / cw, h / ch])
    return fig


def save(fig, fig_dir, name, tight=True):
    """SVG only, white background; tight=False keeps a fixed canvas (matched mouse / monkey panels)."""
    kw = dict(dpi=800, facecolor='white', edgecolor='none')
    if tight: kw['bbox_inches'] = 'tight'
    fig.savefig(fig_dir + name + '.svg', **kw)
    print('saved', fig_dir + name + '.svg')

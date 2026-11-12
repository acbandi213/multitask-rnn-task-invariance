"""Assemble one paper figure's panel SVGs into a full-page SVG laid out like the .ai figure page.

    from figure_layout import assemble
    assemble('figure_4')                       # -> analysis/figures/figure_4/figure_4_full.svg

The page geometry lives in ``analysis/layouts/<figure>.json`` (written once by ``layouts/extract_layouts.py`` from the v4
.ai PDFs, then edited by hand).  Per panel letter the JSON holds

    "label"  : {"text", "x", "y" (top-left of the letter as in the PDF), "origin" (baseline point used to draw it), "size"}
    "kind"   : "svg" (made by the notebook) | "schematic" (drawn in Illustrator)
    "content": [x, y, w, h]  visible extent of the panel on the .ai page (schematic placeholders are drawn here)
    "parts"  : [{"file": glob pattern in the panel folder, "box": [x, y, w, h]}, ...]
               the SVG's top-left goes to (x, y); its scale is w / (SVG width), snapped to 1.0 within +-12 %
               so the fonts stay at the size the notebook drew them.
    "drop_labels": true  (optional, per part) removes bold single-letter <text> elements from that SVG (Figure S8,
               whose notebook draws the whole figure including its own 8-pt letters).
    "keep_background": true (optional, per part) keeps the SVG's white figure background; by default it is made
               transparent so a panel wider than its slot does not paint over its neighbour.
    "fixed": true (optional, per part) the SVG was drawn at its .ai size by ``paper_style.panel_fig``: always placed
               at scale 1.0 at box[:2], with a warning when its size is not the box size.
    "axes", "canvas" (optional, per panel) the .ai plot areas and the panel's canvas in page pt, written by
               ``layouts/extract_layouts.py --axes``; read by ``paper_style.panel_fig`` and ``check_figures.py``.

``extras`` holds page text and rules outside every panel (the 'Mouse' / 'Monkey' section dividers).

Only the standard library and lxml are needed.  Text stays text; every id inside an embedded SVG is prefixed so
clip paths and markers of different panels cannot collide.
"""
import copy
import glob
import json
import os
import re

from lxml import etree

HERE = os.path.dirname(os.path.abspath(__file__))
LAYOUT_DIR = os.path.join(HERE, 'layouts')
FIG_DIR = os.path.join(HERE, 'figures')
SVG_NS = 'http://www.w3.org/2000/svg'
XLINK_NS = 'http://www.w3.org/1999/xlink'
NSMAP = {None: SVG_NS, 'xlink': XLINK_NS}
SNAP = 0.12                                   # |scale - 1| <= SNAP -> scale 1.0
LABEL_FONT = "Arial, Helvetica, 'Liberation Sans', sans-serif"
_UNIT = {'pt': 1.0, 'px': 0.75, 'in': 72.0, 'mm': 72 / 25.4, 'cm': 72 / 2.54, '': 1.0}


# ── reading panel SVGs ─────────────────────────────────────────────────────────────────────────────────────────────
def _len_pt(s):
    m = re.match(r'\s*([-\d.eE+]+)\s*([a-z]*)', s or '')
    return float(m.group(1)) * _UNIT.get(m.group(2), 1.0) if m else None


def read_svg(path):
    """Parsed root element of an SVG file (comments and metadata dropped)."""
    parser = etree.XMLParser(remove_comments=True, huge_tree=True, resolve_entities=False, no_network=True)
    root = etree.parse(path, parser).getroot()
    for md in root.findall('{%s}metadata' % SVG_NS):
        root.remove(md)
    return root


def svg_geometry(root):
    """(viewBox [x, y, w, h], size in pt [w, h])."""
    vb = root.get('viewBox')
    if vb:
        vb = [float(v) for v in re.split(r'[\s,]+', vb.strip())]
    w, h = _len_pt(root.get('width')), _len_pt(root.get('height'))
    if vb is None:
        vb = [0, 0, w, h]
    if w is None or h is None:
        w, h = vb[2], vb[3]
    return vb, [w, h]


_RECT_D = re.compile(r'^\s*M\s*([-\d.]+)\s+([-\d.]+)\s*L\s*([-\d.]+)\s+([-\d.]+)\s*L\s*([-\d.]+)\s+([-\d.]+)\s*'
                     r'L\s*([-\d.]+)\s+([-\d.]+)\s*z\s*$', re.S)


def white_rects(root):
    """Axis-aligned white-filled rectangles in a matplotlib SVG: the figure patch and every axes patch.
    Returns [(kind, [x0, y0, x1, y1])] with kind 'figure' for the first one (patch_1)."""
    out = []
    for p in root.iter('{%s}path' % SVG_NS):
        st = (p.get('style') or '').replace(' ', '').lower()
        if 'fill:#ffffff' not in st:
            continue
        m = _RECT_D.match(p.get('d') or '')
        if not m:
            continue
        v = [float(x) for x in m.groups()]
        xs, ys = v[0::2], v[1::2]
        r = [min(xs), min(ys), max(xs), max(ys)]
        if r[2] - r[0] < 5 or r[3] - r[1] < 5:
            continue
        gid = p.getparent().get('id', '') if p.getparent() is not None else ''
        out.append(('figure' if gid == 'patch_1' else 'axes', r))
    return out


# ── embedding ──────────────────────────────────────────────────────────────────────────────────────────────────────
_URL = re.compile(r'url\(\s*#([^)\s]+)\s*\)')


def _prefix_ids(root, prefix):
    ids = {el.get('id') for el in root.iter() if isinstance(el.tag, str) and el.get('id')}
    href_keys = ('{%s}href' % XLINK_NS, 'href')
    for el in root.iter():
        if not isinstance(el.tag, str):
            continue
        if el.get('id'):
            el.set('id', prefix + el.get('id'))
        for k, v in list(el.attrib.items()):
            if k in href_keys and v.startswith('#') and v[1:] in ids:
                el.set(k, '#' + prefix + v[1:])
            elif 'url(' in v:
                el.set(k, _URL.sub(lambda m: 'url(#%s%s)' % (prefix if m.group(1) in ids else '', m.group(1)), v))
        if el.tag == '{%s}style' % SVG_NS and el.text:      # matplotlib: '*{stroke-linejoin: round; ...}' -> scoped
            txt = _URL.sub(lambda m: 'url(#%s%s)' % (prefix if m.group(1) in ids else '', m.group(1)), el.text)
            el.text = re.sub(r'(^|})\s*\*\s*{', r'\1 #%sroot * {' % prefix, txt)


def _drop_bold_letters(root):
    """Remove <text> elements that are a single bold lowercase letter (a figure's own panel letters)."""
    n = 0
    for t in list(root.iter('{%s}text' % SVG_NS)):
        txt = ''.join(t.itertext()).strip()
        st = (t.get('style') or '') + ' ' + (t.get('font-weight') or '')
        if len(txt) == 1 and txt.islower() and re.search(r'\b(700|bold)\b', st):
            parent = t.getparent()
            if parent is not None:
                parent.remove(t)
                n += 1
    return n


def _clear_background(root):
    """Make matplotlib's white figure patch (patch_1) transparent so a panel that is wider than its .ai slot does
    not paint over its neighbour; the page itself has a white background."""
    for g in root.iter('{%s}g' % SVG_NS):
        if g.get('id') == 'patch_1':
            for p in g.iter('{%s}path' % SVG_NS):
                st = p.get('style') or ''
                if re.search(r'fill:\s*#ffffff', st, re.I):
                    p.set('style', re.sub(r'fill:\s*#ffffff', 'fill: none', st, flags=re.I))
            return True
    return False


def embed(root, box, prefix, snap=SNAP, drop_labels=False, keep_background=False):
    """Nested <svg> for a parsed panel SVG with its top-left at box[:2]; returns (element, scale)."""
    root = copy.deepcopy(root)
    if drop_labels:
        _drop_bold_letters(root)
    if not keep_background:
        _clear_background(root)
    vb, (w, h) = svg_geometry(root)
    scale = box[2] / w if box[2] else 1.0
    if abs(scale - 1) <= snap:
        scale = 1.0
    _prefix_ids(root, prefix)
    el = etree.Element('{%s}svg' % SVG_NS, nsmap=NSMAP)
    el.set('id', prefix + 'root')
    for k, v in (('x', box[0]), ('y', box[1]), ('width', w * scale), ('height', h * scale)):
        el.set(k, '%.3f' % v)
    el.set('viewBox', ' '.join('%.6g' % v for v in vb))
    el.set('overflow', 'visible')
    for k in ('preserveAspectRatio',):
        if root.get(k):
            el.set(k, root.get(k))
    for child in root:
        el.append(child)
    return el, scale


# ── drawing helpers ────────────────────────────────────────────────────────────────────────────────────────────────
def _sub(parent, tag, text=None, **attrs):
    el = etree.SubElement(parent, '{%s}%s' % (SVG_NS, tag))
    for k, v in attrs.items():
        el.set(k.replace('_', '-'), '%.3f' % v if isinstance(v, float) else str(v))
    if text is not None:
        el.text = text
    return el


def _placeholder(parent, box, colour, dash, lines, fs=6.0):
    x, y, w, h = box
    g = _sub(parent, 'g')
    _sub(g, 'rect', x=float(x), y=float(y), width=float(w), height=float(h), fill='none', stroke=colour,
         stroke_width=0.6, stroke_dasharray=dash)
    for i, s in enumerate(lines):
        _sub(g, 'text', s, x=float(x + w / 2), y=float(y + h / 2 + (i - (len(lines) - 1) / 2) * fs * 1.25 + fs * 0.35),
             fill=colour, font_size=fs, font_family=LABEL_FONT, text_anchor='middle')
    return g


def _hex(c):
    if isinstance(c, str):
        return c
    return '#%02x%02x%02x' % tuple(int(round(255 * v)) for v in c)


# ── main entry point ───────────────────────────────────────────────────────────────────────────────────────────────
def load_layout(figure):
    with open(os.path.join(LAYOUT_DIR, figure + '.json')) as f:
        return json.load(f)


def assemble(figure, panel_dir=None, out=None, schematic_dir=None, layout=None, verbose=True):
    """Write the full-page SVG of ``figure`` ('figure_4', 'figure_s8', ...) and return its path.

    panel_dir     folder with the panel SVGs (default analysis/figures/<figure>/)
    out           output path (default <panel_dir>/<figure>_full.svg)
    schematic_dir folder with optional <letter>.svg drawings that replace the grey schematic placeholders
    layout        a layout dict instead of layouts/<figure>.json (for trying edits)
    """
    L = layout or load_layout(figure)
    panel_dir = panel_dir or os.path.join(FIG_DIR, figure)
    out = out or os.path.join(panel_dir, figure + '_full.svg')
    W, H = L['page']['width'], L['page']['height']

    svg = etree.Element('{%s}svg' % SVG_NS, nsmap=NSMAP)
    for k, v in (('version', '1.1'), ('width', '%gpt' % W), ('height', '%gpt' % H), ('viewBox', '0 0 %g %g' % (W, H))):
        svg.set(k, v)
    _sub(svg, 'title', L.get('title', figure))
    _sub(svg, 'rect', x=0, y=0, width=W, height=H, fill='#ffffff', id='page_background')
    panels_g = _sub(svg, 'g', id='panels')
    report, n_embedded = [], 0

    for letter in sorted(L['panels']):
        P = L['panels'][letter]
        g = _sub(panels_g, 'g', id='panel_' + letter)
        kind = P.get('kind', 'svg')
        if kind == 'schematic':
            sfile = os.path.join(schematic_dir, letter + '.svg') if schematic_dir else None
            if sfile and os.path.exists(sfile):
                el, s = embed(read_svg(sfile), P['content'], '%s_%s_schem_' % (figure, letter))
                g.append(el)
                report.append((letter, os.path.basename(sfile), s, 'schematic file'))
            else:
                _placeholder(g, P['content'], '#9A9A9A', '3 2', ['schematic'])
                report.append((letter, None, None, 'schematic placeholder'))
            continue
        for i, part in enumerate(P.get('parts', [])):
            hits = sorted(glob.glob(os.path.join(panel_dir, part['file'])))
            if not hits:
                print('WARNING %s %s: no file matches %s in %s' % (figure, letter, part['file'], panel_dir))
                _placeholder(g, part['box'], '#E0201B', '3 2', ['missing', part['file']], fs=5.0)
                report.append((letter, part['file'], None, 'MISSING'))
                continue
            if len(hits) > 1:
                print('WARNING %s %s: %d files match %s, using %s' % (figure, letter, len(hits), part['file'],
                                                                      os.path.basename(hits[0])))
            root = read_svg(hits[0])
            note = ''
            if part.get('fixed'):                       # drawn at the .ai size by paper_style.panel_fig: always 1:1
                _, (w, h) = svg_geometry(root)
                bw, bh = part['box'][2], part['box'][3]
                if abs(w - bw) > 0.5 or abs(h - bh) > 0.5:
                    note = 'SIZE %.1f x %.1f pt, canvas %.1f x %.1f (saved with a tight bbox?)' % (w, h, bw, bh)
                    print('WARNING %s %s: %s' % (figure, letter, note))
            el, s = embed(root, part['box'], '%s_%s%d_' % (figure.replace('figure_', 'f'), letter, i),
                          snap=float('inf') if part.get('fixed') else SNAP,
                          drop_labels=part.get('drop_labels', False), keep_background=part.get('keep_background', False))
            g.append(el)
            n_embedded += 1
            report.append((letter, os.path.basename(hits[0]), s, note or ('fixed canvas' if part.get('fixed') else '')))

    # extras (section dividers) and panel letters on top
    xg = _sub(svg, 'g', id='page_extras')
    for r in L.get('extras', {}).get('rules', []):
        _sub(xg, 'line', x1=float(r['x0']), y1=float(r['y']), x2=float(r['x1']), y2=float(r['y']),
             stroke=_hex(r.get('color', '#231f20')), stroke_width=float(r.get('width', 1.0)))
    for t in L.get('extras', {}).get('text', []):
        _sub(xg, 'text', t['text'], x=float(t['origin'][0]), y=float(t['origin'][1]), font_size=float(t['size']),
             font_family=LABEL_FONT, font_weight='bold' if t.get('bold') else 'normal', fill=_hex(t.get('color', '#231f20')))
    lg = _sub(svg, 'g', id='panel_letters')
    for letter in sorted(L['panels']):
        lab = L['panels'][letter]['label']
        _sub(lg, 'text', lab['text'], x=float(lab['origin'][0]), y=float(lab['origin'][1]), font_size=float(lab['size']),
             font_family=LABEL_FONT, font_weight='bold', fill='#231f20')

    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    etree.ElementTree(svg).write(out, xml_declaration=True, encoding='utf-8', pretty_print=False)
    if verbose:
        print('%s: %d panel SVGs placed -> %s' % (figure, n_embedded, out))
        for letter, f, s, note in report:
            flag = '' if s in (None, 1.0) else '   <-- scale != 1'
            print('   %s  %-48s %s %s%s' % (letter, f or '', '' if s is None else 'scale %.3f' % s, note, flag))
    return out


def assemble_all(figures=None, out_dir=None, **kw):
    """Assemble every figure that has a layout JSON; out_dir puts all outputs in one folder."""
    figures = figures or sorted(os.path.basename(f)[:-5] for pat in ('figure_*.json', 'extra_*.json')
                                for f in glob.glob(os.path.join(LAYOUT_DIR, pat)))
    return [assemble(f, out=os.path.join(out_dir, f + '_full.svg') if out_dir else None, **kw) for f in figures]


if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('figures', nargs='*')
    ap.add_argument('--out-dir')
    ap.add_argument('--schematic-dir')
    a = ap.parse_args()
    assemble_all(a.figures or None, out_dir=a.out_dir, schematic_dir=a.schematic_dir)

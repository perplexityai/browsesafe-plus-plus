"""Print the README's outlined SVG lockup.

uv run --no-project --with fonttools --with brotli --with uharfbuzz python scripts/build_readme_logo.py FONT.woff2
Typography matches the website: PPLX Sans 460, 25px, -0.045em tracking;
superscript ++ at 400, 0.61em. Outlines avoid GitHub font substitution.
"""
import io
import sys
from pathlib import Path
from fontTools.ttLib import TTFont
from fontTools.varLib.instancer import instantiateVariableFont
from fontTools.pens.svgPathPen import SVGPathPen
import uharfbuzz as hb


def outline(text, weight, size, x, baseline):
    font = TTFont(sys.argv[1])
    # The webfont omits CVT hints; they are not needed for vector outlines.
    if "cvar" in font and "cvt " not in font:
        del font["cvar"]
    # CSS defaults to font-optical-sizing:auto. Pin every axis so the outlines
    # match the browser at this font size, not the webfont's 10px default.
    axes = {axis.axisTag: axis.defaultValue for axis in font["fvar"].axes}
    axes.update(wght=weight, opsz=size)
    font = instantiateVariableFont(font, axes, inplace=True)
    font.flavor = None
    binary = io.BytesIO()
    font.save(binary)
    shaped = hb.Font(hb.Face(binary.getvalue()))
    hb.ot_font_set_funcs(shaped)
    buffer = hb.Buffer()
    buffer.add_str(text)
    buffer.guess_segment_properties()
    hb.shape(shaped, buffer, {"kern": True})
    scale = size / font["head"].unitsPerEm
    glyphs = font.getGlyphSet()
    paths = []
    for info, pos in zip(buffer.glyph_infos, buffer.glyph_positions):
        pen = SVGPathPen(glyphs)
        glyphs[font.getGlyphName(info.codepoint)].draw(pen)
        paths.append(f'<path d="{pen.getCommands()}" transform="translate({x + pos.x_offset * scale:.4f} {baseline - pos.y_offset * scale:.4f}) scale({scale:.8f} {-scale:.8f})"/>')
        x += pos.x_advance * scale - 25 * .045
    return paths, x


# Measured from the leaderboard's .brand at 100% browser zoom: 40px line
# box, 35px centered mark, 6px gap, 29.5px baseline, native <sup> offset.
# Keep the exported canvas identical, rather than independently centering art.
word, end = outline("BrowseSafe", 460, 25, 41, 29.5)
plus, end = outline("++", 400, 25 * .61, end + 1, 20.671875)
mark = (Path(__file__).resolve().parents[1] / "docs/assets/logo.svg").read_text()
mark = mark[mark.index("<g "):mark.index("</svg>")].replace("#27251e", "currentColor")
print(f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {end:.3f} 40" role="img" aria-label="BrowseSafe++">')
print('<style>svg{color:#263c33}@media(prefers-color-scheme:dark){svg{color:#e5eadb}}</style>')
print('<g transform="translate(0 2.5) scale(.3645833333)">' + mark + '</g>')
print('<g fill="currentColor">' + "".join(word + plus) + '</g></svg>')

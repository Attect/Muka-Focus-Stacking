"""Bundle the check images into one clearly-named folder plus a single index page.

Everything is inlined into `index.html` as JPEG data URIs so the page renders
anywhere, without depending on relative paths — and the full-resolution PNGs are
copied alongside for 100% inspection.

    python tools/mkreview.py
"""
import base64
import io
import os
import shutil

from PIL import Image

Image.MAX_IMAGE_PIXELS = None

OUT = "out/检查_2026-09-22"
ITEMS = [
    ("out/full_default_cmp.png", "01_整幅对比_旧默认-新默认.png",
     "整幅并排（各 1600px 宽）。左＝旧默认，右＝新默认。这个尺度上两者看不出差别是正常的"
     "——颗粒是亚像素级的；这张只用来确认没有宏观变化。"),
    ("out/zones_100pct.png", "02_六区域_1比1_旧-新-参照.png",
     "这张是用来判是否完美的一张：六个区域在 1:1 原尺寸下的三联，每行左＝旧默认、中＝新默认、"
     "右＝AF 参照成品。自上而下：靴面（红色带缝线）、手/深色道具、躯干与腰带、轮廓交界处、"
     "背景墙浮雕、底座花与墙。每行左上标注了画布坐标。"),
    ("out/band_gate_cmp.png", "03_轮廓带_4倍_四档对比.png",
     "上一轮你看的那条轮廓带，4 倍放大，五个设置并排：旧默认 / gate 0.6 / gate 0.7（新默认）/ "
     "gate 0.8 / AF 参照。绿线＝轮廓，黄线＝离轮廓的距离。挑观感就用这张。"),
    ("out/full_diff.png", "04_整幅改动量_改动落在哪.png",
     "左：|新−旧|×4（越亮＝改得越多）。右：新默认的图，凡改动 >4 级处标红。"
     "用来确认改动是散在光滑面上，还是连纹理一起改。"),
    ("out/halo_fix_cmp.png", "06_轮廓交界_光晕修复_3倍.png",
     "轮廓交界处（画布 x2760-2880, y2820-2920）3 倍放大，五个面板：旧默认 / 上一版 / "
     "本版（上侧也夹） / AF 参照 / 墙合焦的原帧。下排是 150-215 级拉伸，把微弱的亮环显出来。"),
    ("out/grain_new.png", "05_光滑面颗粒_高通放大.png",
     "上一次交付的那张：光滑面 1:1（上排）＋ 高通×6（下排，颗粒被直接显出来），"
     "三列＝旧默认 / 当时的新默认 / AF 参照。"),
]

INDEX_CSS = """
body{font:15px/1.7 -apple-system,"Segoe UI","Microsoft YaHei",sans-serif;
     margin:0;padding:28px 32px;background:#f5f6f8;color:#1c1e21}
h1{font-size:21px;margin:0 0 4px}
p.lead{margin:0 0 22px;color:#555}
section{background:#fff;border:1px solid #e3e6ea;border-radius:10px;padding:16px 18px;margin:0 0 20px}
h2{font-size:16px;margin:0 0 8px}
.cap{margin:0 0 12px;color:#444}
img{max-width:100%;border:1px solid #dfe3e8;border-radius:6px;display:block}
code{background:#eef1f4;padding:1px 5px;border-radius:4px;font-size:13px}
ul{margin:6px 0 0 18px;padding:0}
li{margin:2px 0}
"""


def preview_png(path, max_w=1700, quality=88):
    im = Image.open(path).convert("RGB")
    if im.size[0] > max_w:
        h = int(round(im.size[1] * max_w / im.size[0]))
        im = im.resize((max_w, h), Image.LANCZOS)
    buf = io.BytesIO()
    im.save(buf, "JPEG", quality=quality)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def main():
    os.makedirs(OUT, exist_ok=True)
    parts = []
    for src, name, cap in ITEMS:
        if not os.path.exists(src):
            print("missing", src)
            continue
        shutil.copy2(src, os.path.join(OUT, name))
        b64 = preview_png(src)
        title = os.path.splitext(name)[0]
        parts.append(
            '<section><h2>%s</h2><p class="cap">%s</p>'
            '<img alt="%s" src="data:image/jpeg;base64,%s"></section>'
            % (title, cap, title, b64)
        )
        print("bundled", src, "->", os.path.join(OUT, name))
    # the finished picture, linked rather than inlined at full size
    for src, name in [("out/可莉-全清晰.png", "07_成品_全分辨率.png"),
                      ("out/可莉-全清晰_预览.jpg", "08_成品预览_1600px.jpg")]:
        if os.path.exists(src):
            shutil.copy2(src, os.path.join(OUT, name))
            print("bundled", src, "->", os.path.join(OUT, name))
    html = (
        '<!DOCTYPE html><html lang="zh-CN"><head><meta charset="utf-8">'
        "<title>FocusMerge 效果检查</title><style>%s</style></head><body>"
        "<h1>FocusMerge 效果检查表</h1>"
        '<p class="lead">全部文件也直接放在这个文件夹里：'
        "<code>B:\\FocusMerge\\out\\检查_2026-09-22\\</code></p>%s"
        "<section><h2>还可用的设置</h2><ul>"
        "<li><code>--band-energy-gate 0.6</code>：更干净（塑料面颗粒 1.46，带内 7.95）</li>"
        "<li><code>--band-energy-gate 0.7</code>：当前默认（1.54 / 8.15）</li>"
        "<li><code>--band-energy-gate 0.8</code>：更接近旧的观感（1.69 / 8.36）</li>"
        "<li><code>--band-energy-smooth 0</code>：完全回到旧版（逐位相同）</li>"
        "<li><code>--no-clamp-hi</code>：关掉本轮的光晕修复（会带回轮廓外侧的亮环）</li>"
        "</ul></section></body></html>" % (INDEX_CSS, "".join(parts))
    )
    with open(os.path.join(OUT, "index.html"), "w", encoding="utf-8") as f:
        f.write(html)
    print("wrote", os.path.join(OUT, "index.html"))


if __name__ == "__main__":
    main()

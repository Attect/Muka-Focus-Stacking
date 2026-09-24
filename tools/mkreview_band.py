"""Bundle this round's evidence: the light band outside a silhouette.

    python tools/mkreview_band.py
"""
import base64
import io
import os
import shutil

from PIL import Image

Image.MAX_IMAGE_PIXELS = None

OUT = "out/检查_亮带2"
CROP = (2250, 1470, 3170, 2210)   # the region the user marked

NUMBERS = """
<p class="cap">轮廓外各距离处，输出比<b>真值</b>亮多少级。真值＝把主体像素排除出测度窗口后
定出的合焦帧（墙在标记区内恒为第 33 帧），逐 8×8 单元取。AF 成品是同口径的对照。</p>
<table>
<tr><th>距轮廓</th><th>+3 px</th><th>+5 px</th><th>+8 px</th><th>+12 px</th><th>+16 px</th>
<th>+24 px</th><th>+40 px</th></tr>
<tr><td>旧默认</td><td>+12.1</td><td>+2.7</td><td>+3.3</td><td>+2.3</td><td>+2.1</td><td>+1.0</td><td>+1.0</td></tr>
<tr><td><b>新默认</b></td><td><b>+3.9</b></td><td><b>+0.7</b></td><td><b>−0.2</b></td>
<td><b>−0.8</b></td><td><b>−0.8</b></td><td><b>−0.6</b></td><td>+0.2</td></tr>
<tr><td>AF 成品（人工修过）</td><td>+10.2</td><td>+0.9</td><td>+0.9</td><td>−0.8</td><td>−0.7</td>
<td>−1.7</td><td>−1.3</td></tr>
</table>
<p class="cap">从 +5 px 往外，新默认已经和 AF 成品同级或更干净；<b>+3 px 那一档 AF 成品反而更高
（+10.2 对 +3.9）</b>，说明紧贴轮廓 1–3 像素的"亮边"在参考成品里同样存在，是边缘过冲本身，
不是这一版新增的。</p>
"""

ITEMS = [
    ("out/rim_cmp.png", "05_残差图_对照真值.png",
     "把每个 8×8 单元相对真值的偏差画成颜色（每 8 级一个色阶）：红＝比真值亮，蓝＝暗。左＝旧默认，一条<b>贴着整个轮廓的宽红晕</b>；右＝新默认，宽晕消失，只剩轮廓本身那条细线。"),
    ("out/depth_err_cmp.png", "01_深度误差_前后.png",
     "蓝＝深度比墙面真实深度<b>偏近</b>，红＝偏远（每 6 帧一级色阶），黄线＝轮廓。"
     "左：旧默认——一条<b>贴着整个轮廓的蓝带</b>，形状和你画的红线一致，宽 10–20 px。"
     "右：新默认——蓝带消失。这就是「光带」的来源：测度窗口横跨主体边缘时，"
     "边缘那个极强的响应赢下 argmax，于是轮廓外一个窗口半径内的墙被判成了主体的深度，"
     "从「墙是虚的」那些帧里取亮度。"),
    ("out/band_chart2.png", "02_亮带曲线_对照真值.png",
     "左：距轮廓 d 处的墙面亮度（4 px 一档）。右：各版本<b>减去真值</b>。"
     "蓝＝旧默认（贴着轮廓 +7.6 级，一直拖到 40 px），红＝上一交付版（减半），"
     "绿＝新默认（8 px 以外基本压在 0 上）。"),
    ("out/bv_c.png", "03_轮廓带_3倍_旧-前-新.png",
     "画布 x2520-3060, y1560-1900，3 倍，三列＝旧默认 / 上一版 / 新默认；上排原图，"
     "下排去掉慢变背景后 ×4。"),
    ("out/band_marked_new.png", "04_你标记的位置_画布坐标.png",
     "你画的红线叠在<b>新默认</b>上（画布 x2250-3170, y1470-2210）。"),
]


def preview_png(path, max_w=1700, quality=88):
    im = Image.open(path).convert("RGB")
    if im.size[0] > max_w:
        im = im.resize((max_w, int(round(im.size[1] * max_w / im.size[0]))), Image.LANCZOS)
    b = io.BytesIO()
    im.save(b, "JPEG", quality=quality)
    return base64.b64encode(b.getvalue()).decode("ascii")


CSS = """
body{font:15px/1.75 -apple-system,"Segoe UI","Microsoft YaHei",sans-serif;
     margin:0;padding:28px 32px;background:#f5f6f8;color:#1c1e21}
h1{font-size:22px;margin:0 0 6px}
p.lead{margin:0 0 22px;color:#555}
section{background:#fff;border:1px solid #e3e6ea;border-radius:10px;padding:16px 18px;margin:0 0 20px}
h2{font-size:16px;margin:0 0 8px}
.cap{margin:0 0 12px;color:#444}
img{max-width:100%;border:1px solid #dfe3e8;border-radius:6px;display:block}
code{background:#eef1f4;padding:1px 5px;border-radius:4px;font-size:13px}
table{border-collapse:collapse;margin:8px 0 4px;font-size:14px}
th,td{border:1px solid #dde2e8;padding:6px 10px;text-align:right}
th:first-child,td:first-child{text-align:left}
th{background:#f0f3f6;font-weight:600}
ul{margin:6px 0 0 18px;padding:0}
b{color:#b00}
"""


def main():
    os.makedirs(OUT, exist_ok=True)
    x0, y0, x1, y1 = CROP
    im = Image.open("out/band_marked_new.png").crop((x0, y0, x1, y1))
    im.save("out/band_marked_crop.png")

    parts = []
    for src, name, cap in ITEMS:
        if not os.path.exists(src):
            print("missing", src)
            continue
        shutil.copy2(src, os.path.join(OUT, name))
        parts.append('<section><h2>%s</h2><p class="cap">%s</p>'
                     '<img src="data:image/jpeg;base64,%s"></section>'
                     % (os.path.splitext(name)[0], cap, preview_png(src)))
        print("bundled", name)
    for src, name in [("out/可莉-全清晰.png", "05_成品_全分辨率.png"),
                      ("out/可莉-全清晰_预览.jpg", "06_成品预览_1600px.jpg")]:
        if os.path.exists(src):
            shutil.copy2(src, os.path.join(OUT, name))
            print("bundled", name)

    html = (
        '<!DOCTYPE html><html lang="zh-CN"><head><meta charset="utf-8">'
        "<title>光带（轮廓外亮带）修复与检查</title><style>%s</style></head><body>"
        "<h1>轮廓外的亮带：根因与修复</h1>"
        '<p class="lead">全部文件也在 <code>out/检查_亮带/</code>。'
        "结论：带内亮度从 <b>+2.81 级</b>（旧默认）降到 <b>+0.93 级</b>，"
        "这个指标的固有底噪是 +0.86，等于已经贴到地板；同一带的纹理比反而从 0.865 升到 0.933，"
        "gradE 略升。做法：只让金字塔最粗的 4 层用“窄候选窗口”，细节层保留宽窗口。</p>"
        "%s<section><h2>数字</h2>%s</section>"
        "<section><h2>两个开关，缺一不可</h2><ul>"
        "<li><code>--edge-aware 16</code>（新默认）：把焦点响应按图像本身加权聚合后再取 argmax，"
        "窗口不再横跨主体边缘 → <b>深度对了</b>（轮廓外 2–8 px 从偏近 5.0 帧变成 0.6 帧）。"
        "这是「光带」的根因修复</li>"
        "<li><code>--coarse-levels 4</code>（上一轮）：最粗 4 层用窄候选窗口，"
        "治「深度对但窗口太宽时仍会挑到邻面泛光」的另一半。"
        "实测<b>只开前者</b>是 +3.27 级，两个都开才是 +0.22</li>"
        "<li>代价：六个参照区域里五个变锐，平坦背景掉 0.9%%（28.08 → 27.84，AF 成品 28.29）</li>"
        "<li>回退：<code>--edge-aware 0 --coarse-levels 0</code> = 逐位回到最早的版本</li>"
        "</ul></section></body></html>" % (CSS, "".join(parts), NUMBERS)
    )
    with open(os.path.join(OUT, "index.html"), "w", encoding="utf-8") as f:
        f.write(html)
    print("wrote", os.path.join(OUT, "index.html"))


main()

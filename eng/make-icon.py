"""Generates the Model2Vec.Net package icon (original artwork).

A rounded-square gradient badge showing text (token lines) transforming into an
embedding vector grid. Represents static-embedding inference. Intentionally
distinct from any upstream logo.
"""
from PIL import Image, ImageDraw

S = 512  # supersample, downscaled at the end
img = Image.new("RGBA", (S, S), (0, 0, 0, 0))
draw = ImageDraw.Draw(img)

# Vertical gradient background: teal -> indigo
top = (14, 165, 165)     # #0EA5A5
bot = (79, 70, 229)      # #4F46E5
for y in range(S):
    t = y / (S - 1)
    r = round(top[0] + (bot[0] - top[0]) * t)
    g = round(top[1] + (bot[1] - top[1]) * t)
    b = round(top[2] + (bot[2] - top[2]) * t)
    draw.line([(0, y), (S, y)], fill=(r, g, b, 255))

# Rounded-square mask
radius = int(S * 0.22)
mask = Image.new("L", (S, S), 0)
ImageDraw.Draw(mask).rounded_rectangle([0, 0, S - 1, S - 1], radius=radius, fill=255)
img.putalpha(mask)

draw = ImageDraw.Draw(img)

white = (255, 255, 255, 255)
accent = (245, 213, 71, 255)  # amber highlight cell

# Left: token / text lines (decreasing length) representing input text
lx = int(S * 0.16)
lh = int(S * 0.05)
gap = int(S * 0.105)
ly = int(S * 0.28)
for frac in (0.30, 0.24, 0.30):
    draw.rounded_rectangle([lx, ly, lx + int(S * frac), ly + lh],
                           radius=lh // 2, fill=white)
    ly += gap

# Arrow: text -> vector
ax0 = int(S * 0.49)
ay = int(S * 0.50)
ax1 = int(S * 0.585)
aw = max(3, int(S * 0.022))
draw.line([(ax0, ay), (ax1, ay)], fill=white, width=aw)
ah = int(S * 0.035)
draw.polygon([(ax1 + int(S * 0.02), ay), (ax1 - ah, ay - ah), (ax1 - ah, ay + ah)], fill=white)

# Right: embedding vector as a 2x3 grid of cells
cell = int(S * 0.105)
cgap = int(S * 0.028)
gx = int(S * 0.62)
gy0 = int(S * 0.27)
cr = int(S * 0.022)
filled = {(0, 1), (1, 0), (1, 2)}  # which cells get the accent
for row in range(3):
    for col in range(2):
        x0 = gx + col * (cell + cgap)
        y0 = gy0 + row * (cell + cgap)
        fill = accent if (row, col) in filled else white
        draw.rounded_rectangle([x0, y0, x0 + cell, y0 + cell], radius=cr, fill=fill)

out = img.resize((128, 128), Image.LANCZOS)
out.save(r"C:\src\ericstj\Model2Vec.Net\eng\icon.png", "PNG")
print("wrote eng/icon.png")

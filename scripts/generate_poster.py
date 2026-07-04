from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter, ImageFont


ROOT = Path(__file__).resolve().parents[1]
LOGO = ROOT / "logo.png"
OUT = ROOT / "assets" / "poster.png"


def load_font(name: str, size: int) -> ImageFont.FreeTypeFont:
    font_path = Path("/System/Library/Fonts/Supplemental") / name
    return ImageFont.truetype(str(font_path), size=size)


def blend(c1: tuple[int, int, int], c2: tuple[int, int, int], t: float) -> tuple[int, int, int]:
    return tuple(int(c1[i] * (1 - t) + c2[i] * t) for i in range(3))


def gradient_background(width: int, height: int) -> Image.Image:
    img = Image.new("RGB", (width, height))
    px = img.load()
    top = (7, 12, 24)
    bottom = (16, 15, 32)
    for y in range(height):
        for x in range(width):
            tx = x / max(width - 1, 1)
            ty = y / max(height - 1, 1)
            row = blend(top, bottom, ty)
            glow = blend((8, 40, 56), (35, 20, 64), tx * 0.7 + (1 - ty) * 0.3)
            px[x, y] = blend(row, glow, 0.24)
    return img.convert("RGBA")


def add_glow(canvas: Image.Image, center: tuple[int, int], radius: int, color: tuple[int, int, int, int], blur: int) -> None:
    layer = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(layer)
    x, y = center
    draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=color)
    canvas.alpha_composite(layer.filter(ImageFilter.GaussianBlur(blur)))


def main() -> None:
    width, height = 1600, 560
    bg = gradient_background(width, height)
    draw = ImageDraw.Draw(bg)

    add_glow(bg, (1150, 255), 185, (44, 255, 224, 72), 115)
    add_glow(bg, (1325, 270), 155, (173, 115, 255, 64), 110)
    add_glow(bg, (250, 520), 180, (30, 160, 182, 22), 120)

    title_font = load_font("Arial Bold.ttf", 82)
    subtitle_font = load_font("Georgia.ttf", 28)
    meta_font = load_font("Arial.ttf", 19)

    left = 110
    top = 108

    draw.text((left, top), "Localisation", font=title_font, fill=(242, 248, 255, 255))
    draw.multiline_text(
        (left, top + 105),
        "Localised receptive fields for oscillatory and PDE systems,\nwith extensions to high-dimensional physics solvers",
        font=subtitle_font,
        fill=(184, 231, 238, 255),
        spacing=8,
    )
    draw.text(
        (left, top + 205),
        "Gaussian gating  •  Heat equation  •  4D targeted sampling",
        font=meta_font,
        fill=(143, 221, 227, 255),
    )

    line_y = top + 248
    draw.line((left, line_y, 700, line_y), fill=(120, 232, 235, 105), width=3)

    logo = Image.open(LOGO).convert("RGBA")
    logo = logo.resize((500, 500), Image.Resampling.LANCZOS)

    logo_layer = Image.new("RGBA", bg.size, (0, 0, 0, 0))
    logo_layer.alpha_composite(logo, dest=(990, 28))
    soft_shadow = logo_layer.filter(ImageFilter.GaussianBlur(28))
    bg.alpha_composite(soft_shadow)
    bg.alpha_composite(logo_layer)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    bg.convert("RGB").save(OUT, quality=96)


if __name__ == "__main__":
    main()

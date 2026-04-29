"""
One-time image generator for product support chatbot test fixtures.

Run from repo root:
    python tests/fixtures/make_product_images.py

Outputs PNG files into tests/fixtures/images/.
Commit the outputs — tests load them as static assets.
"""
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

OUT = Path(__file__).parent / "images"
OUT.mkdir(exist_ok=True)

W, H = 800, 600

FALLBACK = ImageFont.load_default()


def _font(size: int):
    for name in ("arial.ttf", "Arial.ttf", "DejaVuSans.ttf", "LiberationSans-Regular.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return FALLBACK


def make_product_no_damage(filename: str) -> None:
    """A smartphone in good condition — no visible damage."""
    img = Image.new("RGB", (W, H), "#F0F0F0")
    draw = ImageDraw.Draw(img)
    # Phone body
    draw.rounded_rectangle([(250, 60), (550, 530)], radius=30, outline="#333333", width=4, fill="#2C2C2C")
    # Screen (undamaged, uniform dark surface)
    draw.rectangle([(272, 105), (528, 465)], fill="#1A1A2E")
    # Subtle screen glare — no cracks
    draw.line([(282, 115), (330, 165)], fill="#FFFFFF", width=2)
    # Camera cutout
    draw.ellipse([(286, 80), (308, 102)], fill="#444444", outline="#666666")
    # Home button area
    draw.ellipse([(372, 475), (428, 515)], outline="#555555", width=2)
    draw.text((285, 545), "No visible damage", fill="#444444", font=_font(14))
    img.save(OUT / filename)
    print(f"  {filename}  (product in good condition)")


def make_product_impact_damage(filename: str) -> None:
    """A smartphone with a cracked screen — clear impact damage."""
    img = Image.new("RGB", (W, H), "#F0F0F0")
    draw = ImageDraw.Draw(img)
    # Phone body
    draw.rounded_rectangle([(250, 60), (550, 530)], radius=30, outline="#333333", width=4, fill="#2C2C2C")
    # Screen base
    draw.rectangle([(272, 105), (528, 465)], fill="#1A1A2E")
    # Crack pattern radiating from impact point
    ix, iy = 380, 250
    spokes = [
        (ix, iy, ix - 85, iy - 110),
        (ix, iy, ix + 95, iy - 65),
        (ix, iy, ix - 65, iy + 120),
        (ix, iy, ix + 105, iy + 85),
        (ix, iy, ix - 115, iy + 35),
        (ix, iy, ix + 25, iy - 140),
        (ix, iy, ix + 80, iy + 130),
        (ix, iy, ix - 130, iy - 40),
    ]
    for x1, y1, x2, y2 in spokes:
        draw.line([(x1, y1), (x2, y2)], fill="#DDDDDD", width=2)
        mx, my = (x1 + x2) // 2, (y1 + y2) // 2
        draw.line([(mx, my), (mx + 28, my - 22)], fill="#AAAAAA", width=1)
    # Dark impact point
    draw.ellipse([(ix - 9, iy - 9), (ix + 9, iy + 9)], fill="#333333")
    # Camera
    draw.ellipse([(286, 80), (308, 102)], fill="#444444", outline="#666666")
    # Home button
    draw.ellipse([(372, 475), (428, 515)], outline="#555555", width=2)
    draw.text((265, 545), "Cracked screen — impact damage", fill="#444444", font=_font(14))
    img.save(OUT / filename)
    print(f"  {filename}  (product with clear cracked-screen damage)")


def make_product_blurry(filename: str) -> None:
    """An out-of-focus product image — damage state cannot be determined."""
    from PIL.ImageFilter import GaussianBlur
    img = Image.new("RGB", (W, H), "#E8E8E8")
    draw = ImageDraw.Draw(img)
    draw.rounded_rectangle([(250, 60), (550, 530)], radius=30, outline="#555555", width=6, fill="#3C3C3C")
    draw.rectangle([(272, 105), (528, 465)], fill="#2A2A4E")
    draw.ellipse([(372, 475), (428, 515)], outline="#666666", width=3)
    img = img.filter(GaussianBlur(radius=9))
    img.save(OUT / filename)
    print(f"  {filename}  (blurry image — damage state unclear)")


def make_product_normal_wear(filename: str) -> None:
    """A product with minor cosmetic surface scratches — normal use wear."""
    img = Image.new("RGB", (W, H), "#F0F0F0")
    draw = ImageDraw.Draw(img)
    draw.rounded_rectangle([(250, 60), (550, 530)], radius=30, outline="#333333", width=4, fill="#2C2C2C")
    draw.rectangle([(272, 105), (528, 465)], fill="#1A1A2E")
    # Light surface scratches on the body (cosmetic, not structural)
    for i in range(6):
        sx = 268 + i * 12
        draw.line([(sx, 200 + i * 8), (sx + 38, 218 + i * 8)], fill="#555555", width=1)
    draw.ellipse([(286, 80), (308, 102)], fill="#444444", outline="#666666")
    draw.ellipse([(372, 475), (428, 515)], outline="#555555", width=2)
    draw.text((252, 545), "Minor surface scratches — normal wear", fill="#444444", font=_font(13))
    img.save(OUT / filename)
    print(f"  {filename}  (normal cosmetic wear, no manufacturing defect)")


if __name__ == "__main__":
    print(f"Writing product support images to {OUT}/\n")
    make_product_no_damage("product_no_damage.png")
    make_product_impact_damage("product_impact_damage.png")
    make_product_blurry("product_blurry.png")
    make_product_normal_wear("product_normal_wear.png")
    print(f"\nDone — {len(list(OUT.glob('product_*.png')))} images written.")

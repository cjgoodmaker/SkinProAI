"""
Overlay Tool - Generates visual markers for biopsy sites and excision margins
"""

import io
import tempfile
from typing import Tuple, Optional, Dict, Any
from PIL import Image, ImageDraw, ImageFont


class OverlayTool:
    """
    Generates image overlays for clinical decision visualization:
    - Biopsy site markers (circles)
    - Excision margins (dashed outlines with margin indicators)
    """

    # Colors for different marker types
    COLORS = {
        'biopsy': (255, 69, 0, 200),      # Orange-red with alpha
        'excision': (220, 20, 60, 200),   # Crimson with alpha
        'margin': (255, 215, 0, 180),     # Gold for margin line
        'text': (255, 255, 255, 255),     # White text
        'text_bg': (0, 0, 0, 180),        # Semi-transparent black bg
    }

    def __init__(self):
        self.loaded = True

    def generate_biopsy_overlay(
        self,
        image: Image.Image,
        center_x: float,
        center_y: float,
        radius: float = 0.05,
        label: str = "Biopsy Site"
    ) -> Dict[str, Any]:
        """
        Generate biopsy site overlay with circle marker.

        Args:
            image: PIL Image
            center_x: X coordinate as fraction (0-1) of image width
            center_y: Y coordinate as fraction (0-1) of image height
            radius: Radius as fraction of image width
            label: Text label for the marker

        Returns:
            Dict with overlay image and metadata
        """
        # Convert to RGBA for transparency
        img = image.convert("RGBA")
        width, height = img.size

        # Create overlay layer
        overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)

        # Calculate pixel coordinates
        cx = int(center_x * width)
        cy = int(center_y * height)
        r = int(radius * width)

        # Draw outer circle (thicker)
        for offset in range(3):
            draw.ellipse(
                [cx - r - offset, cy - r - offset, cx + r + offset, cy + r + offset],
                outline=self.COLORS['biopsy'],
                width=2
            )

        # Draw crosshairs
        line_len = r // 2
        draw.line([(cx - line_len, cy), (cx + line_len, cy)],
                  fill=self.COLORS['biopsy'], width=2)
        draw.line([(cx, cy - line_len), (cx, cy + line_len)],
                  fill=self.COLORS['biopsy'], width=2)

        # Draw label with background
        try:
            font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 14)
        except:
            font = ImageFont.load_default()

        text_bbox = draw.textbbox((0, 0), label, font=font)
        text_width = text_bbox[2] - text_bbox[0]
        text_height = text_bbox[3] - text_bbox[1]

        text_x = cx - text_width // 2
        text_y = cy + r + 10

        # Background rectangle for text
        padding = 4
        draw.rectangle(
            [text_x - padding, text_y - padding,
             text_x + text_width + padding, text_y + text_height + padding],
            fill=self.COLORS['text_bg']
        )
        draw.text((text_x, text_y), label, fill=self.COLORS['text'], font=font)

        # Composite
        result = Image.alpha_composite(img, overlay)

        # Save to temp file
        temp_file = tempfile.NamedTemporaryFile(suffix="_biopsy_overlay.png", delete=False)
        result.save(temp_file.name, "PNG")
        temp_file.close()

        return {
            "overlay": result,
            "path": temp_file.name,
            "type": "biopsy",
            "coordinates": {
                "center_x": center_x,
                "center_y": center_y,
                "radius": radius
            },
            "label": label
        }

    def generate_excision_overlay(
        self,
        image: Image.Image,
        center_x: float,
        center_y: float,
        lesion_radius: float,
        margin_mm: int = 5,
        pixels_per_mm: float = 10.0,
        label: str = "Excision Margin"
    ) -> Dict[str, Any]:
        """
        Generate excision margin overlay with inner (lesion) and outer (margin) boundaries.

        Args:
            image: PIL Image
            center_x: X coordinate as fraction (0-1)
            center_y: Y coordinate as fraction (0-1)
            lesion_radius: Lesion radius as fraction of image width
            margin_mm: Excision margin in millimeters
            pixels_per_mm: Estimated pixels per mm (for margin calculation)
            label: Text label

        Returns:
            Dict with overlay image and metadata
        """
        img = image.convert("RGBA")
        width, height = img.size

        overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)

        # Calculate coordinates
        cx = int(center_x * width)
        cy = int(center_y * height)
        inner_r = int(lesion_radius * width)

        # Calculate margin in pixels
        margin_px = int(margin_mm * pixels_per_mm)
        outer_r = inner_r + margin_px

        # Draw outer margin (dashed effect using multiple arcs)
        dash_length = 10
        for angle in range(0, 360, dash_length * 2):
            draw.arc(
                [cx - outer_r, cy - outer_r, cx + outer_r, cy + outer_r],
                start=angle,
                end=angle + dash_length,
                fill=self.COLORS['margin'],
                width=3
            )

        # Draw inner lesion boundary (solid)
        draw.ellipse(
            [cx - inner_r, cy - inner_r, cx + inner_r, cy + inner_r],
            outline=self.COLORS['excision'],
            width=2
        )

        # Draw margin indicator lines (radial)
        for angle in [0, 90, 180, 270]:
            import math
            rad = math.radians(angle)
            inner_x = cx + int(inner_r * math.cos(rad))
            inner_y = cy + int(inner_r * math.sin(rad))
            outer_x = cx + int(outer_r * math.cos(rad))
            outer_y = cy + int(outer_r * math.sin(rad))
            draw.line([(inner_x, inner_y), (outer_x, outer_y)],
                      fill=self.COLORS['margin'], width=2)

        # Draw labels
        try:
            font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 12)
            font_small = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 10)
        except:
            font = ImageFont.load_default()
            font_small = font

        # Main label
        text_bbox = draw.textbbox((0, 0), label, font=font)
        text_width = text_bbox[2] - text_bbox[0]
        text_height = text_bbox[3] - text_bbox[1]

        text_x = cx - text_width // 2
        text_y = cy + outer_r + 15

        padding = 4
        draw.rectangle(
            [text_x - padding, text_y - padding,
             text_x + text_width + padding, text_y + text_height + padding],
            fill=self.COLORS['text_bg']
        )
        draw.text((text_x, text_y), label, fill=self.COLORS['text'], font=font)

        # Margin measurement label
        margin_label = f"{margin_mm}mm margin"
        margin_bbox = draw.textbbox((0, 0), margin_label, font=font_small)
        margin_width = margin_bbox[2] - margin_bbox[0]

        margin_text_x = cx + outer_r + 5
        margin_text_y = cy - 6

        draw.rectangle(
            [margin_text_x - 2, margin_text_y - 2,
             margin_text_x + margin_width + 2, margin_text_y + 12],
            fill=self.COLORS['text_bg']
        )
        draw.text((margin_text_x, margin_text_y), margin_label,
                  fill=self.COLORS['margin'], font=font_small)

        # Composite
        result = Image.alpha_composite(img, overlay)

        temp_file = tempfile.NamedTemporaryFile(suffix="_excision_overlay.png", delete=False)
        result.save(temp_file.name, "PNG")
        temp_file.close()

        return {
            "overlay": result,
            "path": temp_file.name,
            "type": "excision",
            "coordinates": {
                "center_x": center_x,
                "center_y": center_y,
                "lesion_radius": lesion_radius,
                "margin_mm": margin_mm,
                "total_radius": outer_r / width
            },
            "label": label
        }

    def generate_comparison_overlay(
        self,
        image1: Image.Image,
        image2: Image.Image,
        label1: str = "Previous",
        label2: str = "Current"
    ) -> Dict[str, Any]:
        """
        Generate side-by-side comparison of two images for follow-up.

        Args:
            image1: First (previous) image
            image2: Second (current) image
            label1: Label for first image
            label2: Label for second image

        Returns:
            Dict with comparison image and metadata
        """
        # Resize to same height
        max_height = 400

        # Calculate sizes maintaining aspect ratio
        w1, h1 = image1.size
        w2, h2 = image2.size

        ratio1 = max_height / h1
        ratio2 = max_height / h2

        new_w1 = int(w1 * ratio1)
        new_w2 = int(w2 * ratio2)

        img1 = image1.resize((new_w1, max_height), Image.Resampling.LANCZOS)
        img2 = image2.resize((new_w2, max_height), Image.Resampling.LANCZOS)

        # Create comparison canvas
        gap = 20
        total_width = new_w1 + gap + new_w2
        header_height = 30
        total_height = max_height + header_height

        canvas = Image.new("RGB", (total_width, total_height), (255, 255, 255))
        draw = ImageDraw.Draw(canvas)

        # Draw labels
        try:
            font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 14)
        except:
            font = ImageFont.load_default()

        # Previous label
        draw.rectangle([0, 0, new_w1, header_height], fill=(70, 130, 180))
        bbox1 = draw.textbbox((0, 0), label1, font=font)
        text_w1 = bbox1[2] - bbox1[0]
        draw.text(((new_w1 - text_w1) // 2, 8), label1, fill=(255, 255, 255), font=font)

        # Current label
        draw.rectangle([new_w1 + gap, 0, total_width, header_height], fill=(60, 179, 113))
        bbox2 = draw.textbbox((0, 0), label2, font=font)
        text_w2 = bbox2[2] - bbox2[0]
        draw.text((new_w1 + gap + (new_w2 - text_w2) // 2, 8), label2,
                  fill=(255, 255, 255), font=font)

        # Paste images
        canvas.paste(img1, (0, header_height))
        canvas.paste(img2, (new_w1 + gap, header_height))

        # Draw divider
        draw.line([(new_w1 + gap // 2, header_height), (new_w1 + gap // 2, total_height)],
                  fill=(200, 200, 200), width=2)

        temp_file = tempfile.NamedTemporaryFile(suffix="_comparison.png", delete=False)
        canvas.save(temp_file.name, "PNG")
        temp_file.close()

        return {
            "comparison": canvas,
            "path": temp_file.name,
            "type": "comparison"
        }


def get_overlay_tool() -> OverlayTool:
    """Get overlay tool instance"""
    return OverlayTool()

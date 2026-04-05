# -*- coding: utf-8 -*-
"""
Handles all PDF generation and formatting for Mudra-Fusion.
Separates PDF operations from Flask app logic.
"""

import os
from fpdf import FPDF


class StoryPDF(FPDF):
    """Custom FPDF class for Mudra Story formatting with headers and footers."""
    
    def header(self):
        """Add header to all pages except the first."""
        if self.page_no() == 1:
            return
        self.set_font("Arial", "I", 9)
        self.set_text_color(120, 120, 120)
        self.cell(0, 8, "Mudra Story Interpretation", 0, 1, "R")
        self.ln(2)

    def footer(self):
        """Add footer with page number to all pages."""
        self.set_y(-12)
        self.set_font("Arial", "I", 9)
        self.set_text_color(120, 120, 120)
        self.cell(0, 8, f"Page {self.page_no()}", 0, 0, "C")


# ===================== TEXT NORMALIZATION =====================

def normalize_story_text(text):
    """
    Clean up garbled Unicode characters from story text.
    Handles common mojibake patterns from encoding/decoding errors.
    
    Args:
        text: Story text with potential encoding issues
    
    Returns:
        Normalized text with special characters resolved
    """
    # Using unicode escape sequences to avoid encoding issues
    replacements = {
        u"\uf0df\u00a9\u00b0": "",       # Dancer emoji garbled
        u"\uf0df\u201d\u00b9": "",       # Red circle emoji garbled
        u"\uf0df\u201d\u009c": "",       # Bookmark emoji garbled
        u"\uf0df\u201d\u00a4": "",       # Think face emoji garbled
        u"\uf0df\u201d\u02dc": "",       # Smile emoji garbled
        u"\uf0df\u00a7\u00a0": "",       # Brain emoji garbled
        u"\uf0df\u008e\u00ad": "",       # Performing arts emoji garbled
        u"\uf0df\u201d": "",             # Magnifying glass emoji garbled
        u"\u00e2\u009c\u00a8": "",       # Sparkles emoji garbled
        u"\u00e2\u0152": "No result:",   # Cross mark emoji
        u"\u00e2\u0086\u0091": " -> ",   # Arrow emoji
        u"\u00e2\u0080\u0093": "-",      # Em dash
        u"\u00e2\u0080\u0094": "-",      # En dash
        u"\u00e2\u0080\u0099": "'",      # Apostrophe
        u"\u00e2\u0080\u009c": '"',      # Left quote
        u"\u00e2\u0080\u009d": '"',      # Right quote
    }
    normalized = text
    for old, new in replacements.items():
        try:
            normalized = normalized.replace(old, new)
        except:
            pass
    return normalized


# ===================== PDF RENDERING HELPERS =====================

def render_section_heading(pdf, title):
    """
    Render a formatted section heading in the PDF.
    
    Args:
        pdf: FPDF instance
        title: Heading text
    """
    pdf.ln(2)
    pdf.set_font("Arial", "B", 13)
    pdf.set_text_color(22, 57, 92)
    pdf.set_fill_color(235, 242, 250)
    pdf.cell(0, 9, title, 0, 1, "L", True)
    pdf.ln(1)


def render_label_value(pdf, label, value):
    """
    Render a key-value pair with bold label and normal value.
    
    Args:
        pdf: FPDF instance
        label: Label text (will be bold)
        value: Value text (will be normal weight)
    """
    pdf.set_font("Arial", "B", 11)
    pdf.set_text_color(45, 45, 45)
    pdf.cell(30, 7, f"{label}:")
    pdf.set_font("Arial", "", 11)
    pdf.set_text_color(20, 20, 20)
    pdf.multi_cell(0, 7, value)


def render_body_paragraph(pdf, text):
    """
    Render a paragraph of body text.
    
    Args:
        pdf: FPDF instance
        text: Paragraph text
    """
    pdf.set_font("Arial", "", 11)
    pdf.set_text_color(30, 30, 30)
    pdf.multi_cell(0, 7, text)
    pdf.ln(1)


# ===================== MAIN PDF GENERATION =====================

def generate_pdf_from_story(story_text, output_path="static/output_story.pdf"):
    """
    Generate a formatted PDF from story text.
    
    Args:
        story_text: Full story text from run_story_engine()
        output_path: Where to save the PDF file (default: static/output_story.pdf)
    
    Returns:
        Path to the generated PDF file
    """
    # Create static folder if it doesn't exist
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Initialize PDF
    pdf = StoryPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.set_margins(18, 18, 18)
    pdf.add_page()

    # --- Title Section ---
    pdf.set_fill_color(230, 238, 247)
    pdf.rect(10, 10, 190, 22, "F")
    pdf.set_font("Arial", "B", 20)
    pdf.set_text_color(22, 57, 92)
    pdf.cell(0, 12, "Mudra Story Interpretation", 0, 1, "C")
    pdf.set_font("Arial", "I", 10)
    pdf.set_text_color(95, 95, 95)
    pdf.cell(0, 6, "A formatted narrative report from the detected mudra sequence", 0, 1, "C")
    pdf.ln(6)

    # --- Parse and render story content ---
    clean_text = normalize_story_text(story_text).encode("latin-1", "ignore").decode("latin-1")
    lines = [line.strip() for line in clean_text.split("\n") if line.strip()]

    # Map emoji/text headers to PDF section headings
    heading_map = {
        "Mudra Sequence:": "Mudra Sequence",
        "Matched Verse:": "Matched Verse",
        "Translation:": "Translation",
        "Commentary Summary:": "Commentary Summary",
        "Theme:": "Theme",
        "Interpretation (AI-generated):": "Interpretation",
        "Interpretation (Template-based narrative):": "Interpretation",
        "Match Analysis:": "Match Analysis",
    }

    skipped_sections = {"Sanskrit:", "Transliteration:"}
    skip_content = False

    for line in lines:
        # Skip main title (already rendered at top)
        if line == "MUDRA-BASED STORY INTERPRETATION":
            continue

        if line in skipped_sections:
            skip_content = True
            continue

        # Render section headings
        if line in heading_map:
            skip_content = False
            render_section_heading(pdf, heading_map[line])
            continue

        if skip_content:
            continue

        # Render special fields with formatting
        if line.startswith("Source :"):
            render_label_value(pdf, "Source", line.split(":", 1)[1].strip())
            continue

        if line.startswith("Speaker:"):
            render_label_value(pdf, "Speaker", line.split(":", 1)[1].strip())
            continue

        if line.startswith("Match score"):
            render_label_value(pdf, "Match score", line.split(":", 1)[1].strip())
            continue

        if line.startswith("Keywords matched"):
            render_label_value(pdf, "Keywords matched", line.split(":", 1)[1].strip())
            continue

        if line.startswith("Emotions matched"):
            render_label_value(pdf, "Emotions matched", line.split(":", 1)[1].strip())
            continue

        # Render regular body paragraphs
        render_body_paragraph(pdf, line)

    # Save PDF
    pdf.output(output_path)
    return output_path


def get_pdf_static_path():
    """Get the static path for the output PDF file."""
    return "output_story.pdf"

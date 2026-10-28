"""Generate a PDF API reference from the package docstrings."""

import inspect
from importlib.metadata import version
import re
from pathlib import Path
from xml.sax.saxutils import escape

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Preformatted,
    HRFlowable,
)

from minkowski_correlations import minkowski_functionals


ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "docs" / "minkowski-correlations-documentation.pdf"

BLUE = colors.HexColor("#17365D")
GRAY = colors.HexColor("#526071")


def parse_docstring(docstring):
    """Split a NumPy-style docstring into named sections."""
    lines = inspect.cleandoc(docstring).splitlines()
    sections = {"Description": []}
    current = "Description"

    for i, line in enumerate(lines):
        if (
            i + 1 < len(lines)
            and re.fullmatch(r"-{3,}", lines[i + 1].strip())
        ):
            current = line.strip()
            sections[current] = []
            continue

        if i > 0 and re.fullmatch(r"-{3,}", line.strip()):
            continue

        sections[current].append(line)

    return sections


def add_section(story, heading, lines, styles):
    """Add a docstring section while preserving its text."""
    story.append(Spacer(1, 0.3 * cm))
    story.append(Paragraph(escape(heading), styles["SectionTitle"]))

    text = "\n".join(lines).strip()
    if not text:
        return

    if heading == "Description":
        paragraphs = re.split(r"\n\s*\n", text)
        for paragraph in paragraphs:
            story.append(
                Paragraph(
                    escape(" ".join(paragraph.split())),
                    styles["BodyCustom"],
                )
            )
        return

    # Parameter-like entries have an unindented name/type line
    # followed by indented description lines.
    if heading in {"Parameters", "Returns", "Raises"}:
        entries = []
        name = None
        description = []

        for line in lines:
            if not line.strip():
                continue

            if not line[0].isspace():
                if name is not None:
                    entries.append((name, description))
                name = line.strip()
                description = []
            else:
                description.append(line.strip())

        if name is not None:
            entries.append((name, description))

        for name, description in entries:
            story.append(
                Paragraph(escape(name), styles["EntryTitle"])
            )
            story.append(
                Paragraph(
                    escape(" ".join(description)),
                    styles["EntryBody"],
                )
            )
        return

    for paragraph in re.split(r"\n\s*\n", text):
        story.append(
            Paragraph(
                escape(" ".join(paragraph.split())),
                styles["BodyCustom"],
            )
        )


def add_page_number(canvas, document):
    canvas.saveState()
    canvas.setFont("Helvetica", 8)
    canvas.setFillColor(GRAY)
    canvas.drawString(
        2.2 * cm,
        1.4 * cm,
        "minkowski-correlations | API Reference",
    )
    canvas.drawRightString(
        19 * cm,
        1.4 * cm,
        str(document.page),
    )
    canvas.restoreState()


def main():
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)

    styles = getSampleStyleSheet()

    styles.add(ParagraphStyle(
        name="DocumentTitle",
        fontName="Helvetica-Bold",
        fontSize=22,
        leading=28,
        textColor=BLUE,
        spaceAfter=12,
        alignment=TA_CENTER,
    ))

    styles.add(ParagraphStyle(
        name="SubtitleCustom",
        fontName="Helvetica",
        fontSize=11,
        leading=16,
        textColor=GRAY,
        alignment=TA_CENTER,
        spaceAfter=20,
    ))

    styles.add(ParagraphStyle(
        name="SectionTitle",
        fontName="Helvetica-Bold",
        fontSize=13,
        leading=17,
        textColor=BLUE,
        spaceBefore=10,
        spaceAfter=10,
        keepWithNext=True,
    ))

    styles.add(ParagraphStyle(
        name="BodyCustom",
        fontName="Helvetica",
        fontSize=9.5,
        leading=14.5,
        spaceAfter=9,
    ))

    styles.add(ParagraphStyle(
        name="EntryTitle",
        fontName="Helvetica-Bold",
        fontSize=9.5,
        leading=13,
        textColor=BLUE,
        spaceBefore=8,
        spaceAfter=3,
        keepWithNext=True,
    ))

    styles.add(ParagraphStyle(
        name="EntryBody",
        parent=styles["BodyCustom"],
        leftIndent=12,
        spaceAfter=7,
    ))

    document = SimpleDocTemplate(
        str(OUTPUT),
        pagesize=(21 * cm, 29.7 * cm),
        rightMargin=2.2 * cm,
        leftMargin=2.2 * cm,
        topMargin=2.3 * cm,
        bottomMargin=2.3 * cm,
        title="minkowski-correlations — API Reference",
        author="minkowski-correlations",
    )

    story = []

    story.append(Paragraph(
        "minkowski-correlations",
        styles["DocumentTitle"],
    ))
    story.append(Paragraph(
        f"API Reference | Version {version('minkowski-correlations')}",
        styles["SubtitleCustom"],
    ))
    story.append(HRFlowable(
        width="100%",
        thickness=1,
        color=BLUE,
        spaceAfter=18,
    ))

    story.append(Paragraph(
        "minkowski_functionals()",
        styles["SectionTitle"],
    ))

    signature = inspect.signature(minkowski_functionals)
    signature_lines = [
        "minkowski_functionals(",
    ]

    for parameter in signature.parameters.values():
        signature_lines.append(f"    {parameter},")

    signature_lines.append(")")

    story.append(Preformatted(
        "\n".join(signature_lines),
        ParagraphStyle(
            name="Signature",
            fontName="Courier",
            fontSize=7.5,
            leading=11,
            textColor=colors.HexColor("#263746"),
            leftIndent=8,
            spaceAfter=12,
        ),
    ))

    docstring = inspect.getdoc(minkowski_functionals)
    if not docstring:
        raise RuntimeError("No docstring found for minkowski_functionals.")

    sections = parse_docstring(docstring)

    for heading, lines in sections.items():
        add_section(story, heading, lines, styles)

    document.build(
        story,
        onFirstPage=add_page_number,
        onLaterPages=add_page_number,
    )

    print(f"PDF created: {OUTPUT}")


if __name__ == "__main__":
    main()

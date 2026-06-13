"""
Generate a small sample notes PDF (pure stdlib, no dependencies) so the chatbot
can be tested without supplying your own document.

    python make_sample_pdf.py   ->   sample/notes.pdf
"""

from pathlib import Path

# Each inner list is one page's worth of lines.
PAGES = [
    [
        "Photosynthesis - Study Notes",
        "",
        "Photosynthesis is the process by which green plants, algae and some",
        "bacteria convert light energy into chemical energy stored as glucose.",
        "",
        "The overall equation is:",
        "  6 CO2 + 6 H2O + light energy -> C6H12O6 + 6 O2",
        "",
        "It occurs mainly in the leaves, inside organelles called chloroplasts.",
        "Chloroplasts contain the green pigment chlorophyll, which absorbs",
        "mostly red and blue light and reflects green light.",
    ],
    [
        "Two Stages of Photosynthesis",
        "",
        "1. Light-dependent reactions:",
        "   - Take place in the thylakoid membranes.",
        "   - Water is split (photolysis), releasing oxygen as a by-product.",
        "   - Produce ATP and NADPH.",
        "",
        "2. Light-independent reactions (the Calvin cycle):",
        "   - Take place in the stroma of the chloroplast.",
        "   - Use ATP and NADPH to fix carbon dioxide into glucose.",
        "   - This stage does not directly require light.",
        "",
        "Factors that affect the rate of photosynthesis include light",
        "intensity, carbon dioxide concentration, and temperature.",
    ],
]


def escape(text: str) -> str:
    return text.replace("\\", r"\\").replace("(", r"\(").replace(")", r"\)")


def make_content_stream(lines: list[str]) -> bytes:
    parts = ["BT", "/F1 12 Tf", "14 TL", "72 720 Td"]
    for line in lines:
        parts.append(f"({escape(line)}) Tj")
        parts.append("T*")
    parts.append("ET")
    return ("\n".join(parts)).encode("latin-1")


def build_pdf(pages: list[list[str]]) -> bytes:
    objects: list[bytes] = []

    # 1: Catalog, 2: Pages tree, 3: Font. Page + content objects follow.
    font_obj = b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>"

    page_obj_ids = []
    content_objs = []
    n = 4  # next free object id after catalog(1), pages(2), font(3)
    for lines in pages:
        content = make_content_stream(lines)
        content_id = n
        page_id = n + 1
        content_objs.append(
            (content_id,
             b"<< /Length %d >>\nstream\n%s\nendstream" % (len(content), content))
        )
        page_obj_ids.append(page_id)
        n += 2

    kids = " ".join(f"{pid} 0 R" for pid in page_obj_ids)
    catalog = b"<< /Type /Catalog /Pages 2 0 R >>"
    pages_tree = (
        b"<< /Type /Pages /Count %d /Kids [%s] >>"
        % (len(page_obj_ids), kids.encode())
    )

    # assemble all objects in id order
    by_id: dict[int, bytes] = {1: catalog, 2: pages_tree, 3: font_obj}
    for cid, body in content_objs:
        by_id[cid] = body
    for i, pid in enumerate(page_obj_ids):
        content_id = pid - 1
        by_id[pid] = (
            b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
            b"/Resources << /Font << /F1 3 0 R >> >> /Contents %d 0 R >>"
            % content_id
        )

    out = bytearray(b"%PDF-1.4\n")
    offsets = {}
    for oid in sorted(by_id):
        offsets[oid] = len(out)
        out += b"%d 0 obj\n%s\nendobj\n" % (oid, by_id[oid])

    xref_pos = len(out)
    count = len(by_id) + 1
    out += b"xref\n0 %d\n" % count
    out += b"0000000000 65535 f \n"
    for oid in range(1, count):
        out += b"%010d 00000 n \n" % offsets[oid]
    out += b"trailer\n<< /Size %d /Root 1 0 R >>\nstartxref\n%d\n%%%%EOF" % (
        count, xref_pos)
    return bytes(out)


if __name__ == "__main__":
    Path("sample").mkdir(exist_ok=True)
    path = Path("sample/notes.pdf")
    path.write_bytes(build_pdf(PAGES))
    print(f"Wrote {path} ({path.stat().st_size} bytes)")

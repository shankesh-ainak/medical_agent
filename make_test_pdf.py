"""Generate a SYNTHETIC test medical record that deliberately exercises every
graded requirement of dscribe. All data is fictional.

What each page plants (so you can verify the agent's behaviour):

  Page 1  Admission Note (typed -> text-layer, doc_type=admission_note)
            - demographics, admission date, allergies (penicillin)
            - principal diagnosis = "Acute Gastroenteritis"   <-- CONFLICT A
            - admission medication list (baseline for reconciliation)
  Page 2  Progress Note (typed -> text-layer, doc_type=other)
            - principal diagnosis = "Diabetic Ketoacidosis"   <-- CONFLICT B
            - serum sodium 134                                 <-- NA CONFLICT A
  Page 3  Lab Report (typed -> text-layer, doc_type=lab_report)
            - serum sodium 128                                 <-- NA CONFLICT B
            - blood & stool cultures AWAITED                   <-- PENDING
  Page 4  Discharge Note + Med Record (typed -> discharge_note)
            - discharge date, hospital course, follow-up, condition
            - discharge meds: Metformin dose CHANGED w/ NO reason  <-- needs_reconciliation
              Tramadol ADDED, Pantoprazole STOPPED, others CONTINUED
            - discharge set has ondansetron(Emeset)+Tramadol   <-- DRUG INTERACTION
            - NO procedures documented                         <-- forces MISSING
  Page 5  Nursing Note (image only -> NO text layer -> VISION OCR path)
            - DAMA/AMA detail + corroborates penicillin allergy

Run:  .toolvenv/bin/python make_test_pdf.py
Out:  test_patient_record.pdf
"""

from __future__ import annotations

import fitz  # PyMuPDF

PAGE = fitz.paper_rect("a4")
MARGIN = 56
LEADING = 15


def _block(page: fitz.Page, lines: list[tuple[str, int, bool]], y: float) -> float:
    """Write (text, fontsize, bold) lines down the page; return new y."""
    for text, size, bold in lines:
        font = "helv" if not bold else "hebo"
        page.insert_text((MARGIN, y), text, fontsize=size, fontname=font)
        y += LEADING + (size - 10) * 0.6
    return y


def add_text_page(doc: fitz.Document, lines: list[tuple[str, int, bool]]) -> None:
    page = doc.new_page(width=PAGE.width, height=PAGE.height)
    _block(page, lines, MARGIN)


def add_handwritten_page(doc: fitz.Document, note_lines: list[str]) -> None:
    """Render the note as an IMAGE on an otherwise empty page, so the text layer
    is empty and the extractor must route it to the vision-OCR path."""
    tmp = fitz.open()
    tp = tmp.new_page(width=PAGE.width, height=PAGE.height)
    # times-italic, larger, to mimic a quick hand-written chart entry
    y = MARGIN + 10
    for ln in note_lines:
        tp.insert_text((MARGIN, y), ln, fontsize=15, fontname="tiit")
        y += 26
    pix = tp.get_pixmap(dpi=150)
    png = pix.tobytes("png")
    tmp.close()

    page = doc.new_page(width=PAGE.width, height=PAGE.height)
    page.insert_image(PAGE, stream=png)  # full-page image, no selectable text


def build() -> None:
    doc = fitz.open()

    # ---------------- Page 1: Admission Note ----------------
    add_text_page(doc, [
        ("CITYCARE GENERAL HOSPITAL", 15, True),
        ("ADMISSION NOTE", 13, True),
        ("", 10, False),
        ("Patient Name: Ravi Kumar        Age/Sex: 54 / Male", 11, False),
        ("MRN: 884213                     Ward: Medical-3, Bed 12", 11, False),
        ("Reason for Admission / Date of Admission: 21/05/2026", 11, False),
        ("", 10, False),
        ("Chief Complaint:", 11, True),
        ("Vomiting and loose stools x 3 days, generalised weakness.", 11, False),
        ("", 10, False),
        ("History of Present Illness:", 11, True),
        ("Known Type 2 Diabetes Mellitus on oral agents. Presents with", 11, False),
        ("acute onset vomiting and watery diarrhoea, reduced oral intake.", 11, False),
        ("", 10, False),
        ("Provisional / Principal Diagnosis:", 11, True),
        ("Acute Gastroenteritis with dehydration.", 11, False),
        ("Secondary: Type 2 Diabetes Mellitus.", 11, False),
        ("", 10, False),
        ("Known Allergies: PENICILLIN (rash).", 11, True),
        ("", 10, False),
        ("Medications on Admission:", 11, True),
        ("  Tab. Metformin 500mg        1-0-1", 11, False),
        ("  Tab. Telmisartan 40mg       1-0-0", 11, False),
        ("  Tab. Pantoprazole 40mg      1-0-0", 11, False),
        ("  Inj. Emeset 4mg             IV SOS", 11, False),
    ])

    # ---------------- Page 2: Progress Note (conflict source) ----------------
    add_text_page(doc, [
        ("CITYCARE GENERAL HOSPITAL", 15, True),
        ("PHYSICIAN PROGRESS NOTE  -  Day 2 (22/05/2026)", 13, True),
        ("", 10, False),
        ("Patient reassessed. Drowsy, dehydrated, deep breathing noted.", 11, False),
        ("Capillary glucose 412 mg/dL. Urine ketones 3+. Venous pH 7.28.", 11, False),
        ("Serum Sodium 134 mmol/L.", 11, False),
        ("", 10, False),
        ("Revised / Final Diagnosis:", 11, True),
        ("Diabetic Ketoacidosis (DKA), precipitated by gastroenteritis.", 11, False),
        ("", 10, False),
        ("Plan: Start insulin infusion, IV fluids, hourly glucose, K+ monitoring.", 11, False),
        ("Microbiology samples sent (see lab report).", 11, False),
    ])

    # ---------------- Page 3: Lab Report (pending + conflict) ----------------
    add_text_page(doc, [
        ("CITYCARE GENERAL HOSPITAL  -  LABORATORY REPORT", 14, True),
        ("Collected: 22/05/2026", 11, False),
        ("", 10, False),
        ("Investigations / Biochemistry:", 11, True),
        ("  Serum Sodium ............ 128 mmol/L   (135-145)", 11, False),
        ("  Serum Potassium ........ 3.2 mmol/L   (3.5-5.1)", 11, False),
        ("  Random Blood Sugar ..... 415 mg/dL", 11, False),
        ("  Serum Creatinine ....... 1.3 mg/dL", 11, False),
        ("", 10, False),
        ("Haemogram:", 11, True),
        ("  WBC .................... 14,200 /uL", 11, False),
        ("  Haemoglobin ........... 13.1 g/dL", 11, False),
        ("", 10, False),
        ("Microbiology:", 11, True),
        ("  Blood Culture (22/05/2026) ... sample collected, RESULT AWAITED.", 11, False),
        ("  Stool Culture .............. PENDING, report to follow.", 11, False),
    ])

    # ---------------- Page 4: Discharge Note + Med Record ----------------
    add_text_page(doc, [
        ("CITYCARE GENERAL HOSPITAL", 15, True),
        ("DISCHARGE NOTE", 13, True),
        ("", 10, False),
        ("Date of Discharge: 25/05/2026", 11, False),
        ("", 10, False),
        ("Course in the Hospital:", 11, True),
        ("DKA managed with insulin infusion and IV fluids; antiemetics for", 11, False),
        ("vomiting. Acidosis resolved by day 3. Tolerating oral diet at discharge.", 11, False),
        ("", 10, False),
        ("Discharge Medications  (Medication Name / Dosage / Frequency):", 11, True),
        ("  Tab. Metformin 1000mg       1-0-1", 11, False),
        ("  Tab. Telmisartan 40mg       1-0-0", 11, False),
        ("  Tab. Emeset 4mg             1-0-1 SOS", 11, False),
        ("  Tab. Tramadol 50mg          1-0-1   (for abdominal pain)", 11, False),
        ("", 10, False),
        ("Advice on Discharge / Follow-up Instructions:", 11, True),
        ("Review Endocrinology OPD after 1 week. Repeat serum sodium in 3 days.", 11, False),
        ("Return if vomiting, drowsiness or reduced urine output.", 11, False),
        ("", 10, False),
        ("Condition at Discharge: Haemodynamically stable, afebrile, ambulant.", 11, True),
    ])

    # ---------------- Page 5: Handwritten Nursing Note (vision OCR) -------
    add_handwritten_page(doc, [
        "Nursing Note  -  24/05/2026  (2nd shift)",
        "Pt counselled regarding diabetic diet and insulin.",
        "Pt initially insisted on going home early -",
        "DAMA form explained; pt agreed to stay till review.",
        "Allergy band in place: PENICILLIN.",
        "Vitals: BP 130/80, HR 84, Temp 98.4F. Stable.",
        "                                     - Staff Nurse, M-3",
    ])

    out = "test_patient_record.pdf"
    doc.save(out)
    doc.close()
    print(f"wrote {out}  ({len(fitz.open(out))} pages)")


if __name__ == "__main__":
    build()

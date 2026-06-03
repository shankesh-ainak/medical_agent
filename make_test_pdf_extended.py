"""Generate an EXTENSIVE synthetic test record for dscribe. All data fictional.

This is a larger, harder bundle than make_test_pdf.py: 12 pages, several of them
HANDWRITTEN (image-only -> vision-OCR path), with cross-source conflicts that
span typed and handwritten pages.

Patient: Anita Sharma, 67F.  Scenario: admitted as ?pneumonia, later ?PE; AKI;
on warfarin; multiple cultures pending; documentation gaps and disagreements.

Requirement coverage (verify the agent flags each):

  PDF ingestion (hybrid)     pages 1-8 typed (text-layer); pages 9-12 handwritten (vision)
  No fabrication / MISSING    no "follow-up instructions" anywhere -> must be MISSING
  PENDING                     blood culture, sputum culture, pleural-fluid cytology AWAITED
  CONFLICT (never picks):
     principal diagnosis       "Community-Acquired Pneumonia" (p2) vs "Pulmonary Embolism" (p3)
     serum potassium           5.8 (p4 lab) vs 4.2 (p3 progress note)
     allergy                   "NKDA" typed (p1) vs "SULFA - rash" handwritten (p11)
     warfarin dose             5 mg typed discharge (p8) vs 3 mg handwritten order (p10)
  Med reconciliation + flag    dose changes w/ + w/o reasons; added / stopped / continued
  External tool (interactions) discharge set has meropenem+valproate (major) AND
                               ondansetron+tramadol (moderate); first call also fails -> retry
  Procedures (present)         thoracentesis + CT pulmonary angiogram documented (p5-6)

Run:  .toolvenv/bin/python make_test_pdf_extended.py
Out:  test_patient_record_extended.pdf
"""

from __future__ import annotations

import random

import fitz  # PyMuPDF

PAGE = fitz.paper_rect("a4")
MARGIN = 54
LEADING = 14
random.seed(7)


def add_text_page(doc: fitz.Document, lines: list[tuple[str, int, bool]]) -> None:
    page = doc.new_page(width=PAGE.width, height=PAGE.height)
    y = MARGIN
    for text, size, bold in lines:
        page.insert_text((MARGIN, y), text, fontsize=size,
                         fontname="hebo" if bold else "helv")
        y += LEADING + (size - 10) * 0.6


def add_handwritten_page(doc: fitz.Document, lines: list[str], title: str) -> None:
    """Render lines as an IMAGE (no text layer) so the extractor must use vision
    OCR. Slight x/y jitter + italic face to mimic a hand-written chart entry."""
    tmp = fitz.open()
    tp = tmp.new_page(width=PAGE.width, height=PAGE.height)
    tp.insert_text((MARGIN, MARGIN + 6), title, fontsize=16, fontname="tiit")
    y = MARGIN + 44
    for ln in lines:
        x = MARGIN + random.uniform(-3, 8)
        y += 24 + random.uniform(-2, 3)
        tp.insert_text((x, y), ln, fontsize=14 + random.uniform(-0.6, 1.0),
                       fontname="tiit")
    pix = tp.get_pixmap(dpi=120)
    png = pix.tobytes("png")
    tmp.close()
    page = doc.new_page(width=PAGE.width, height=PAGE.height)
    page.insert_image(PAGE, stream=png)


def build() -> None:
    doc = fitz.open()

    # 1 — Face sheet / demographics (typed)
    add_text_page(doc, [
        ("METRO CITY HOSPITAL", 16, True),
        ("PATIENT FACE SHEET", 13, True),
        ("", 10, False),
        ("Patient Name: Anita Sharma            Age / Sex: 67 / Female", 11, False),
        ("MRN: 552901                          Blood Group: O+", 11, False),
        ("Address: 14 Lake View Road, Pune      Phone: 98xxxxxx12", 11, False),
        ("Date of Admission: 03/05/2026         Date of Discharge: 11/05/2026", 11, False),
        ("Attending: Dr. R. Menon (Internal Medicine)", 11, False),
        ("Consultants: Pulmonology, Nephrology", 11, False),
        ("", 10, False),
        ("Known Allergies: NKDA (No Known Drug Allergies).", 11, True),
        ("Emergency Contact: Vikram Sharma (son), 99xxxxxx45.", 11, False),
    ])

    # 2 — Admission note (typed -> admission_note); principal dx A
    add_text_page(doc, [
        ("METRO CITY HOSPITAL  -  ADMISSION NOTE", 14, True),
        ("", 10, False),
        ("Reason for Admission / Date: 03/05/2026", 11, False),
        ("Chief Complaint:", 11, True),
        ("Fever, productive cough and breathlessness for 5 days.", 11, False),
        ("", 10, False),
        ("History of Present Illness:", 11, True),
        ("67F, hypertensive, presents with high-grade fever, purulent sputum,", 11, False),
        ("pleuritic right-sided chest pain and exertional dyspnoea. No haemoptysis.", 11, False),
        ("", 10, False),
        ("Past History: Hypertension (10 yrs), Hypothyroidism. Ex-smoker.", 11, False),
        ("", 10, False),
        ("Provisional / Principal Diagnosis:", 11, True),
        ("Community-Acquired Pneumonia (right lower lobe).", 11, False),
        ("Secondary Diagnoses: Hypertension; Hypothyroidism.", 11, False),
        ("", 10, False),
        ("Examination: T 39.1C, HR 104, BP 148/88, SpO2 91% on room air,", 11, False),
        ("RR 26. Right basal crepitations.", 11, False),
    ])

    # 3 — Progress note (typed -> other); principal dx B (CONFLICT) + K 4.2 (CONFLICT)
    add_text_page(doc, [
        ("METRO CITY HOSPITAL  -  PHYSICIAN PROGRESS NOTE", 14, True),
        ("Day 3  (05/05/2026)", 12, True),
        ("", 10, False),
        ("Persistent hypoxia despite antibiotics. Sudden desaturation episode.", 11, False),
        ("D-dimer markedly elevated. CT pulmonary angiogram ordered (see imaging).", 11, False),
        ("", 10, False),
        ("Revised / Final Diagnosis:", 11, True),
        ("Acute Pulmonary Embolism (segmental, right lower lobe).", 11, False),
        ("Pneumonia now considered less likely given imaging.", 11, False),
        ("", 10, False),
        ("Relevant labs today: Serum Potassium 4.2 mmol/L, Creatinine 1.9 mg/dL.", 11, False),
        ("Plan: start anticoagulation (heparin -> warfarin), continue O2.", 11, False),
    ])

    # 4 — Laboratory report (typed -> lab_report); K 5.8 (CONFLICT) + pending cultures
    add_text_page(doc, [
        ("METRO CITY HOSPITAL  -  LABORATORY REPORT", 14, True),
        ("Collected: 04/05/2026", 11, False),
        ("", 10, False),
        ("Investigations / Biochemistry:", 11, True),
        ("  Serum Sodium ........... 137 mmol/L", 11, False),
        ("  Serum Potassium ....... 5.8 mmol/L   (3.5-5.1)  *HIGH*", 11, False),
        ("  Serum Creatinine ...... 2.3 mg/dL    (0.6-1.1)  *HIGH*", 11, False),
        ("  Urea .................. 64 mg/dL", 11, False),
        ("  CRP ................... 188 mg/L", 11, False),
        ("  D-dimer ............... 4.9 ug/mL    *HIGH*", 11, False),
        ("", 10, False),
        ("Haemogram:", 11, True),
        ("  WBC ................... 17,800 /uL", 11, False),
        ("  Platelets ............. 388,000 /uL", 11, False),
        ("", 10, False),
        ("Microbiology:", 11, True),
        ("  Blood Culture (04/05) ...... RESULT AWAITED.", 11, False),
        ("  Sputum Culture ............. PENDING.", 11, False),
        ("  Pleural Fluid Cytology ..... sample sent, REPORT AWAITED.", 11, False),
    ])

    # 5 — Radiology report (typed -> other); procedure (CTPA) documented
    add_text_page(doc, [
        ("METRO CITY HOSPITAL  -  RADIOLOGY REPORT", 14, True),
        ("Study: CT Pulmonary Angiogram (CTPA)   Date: 05/05/2026", 11, False),
        ("", 10, False),
        ("Findings:", 11, True),
        ("Filling defect in the right lower lobe segmental pulmonary artery,", 11, False),
        ("consistent with acute pulmonary embolism. Small right pleural effusion.", 11, False),
        ("Patchy right basal consolidation also noted.", 11, False),
        ("", 10, False),
        ("Impression: Acute segmental PE with small pleural effusion.", 11, False),
    ])

    # 6 — Procedure note (typed -> other); procedure (thoracentesis) documented
    add_text_page(doc, [
        ("METRO CITY HOSPITAL  -  PROCEDURE NOTE", 14, True),
        ("Procedure: Diagnostic Thoracentesis (right)   Date: 06/05/2026", 11, False),
        ("Operator: Dr. S. Iyer (Pulmonology)", 11, False),
        ("", 10, False),
        ("Indication: right pleural effusion, diagnostic sampling.", 11, False),
        ("Under aseptic precautions and ultrasound guidance, 60 mL of straw-", 11, False),
        ("coloured pleural fluid aspirated and sent for cytology and culture.", 11, False),
        ("No immediate complications. Post-procedure CXR: no pneumothorax.", 11, False),
    ])

    # 7 — Medication administration record (typed -> medication_record); admission meds
    add_text_page(doc, [
        ("METRO CITY HOSPITAL  -  MEDICATION ADMINISTRATION RECORD", 13, True),
        ("(Inpatient / on admission)", 11, False),
        ("", 10, False),
        ("Medication Name        Dosage        Frequency", 11, True),
        ("  Tab. Amlodipine       5mg           1-0-0", 11, False),
        ("  Tab. Thyronorm        50mcg         1-0-0 (before food)", 11, False),
        ("  Inj. Meropenem        1g            1-1-1 (IV)", 11, False),
        ("  Inj. Heparin          5000 IU       SC BD", 11, False),
        ("  Tab. Valproate        500mg         1-0-1", 11, False),
        ("  Inj. Emeset           4mg           IV SOS", 11, False),
        ("", 10, False),
        ("Note: Valproate continued from home (seizure prophylaxis, old CVA).", 11, False),
    ])

    # 8 — Discharge summary + meds (typed -> discharge_note); reconciliation + interactions
    #     NOTE: deliberately NO 'follow-up instructions' anywhere -> MISSING test.
    add_text_page(doc, [
        ("METRO CITY HOSPITAL  -  DISCHARGE SUMMARY", 14, True),
        ("Date of Discharge: 11/05/2026", 11, False),
        ("", 10, False),
        ("Course in the Hospital:", 11, True),
        ("Treated for acute PE with anticoagulation; AKI improved with hydration.", 11, False),
        ("Thoracentesis performed for effusion. Afebrile from day 6.", 11, False),
        ("", 10, False),
        ("Discharge Medications  (Medication Name / Dosage / Frequency):", 11, True),
        ("  Tab. Warfarin 5mg          0-0-1   (start anticoagulation for PE)", 11, False),
        ("  Tab. Amlodipine 10mg       1-0-0", 11, False),
        ("  Tab. Thyronorm 50mcg       1-0-0", 11, False),
        ("  Tab. Valproate 500mg       1-0-1", 11, False),
        ("  Tab. Tramadol 50mg         1-0-1   (pleuritic pain)", 11, False),
        ("  Tab. Emeset 4mg            1-0-1 SOS", 11, False),
        ("", 10, False),
        ("(Meropenem and Heparin stopped at discharge.)", 11, False),
        ("", 10, False),
        ("Condition at Discharge: stable, SpO2 96% on room air, ambulant.", 11, True),
    ])

    # 9 — Handwritten nursing note (vision)
    add_handwritten_page(doc, [
        "Pt resting comfortably, afebrile this shift.",
        "O2 weaned to room air, SpO2 95-96%.",
        "Anticoagulation started - watch for bleeding,",
        "gums / urine. INR sample sent.",
        "Pt anxious about going home, reassured.",
        "Pain score 3/10 - given Tramadol as charted.",
        "          - Staff Nurse, Ward 5B",
    ], "Nursing Progress Note   08/05/2026")

    # 10 — Handwritten doctor's order sheet (vision); warfarin 3mg (CONFLICT w/ p8)
    add_handwritten_page(doc, [
        "1) Warfarin 3 mg PO once daily at night.",
        "   ** check INR in 3 days, target 2-3 **",
        "2) Continue Valproate 500 mg BD.",
        "3) Stop Meropenem - course complete.",
        "4) Amlodipine - increase to 10 mg OD",
        "   (BP still 150/90).",
        "5) Tramadol 50 mg BD PRN for pain.",
        "                 - Dr. R. Menon",
    ], "Doctor's Order Sheet   10/05/2026")

    # 11 — Handwritten allergy + vitals chart (vision); SULFA allergy (CONFLICT w/ p1 NKDA)
    add_handwritten_page(doc, [
        "ALLERGY ALERT: SULFA drugs - rash (per pt's daughter).",
        "Band applied. Charted in red.",
        "",
        "Vitals (10/05):",
        "  Temp 98.6 F   HR 80   BP 150/90",
        "  RR 18   SpO2 96% RA",
        "Weight 62 kg.   Intake/output adequate.",
    ], "Allergy & Vitals Chart   10/05/2026")

    # 12 — Handwritten discharge counselling slip (vision)
    add_handwritten_page(doc, [
        "Discharge counselling done with pt + son.",
        "Explained warfarin - take same time daily,",
        "avoid NSAIDs, watch for bleeding.",
        "Diet advice given. Pt verbalised understanding.",
        "(Next-visit / follow-up date NOT yet fixed -",
        " to be decided by OPD.)",
        "                 - Staff Nurse, Ward 5B",
    ], "Discharge Counselling Slip   11/05/2026")

    out = "test_patient_record_extended.pdf"
    doc.save(out, deflate=True)
    doc.close()
    print(f"wrote {out}  ({len(fitz.open(out))} pages)")


if __name__ == "__main__":
    build()

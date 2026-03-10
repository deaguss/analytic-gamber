import json
import pandas as pd
from pathlib import Path

CAT_FILL = {
    'JUDI':       'FFCCCC',   
    'TIDAK_JUDI': 'CCFFCC',   
    'AMBIGU':     'FFF2CC',   
    'UNKNOWN':    'F2F2F2',  
}
CAT_LABEL = {
    'JUDI':       'Berindikasi Judi',
    'TIDAK_JUDI': 'Tidak Judi',
    'AMBIGU':     'Ambigu',
    'UNKNOWN':    'Tidak Diketahui',
}


def apply_colors(writer, df: pd.DataFrame, sheet_name: str):
    from openpyxl.styles import PatternFill, Font, Alignment
    from openpyxl.utils import get_column_letter

    code_series = df['_cat_code'].tolist() if '_cat_code' in df.columns else []
    display     = df.drop(columns=['_cat_code'], errors='ignore')

    display.to_excel(writer, sheet_name=sheet_name, index=False)
    ws = writer.sheets[sheet_name]

    for cell in ws[1]:
        cell.font      = Font(bold=True)
        cell.alignment = Alignment(horizontal='center', vertical='center')

    for ci, col in enumerate(display.columns, 1):
        vals    = display[col].astype(str).tolist() + [str(col)]
        max_len = max(len(v) for v in vals) + 4
        ws.column_dimensions[get_column_letter(ci)].width = min(max_len, 45)

    for ri, cat in enumerate(code_series, start=2):
        fill = PatternFill(
            start_color=CAT_FILL.get(cat, 'F2F2F2'),
            end_color=CAT_FILL.get(cat, 'F2F2F2'),
            fill_type='solid'
        )
        for ci in range(1, len(display.columns) + 1):
            ws.cell(row=ri, column=ci).fill = fill


def add_legend(writer):
    from openpyxl.styles import PatternFill, Font, Alignment
    ws = writer.book.create_sheet('Legenda Kategori')
    for ci, h in enumerate(['Kode', 'Label', 'Keterangan'], 1):
        c = ws.cell(row=1, column=ci, value=h)
        c.font = Font(bold=True)
        c.alignment = Alignment(horizontal='center')
    rows = [
        ('JUDI',       'Berindikasi Judi',  'Merah muda  — video berindikasi komentar judi dominan'),
        ('TIDAK_JUDI', 'Tidak Judi',        'Hijau muda  — video dengan komentar dominan bukan judi'),
        ('AMBIGU',     'Ambigu',            'Kuning muda — video dengan komentar campuran / tidak jelas'),
    ]
    for ri, (code, label, desc) in enumerate(rows, start=2):
        ws.cell(row=ri, column=1, value=code)
        ws.cell(row=ri, column=2, value=label)
        ws.cell(row=ri, column=3, value=desc)
        fill = PatternFill(start_color=CAT_FILL[code], end_color=CAT_FILL[code], fill_type='solid')
        for ci in range(1, 4):
            ws.cell(row=ri, column=ci).fill = fill
    for col, w in [('A', 14), ('B', 22), ('C', 55)]:
        ws.column_dimensions[col].width = w


def generate_excel():
    results_dir = Path("results")
    json_path   = results_dir / "aggregate_results.json"

    if not json_path.exists():
        print(f"File {json_path} tidak ditemukan.")
        return

    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    videos = data.get('per_video_results', [])
    if not videos:
        print("Tidak ada data video untuk diekspor.")
        return

    def code(v):  return v.get('sample_category', 'UNKNOWN')
    def label(v): return CAT_LABEL.get(code(v), code(v))

    # Rekap Keseluruhan
    df_rekap = pd.DataFrame(videos)
    if 'sample_category' not in df_rekap.columns:
        df_rekap['sample_category'] = 'UNKNOWN'
    df_rekap['Kategori Sampel'] = df_rekap['sample_category'].map(lambda c: CAT_LABEL.get(c, c))
    df_rekap['_cat_code']       = df_rekap['sample_category']
    cols = ['Kategori Sampel', '_cat_code'] + [
        c for c in df_rekap.columns if c not in ('Kategori Sampel', '_cat_code', 'sample_category')
    ]
    df_rekap = df_rekap[cols]

    # Agreement
    df_agreement = pd.DataFrame([{
        'Kategori Sampel'   : label(v),
        'Video ID'          : v.get('video_id'),
        'Komentar Ditarik'  : v.get('total_comments'),
        'Anomali (Masuk AI)': v.get('rule_based_ambigu'),
        'Gemini (Judi)'     : v.get('gemini_judi'),
        'GPT (Judi)'        : v.get('gpt_judi'),
        'Sepakat'           : v.get('agreement_count'),
        'Tidak Sepakat'     : v.get('disagreement_count'),
        'Agreement Rate (%)': v.get('agreement_rate'),
        '_cat_code'         : code(v),
    } for v in videos])

    # Confidence
    df_confidence = pd.DataFrame([{
        'Kategori Sampel'      : label(v),
        'Video ID'             : v.get('video_id'),
        'Anomali (Masuk AI)'   : v.get('rule_based_ambigu'),
        'Gemini Avg Confidence': v.get('gemini_avg_confidence'),
        'GPT Avg Confidence'   : v.get('gpt_avg_confidence'),
        '_cat_code'            : code(v),
    } for v in videos])

    # Latency
    df_latency = pd.DataFrame([{
        'Kategori Sampel'          : label(v),
        'Video ID'                 : v.get('video_id'),
        'Anomali (Masuk AI)'       : v.get('rule_based_ambigu'),
        'Gemini Avg Latency (ms)'  : v.get('gemini_avg_latency'),
        'GPT Avg Latency (ms)'     : v.get('gpt_avg_latency'),
        'Total Processing Time (s)': v.get('processing_time_seconds'),
        '_cat_code'                : code(v),
    } for v in videos])

    # Cost
    df_cost = pd.DataFrame([{
        'Kategori Sampel'     : label(v),
        'Video ID'            : v.get('video_id'),
        'Anomali (Masuk AI)'  : v.get('rule_based_ambigu'),
        'Gemini Input Tokens' : v.get('gemini_total_input_tokens', 0),
        'Gemini Output Tokens': v.get('gemini_total_output_tokens', 0),
        'Gemini Cost (USD)'   : v.get('gemini_total_cost_usd', 0.0),
        'GPT Input Tokens'    : v.get('gpt_total_input_tokens', 0),
        'GPT Output Tokens'   : v.get('gpt_total_output_tokens', 0),
        'GPT Cost (USD)'      : v.get('gpt_total_cost_usd', 0.0),
        'Total Cost (USD)'    : v.get('total_cost_usd', 0.0),
        '_cat_code'           : code(v),
    } for v in videos])

    file_map = [
        (df_rekap,      "1_Rekap_Keseluruhan.xlsx"),
        (df_agreement,  "2_Report_Agreement.xlsx"),
        (df_confidence, "3_Report_Confidence.xlsx"),
        (df_latency,    "4_Report_Latency.xlsx"),
        (df_cost,       "5_Report_Cost.xlsx"),
    ]

    for df, fname in file_map:
        path = results_dir / fname
        with pd.ExcelWriter(path, engine='openpyxl') as writer:
            apply_colors(writer, df, 'Data')
            add_legend(writer)
        print(f"  Saved: {fname}")

    print("\nBerhasil membuat 5 file Excel dengan color coding kategori sampel!")
    print("  Merah muda  = JUDI (Berindikasi Judi)")
    print("  Hijau muda  = TIDAK_JUDI (Tidak Judi)")
    print("  Kuning muda = AMBIGU")


if __name__ == "__main__":
    generate_excel()
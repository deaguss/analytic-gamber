import json
import pandas as pd
from pathlib import Path

def generate_excel():
    results_dir = Path("results")
    json_path = results_dir / "aggregate_results.json"

    if not json_path.exists():
        print(f"File {json_path} tidak ditemukan.")
        return

    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    videos = data.get('per_video_results', [])
    if not videos:
        print("Tidak ada data video untuk diekspor.")
        return

    df_rekap = pd.DataFrame(videos)

    df_agreement = pd.DataFrame([{
        'Video ID': v.get('video_id'),
        'Komentar Ditarik': v.get('total_comments'),
        'Anomali (Masuk AI)': v.get('rule_based_ambigu'),
        'Gemini (Judi)': v.get('gemini_judi'),
        'GPT (Judi)': v.get('gpt_judi'),
        'Sepakat': v.get('agreement_count'),
        'Tidak Sepakat': v.get('disagreement_count'),
        'Agreement Rate (%)': v.get('agreement_rate')
    } for v in videos])

    df_confidence = pd.DataFrame([{
        'Video ID': v.get('video_id'),
        'Anomali (Masuk AI)': v.get('rule_based_ambigu'),
        'Gemini Avg Confidence': v.get('gemini_avg_confidence'),
        'GPT Avg Confidence': v.get('gpt_avg_confidence')
    } for v in videos])

    df_latency = pd.DataFrame([{
        'Video ID': v.get('video_id'),
        'Anomali (Masuk AI)': v.get('rule_based_ambigu'),
        'Gemini Avg Latency (ms)': v.get('gemini_avg_latency'),
        'GPT Avg Latency (ms)': v.get('gpt_avg_latency'),
        'Total Processing Time (s)': v.get('processing_time_seconds')
    } for v in videos])

    df_cost = pd.DataFrame([{
        'Video ID': v.get('video_id'),
        'Anomali (Masuk AI)': v.get('rule_based_ambigu'),
        'Gemini Input Tokens': v.get('gemini_total_input_tokens', 0),
        'Gemini Output Tokens': v.get('gemini_total_output_tokens', 0),
        'Gemini Cost (USD)': v.get('gemini_total_cost_usd', 0.0),
        'GPT Input Tokens': v.get('gpt_total_input_tokens', 0),
        'GPT Output Tokens': v.get('gpt_total_output_tokens', 0),
        'GPT Cost (USD)': v.get('gpt_total_cost_usd', 0.0),
        'Total Cost (USD)': v.get('total_cost_usd', 0.0),
    } for v in videos])

    df_rekap.to_excel(results_dir / "1_Rekap_Keseluruhan.xlsx", index=False)
    df_agreement.to_excel(results_dir / "2_Report_Agreement.xlsx", index=False)
    df_confidence.to_excel(results_dir / "3_Report_Confidence.xlsx", index=False)
    df_latency.to_excel(results_dir / "4_Report_Latency.xlsx", index=False)
    df_cost.to_excel(results_dir / "5_Report_Cost.xlsx", index=False)

    print("Berhasil membuat 5 file Excel terpisah di folder 'results'!")

if __name__ == "__main__":
    generate_excel()
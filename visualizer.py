import json
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
import numpy as np
import seaborn as sns

# Set style - Flat design
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['axes.titlesize'] = 13
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['legend.fontsize'] = 10

class ResearchVisualizer:
    """Visualizer untuk hasil penelitian V2 (Per-Video Aggregation)"""
    
    def __init__(self, results_dir="results", output_dir="visualizations"):
        self.results_dir = Path(results_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Load data dari 1 file utama
        self.agg_data = self._load_json("aggregate_results.json")
        
        # Ekstrak data untuk mempermudah plotting
        if self.agg_data:
            self._extract_metrics()
    
    def _load_json(self, filename):
        filepath = self.results_dir / filename
        if not filepath.exists():
            print(f"Warning: {filename} tidak ditemukan di {self.results_dir}")
            return None
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)

    def _extract_metrics(self):
        """Menghitung total, min, max, median dari array per_video_results"""
        self.summary = self.agg_data.get('summary', {})
        self.averages = self.agg_data.get('averages', {})
        videos = self.agg_data.get('per_video_results', [])
        
        # Mengumpulkan array untuk dihitung distribusinya
        self.g_conf_arr = [v['gemini_avg_confidence'] for v in videos if v['gemini_avg_confidence'] > 0]
        self.gpt_conf_arr = [v['gpt_avg_confidence'] for v in videos if v['gpt_avg_confidence'] > 0]
        self.g_lat_arr = [v['gemini_avg_latency'] for v in videos if v['gemini_avg_latency'] > 0]
        self.gpt_lat_arr = [v['gpt_avg_latency'] for v in videos if v['gpt_avg_latency'] > 0]
        
        # Total deteksi
        self.tot_gemini_judi = sum(v['gemini_judi'] for v in videos)
        self.tot_gemini_bukan = sum(v['gemini_bukan_judi'] for v in videos)
        self.tot_gpt_judi = sum(v['gpt_judi'] for v in videos)
        self.tot_gpt_bukan = sum(v['gpt_bukan_judi'] for v in videos)
        
        # Total Agreement
        self.tot_agreed = sum(v['agreement_count'] for v in videos)
        self.tot_disagreed = sum(v['disagreement_count'] for v in videos)
        
        # Labeling (Warna & Nama)
        self.g_label = 'Platform Gemini'
        self.gpt_label = 'OpenAI GPT (ChatGPT)'
        self.g_color = '#4285F4' # Biru Google
        self.gpt_color = '#34A853' # Hijau OpenAI

    def add_labels_and_space(self, ax, bars_list):
        max_height = 0
        for bars in bars_list:
            for bar in bars:
                height = bar.get_height()
                if height > max_height: max_height = height
                ax.text(bar.get_x() + bar.get_width()/2., height,
                        f'{float(height):.2f}' if height % 1 != 0 else f'{int(height)}',
                        ha='center', va='bottom', fontsize=9, fontweight='bold')
        ax.set_ylim(0, max_height * 1.25)
    
    def create_all_visualizations(self):
        if not self.agg_data:
            print("Visualisasi dibatalkan. Data tidak tersedia.")
            return
            
        print("\nCreating visualizations...\n")
        self.plot_confidence_distribution()
        self.plot_latency_distribution()
        self.plot_agreement_analysis()
        self.plot_classification_comparison()
        self.plot_complete_dashboard()
        self.plot_research_flow()
        print("\nAll visualizations created!")
        print(f" Saved to: {self.output_dir}/")
    
    def plot_confidence_distribution(self):
        fig, ax = plt.subplots(figsize=(12, 6))
        
        metrics = ['Mean', 'Median', 'Min', 'Max']
        gemini_values = [np.mean(self.g_conf_arr), np.median(self.g_conf_arr), np.min(self.g_conf_arr), np.max(self.g_conf_arr)] if self.g_conf_arr else [0,0,0,0]
        gpt_values = [np.mean(self.gpt_conf_arr), np.median(self.gpt_conf_arr), np.min(self.gpt_conf_arr), np.max(self.gpt_conf_arr)] if self.gpt_conf_arr else [0,0,0,0]
        
        x = np.arange(len(metrics))
        width = 0.35
        
        bars1 = ax.bar(x - width/2, gemini_values, width, label=self.g_label, color=self.g_color, alpha=0.8)
        bars2 = ax.bar(x + width/2, gpt_values, width, label=self.gpt_label, color=self.gpt_color, alpha=0.8)

        ax.set_title('Confidence Score Distribution - Comparison (Averages per Video)', fontweight='bold', fontsize=14)
        ax.set_ylabel('Confidence Score', fontsize=11)
        ax.set_xlabel('Metrics', fontsize=11)
        ax.set_xticks(x)
        ax.set_xticklabels(metrics)
        ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.0), ncol=2, framealpha=0.9)
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        
        self.add_labels_and_space(ax, [bars1, bars2])
        
        plt.tight_layout()
        plt.savefig(self.output_dir / '1_confidence_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("Created: 1_confidence_distribution.png")
    
    def plot_latency_distribution(self):
        fig, ax = plt.subplots(figsize=(12, 6))
        
        metrics = ['Mean', 'Median', 'Min', 'Max']
        gemini_values = [np.mean(self.g_lat_arr), np.median(self.g_lat_arr), np.min(self.g_lat_arr), np.max(self.g_lat_arr)] if self.g_lat_arr else [0,0,0,0]
        gpt_values = [np.mean(self.gpt_lat_arr), np.median(self.gpt_lat_arr), np.min(self.gpt_lat_arr), np.max(self.gpt_lat_arr)] if self.gpt_lat_arr else [0,0,0,0]
        
        x = np.arange(len(metrics))
        width = 0.35
        
        bars1 = ax.bar(x - width/2, gemini_values, width, label=self.g_label, color=self.g_color, alpha=0.8)
        bars2 = ax.bar(x + width/2, gpt_values, width, label=self.gpt_label, color=self.gpt_color, alpha=0.8)
        
        ax.set_title('Latency Distribution - Comparison (ms)', fontweight='bold', fontsize=14)
        ax.set_ylabel('Latency (milliseconds)', fontsize=11)
        ax.set_xlabel('Metrics', fontsize=11)
        ax.set_xticks(x)
        ax.set_xticklabels(metrics)
        ax.legend(loc='upper left', fontsize=10)
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        
        self.add_labels_and_space(ax, [bars1, bars2])
        
        plt.tight_layout()
        plt.savefig(self.output_dir / '2_latency_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("Created: 2_latency_distribution.png")
    
    def plot_agreement_analysis(self):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        sizes = [self.tot_agreed, self.tot_disagreed]
        labels = [f"Setuju\n{self.tot_agreed} komentar", f"Tidak Setuju\n{self.tot_disagreed} komentar"]
        colors = ['#34A853', '#EA4335']
        
        # Handle division by zero jika datanya kosong
        total_ambigu = self.summary.get('total_ambigu', 0)
        agreement_rate = (self.tot_agreed / total_ambigu * 100) if total_ambigu > 0 else 0

        ax1.pie(sizes, explode=(0.05, 0.05), labels=labels, colors=colors,
               autopct='%1.1f%%', shadow=False, startangle=90,
               textprops={'fontsize': 11, 'fontweight': 'bold'})
        ax1.set_title(f'Overall Agreement Rate: {agreement_rate:.2f}%', fontsize=14, fontweight='bold', pad=20)
        
        categories = ['Total\nAmbigu', 'Kesepakatan\n(Agreement)', 'Ketidaksepakatan\n(Disagreement)']
        values = [total_ambigu, self.tot_agreed, self.tot_disagreed]
        bars = ax2.bar(categories, values, color=['#4285F4', '#34A853', '#EA4335'])
        ax2.set_title('Breakdown Analisis Kesepakatan', fontweight='bold')
        ax2.set_ylabel('Jumlah Komentar')
        ax2.grid(axis='y', alpha=0.3)
        self.add_labels_and_space(ax2, [bars])
        
        plt.tight_layout()
        plt.savefig(self.output_dir / '3_agreement_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("Created: 3_agreement_analysis.png")
    
    def plot_classification_comparison(self):
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))
        
        # 1. Classification counts
        x = np.arange(2)
        width = 0.35
        bars1 = ax1.bar(x - width/2, [self.tot_gemini_judi, self.tot_gemini_bukan],
                       width, label=self.g_label, color=self.g_color, alpha=0.8)
        bars2 = ax1.bar(x + width/2, [self.tot_gpt_judi, self.tot_gpt_bukan],
                       width, label=self.gpt_label, color=self.gpt_color, alpha=0.8)
        
        ax1.set_title('Perbandingan Hasil Klasifikasi LLM', fontweight='bold', fontsize=12)
        ax1.set_ylabel('Jumlah Komentar')
        ax1.set_xticks(x)
        ax1.set_xticklabels(['Judi Online', 'Bukan Judi Online'])
        ax1.legend(loc='upper right', framealpha=0.9)
        ax1.grid(axis='y', alpha=0.3, linestyle='--')
        self.add_labels_and_space(ax1, [bars1, bars2])
        
        # 2. Detection Rate Comparison
        total_ambigu = self.summary.get('total_ambigu', 1)
        gemini_rate = (self.tot_gemini_judi / total_ambigu) * 100
        gpt_rate = (self.tot_gpt_judi / total_ambigu) * 100
        
        bars = ax2.bar([self.g_label, self.gpt_label], [gemini_rate, gpt_rate], 
                       color=[self.g_color, self.gpt_color], alpha=0.8)
        ax2.set_title('Tingkat Deteksi Judi (Detection Rate)', fontweight='bold', fontsize=12)
        ax2.set_ylabel('Detection Rate (%)')
        ax2.grid(axis='y', alpha=0.3, linestyle='--')
        self.add_labels_and_space(ax2, [bars])
        
        # 3. Confidence Comparison
        bars = ax3.bar([self.g_label, self.gpt_label], [self.averages.get('gemini_confidence',0), self.averages.get('gpt_confidence',0)],
                      color=[self.g_color, self.gpt_color], alpha=0.8)
        ax3.set_title('Average Confidence Score', fontweight='bold', fontsize=12)
        ax3.set_ylabel('Confidence Score')
        ax3.grid(axis='y', alpha=0.3, linestyle='--')
        self.add_labels_and_space(ax3, [bars])
        
        # 4. Latency Comparison
        bars = ax4.bar([self.g_label, self.gpt_label], [self.averages.get('gemini_latency_ms',0), self.averages.get('gpt_latency_ms',0)],
                      color=[self.g_color, self.gpt_color], alpha=0.8)
        ax4.set_title('Average Latency (Lower is Better)', fontweight='bold', fontsize=12)
        ax4.set_ylabel('Latency (ms)')
        ax4.grid(axis='y', alpha=0.3, linestyle='--')
        self.add_labels_and_space(ax4, [bars])
        
        plt.tight_layout()
        plt.savefig(self.output_dir / '4_classification_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("Created: 4_classification_comparison.png")
    
    def plot_complete_dashboard(self):
        fig = plt.figure(figsize=(16, 12))
        gs = fig.add_gridspec(3, 3, hspace=0.4, wspace=0.3)
        
        fig.suptitle('DASHBOARD ANALISIS KOMPARASI: GEMINI vs OPENAI GPT\nDeteksi Konten Judi Online pada Komentar YouTube',
                    fontsize=16, fontweight='bold', y=0.98)

        # 1. Rule-based Statistics (Pencocokan Karakter)
        ax1 = fig.add_subplot(gs[0, 0])
        labels = ['Judi\nOnline', 'Bukan\nJudi', 'Ambigu\n(Masuk AI)']
        values = [self.summary.get('total_judi_online', 0), self.summary.get('total_bukan_judi', 0), self.summary.get('total_ambigu', 0)]
        
        bars = ax1.bar(labels, values, color=['#EA4335', '#34A853', '#FBBC04'])
        ax1.set_title('Pencocokan Karakter\n(Regex & Fuzzy)', fontweight='bold')
        ax1.set_ylabel('Jumlah Komentar')
        ax1.grid(axis='y', alpha=0.3)
        self.add_labels_and_space(ax1, [bars])
        
        # 2. Agreement Rate
        ax2 = fig.add_subplot(gs[0, 1])
        agreement_rate = self.averages.get('agreement_rate', 0)
        ax2.pie([agreement_rate, 100 - agreement_rate], colors=['#34A853', '#E8E8E8'], 
                autopct='%1.1f%%', startangle=90, shadow=False, textprops={'fontweight': 'bold'})
        ax2.set_title(f'Avg Agreement Rate\n{agreement_rate:.1f}%', fontweight='bold')
        
        # 3. Summary Text
        ax3 = fig.add_subplot(gs[0, 2])
        ax3.axis('off')
        
        g_lat, gpt_lat = self.averages.get('gemini_latency_ms',0), self.averages.get('gpt_latency_ms',0)
        faster = "OpenAI GPT" if gpt_lat < g_lat else "Platform Gemini"
        
        g_conf, gpt_conf = self.averages.get('gemini_confidence',0), self.averages.get('gpt_confidence',0)
        confident = "OpenAI GPT" if gpt_conf > g_conf else "Platform Gemini"
        
        summary_text = f"""
PERFORMA MODEL SUMMARY (All Videos)

{self.g_label} (Blue):
   Judi Online: {self.tot_gemini_judi}
   Bukan Judi: {self.tot_gemini_bukan}
   Avg Conf: {g_conf:.3f}
   Avg Latency: {g_lat:.1f}ms

{self.gpt_label} (Green):
   Judi Online: {self.tot_gpt_judi}
   Bukan Judi: {self.tot_gpt_bukan}
   Avg Conf: {gpt_conf:.3f}
   Avg Latency: {gpt_lat:.1f}ms

KESIMPULAN UMUM:
✓ API Tercepat: {faster}
✓ Confidence Tertinggi: {confident}
✓ Rata-rata Konsistensi: {agreement_rate:.1f}%
        """
        ax3.text(0.1, 0.5, summary_text.strip(), transform=ax3.transAxes,
                fontsize=9, verticalalignment='center', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
        
        # 4. Classification Distribution
        ax4 = fig.add_subplot(gs[1, :])
        x = np.arange(2)
        width = 0.35
        
        bars1 = ax4.bar(x - width/2, [self.tot_gemini_judi, self.tot_gemini_bukan], width, label=self.g_label, color=self.g_color)
        bars2 = ax4.bar(x + width/2, [self.tot_gpt_judi, self.tot_gpt_bukan], width, label=self.gpt_label, color=self.gpt_color)
        
        ax4.set_title('Perbandingan Klasifikasi LLM pada Data Ambigu', fontweight='bold', fontsize=12)
        ax4.set_ylabel('Jumlah Komentar')
        ax4.set_xticks(x)
        ax4.set_xticklabels(['Judi Online', 'Bukan Judi Online'])
        ax4.legend(loc='upper center', bbox_to_anchor=(0.5, 1.0), ncol=2)
        ax4.grid(axis='y', alpha=0.3)
        self.add_labels_and_space(ax4, [bars1, bars2])
        
        # 5. Confidence
        ax5 = fig.add_subplot(gs[2, 0])
        bars1 = ax5.bar(x - width/2, [np.mean(self.g_conf_arr), np.median(self.g_conf_arr)], width, color=self.g_color)
        bars2 = ax5.bar(x + width/2, [np.mean(self.gpt_conf_arr), np.median(self.gpt_conf_arr)], width, color=self.gpt_color)
        ax5.set_title('Confidence Score (Video Averages)', fontweight='bold')
        ax5.set_xticks(x)
        ax5.set_xticklabels(['Mean', 'Median'])
        ax5.grid(axis='y', alpha=0.3)
        self.add_labels_and_space(ax5, [bars1, bars2])
        
        # 6. Latency
        ax6 = fig.add_subplot(gs[2, 1])
        bars1 = ax6.bar(x - width/2, [np.mean(self.g_lat_arr), np.median(self.g_lat_arr)], width, color=self.g_color)
        bars2 = ax6.bar(x + width/2, [np.mean(self.gpt_lat_arr), np.median(self.gpt_lat_arr)], width, color=self.gpt_color)
        ax6.set_title('Latency ms (Video Averages)', fontweight='bold')
        ax6.set_xticks(x)
        ax6.set_xticklabels(['Mean', 'Median'])
        ax6.grid(axis='y', alpha=0.3)
        self.add_labels_and_space(ax6, [bars1, bars2])
        
        # 7. Metadata
        ax7 = fig.add_subplot(gs[2, 2])
        ax7.axis('off')
        metadata_text = f"""
METADATA PENELITIAN
Total Video Diproses: {self.summary.get('total_videos', 0)}
Total Komentar Ditarik: {self.summary.get('total_comments', 0)}

Hasil Pencocokan Karakter (Tahap 1):
  - Ekplisit Judi: {self.summary.get('total_judi_online', 0)}
  - Jelas Bukan: {self.summary.get('total_bukan_judi', 0)}
  - Lolos sbg Ambigu: {self.summary.get('total_ambigu', 0)}
  
Di-generate pada: 
{self.agg_data.get('generated_at', '')[:10]}
        """
        ax7.text(0.1, 0.5, metadata_text.strip(), transform=ax7.transAxes,
                fontsize=9, verticalalignment='center', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.2))
        
        plt.savefig(self.output_dir / '5_complete_dashboard.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("Created: 5_complete_dashboard.png")
    
    def plot_research_flow(self):
        fig, ax = plt.subplots(figsize=(14, 10))
        ax.axis('off')
        
        ax.text(0.5, 0.95, 'ALUR METODOLOGI PENELITIAN', ha='center', va='top', fontsize=16, fontweight='bold')
        ax.text(0.5, 0.92, 'Deteksi Konten Judi Online: Cascading Filtering Approach', ha='center', va='top', fontsize=11, style='italic')
        
        stages = [
            {'y': 0.85, 'title': '1. INPUT DATA', 'text': f"Komentar dari {self.summary.get('total_videos', 'X')} Video YouTube\n(JSON Data)", 'color': '#E3F2FD', 'border': '#2196F3'},
            {'y': 0.75, 'title': '2. PRA-PEMROSESAN', 'text': 'Case Folding -> Leet Speak Conversion\n-> Tokenisasi -> Pembersihan Tanda Baca', 'color': '#E8F5E9', 'border': '#4CAF50'},
            {'y': 0.65, 'title': '3. PENCOCOKAN KARAKTER (Regex & Fuzzy)', 'text': 'Pattern Matching Sederhana Tanpa AI', 'color': '#FFF3E0', 'border': '#FF9800'},
            {'y': 0.52, 'title': '   HASIL PENCOCOKAN AWAL', 'text': '   Sangat Mirip -> JUDI ONLINE\n   Tidak Mirip -> BUKAN JUDI\n   Samar-samar -> DATA AMBIGU', 'color': '#FCE4EC', 'border': '#E91E63'},
            {'y': 0.38, 'title': '4. KLASIFIKASI AI (Untuk Data Ambigu)', 'text': f'Platform Gemini\nvs\n{self.gpt_label}', 'color': '#F3E5F5', 'border': '#9C27B0'},
            {'y': 0.25, 'title': '5. AGREGASI DATA PER VIDEO', 'text': 'Menghitung Rata-rata Agreement Rate,\nConfidence Distribution, & Latency', 'color': '#E0F2F1', 'border': '#009688'},
            {'y': 0.15, 'title': '6. PELAPORAN EXCEL & VISUALISASI', 'text': 'Grafik Visualisasi + Tabel Komparasi + Kesimpulan', 'color': '#FFF9C4', 'border': '#FBC02D'}
        ]
        
        for stage in stages:
            rect = mpatches.FancyBboxPatch(
                (0.15, stage['y'] - 0.05), 0.7, 0.08, boxstyle="round,pad=0.01", facecolor=stage['color'], edgecolor=stage['border'], linewidth=2)
            ax.add_patch(rect)
            ax.text(0.5, stage['y'] + 0.015, stage['title'], ha='center', va='center', fontsize=11, fontweight='bold')
            ax.text(0.5, stage['y'] - 0.015, stage['text'], ha='center', va='center', fontsize=9, style='italic')
            if stage['y'] > 0.2:
                ax.annotate('', xy=(0.5, stage['y'] - 0.06), xytext=(0.5, stage['y'] - 0.04), arrowprops=dict(arrowstyle='->', lw=2, color='#424242'))
        
        legend_y = 0.05
        ax.text(0.15, legend_y, 'KEY INSIGHT:', fontsize=10, fontweight='bold')
        ax.text(0.15, legend_y - 0.02, 
               'Cascading approach: Filter Rule-Based menangani kasus yang jelas dengan cepat,\nAI hanya memproses data ambigu lalu dirata-rata per video agar pengujian lebih kuat secara statistik.',
               fontsize=8, style='italic')
        
        plt.savefig(self.output_dir / '6_research_flow_diagram.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("Created: 6_research_flow_diagram.png")

def main():
    print("=" * 70)
    print("RESEARCH RESULTS VISUALIZER V2")
    print("=" * 70)
    visualizer = ResearchVisualizer()
    visualizer.create_all_visualizations()

if __name__ == "__main__":
    main()
import json
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
import numpy as np
import seaborn as sns

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['axes.titlesize'] = 13
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['legend.fontsize'] = 10

class ResearchVisualizer:
    """Visualizer untuk hasil penelitian"""
    
    def __init__(self, data_dir="data", output_dir="visualizations"):
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Load data
        self.load_data()
    
    def load_data(self):
        """Load semua data hasil analisis"""
        print("Loading data...")
        
        self.final_report = self._load_json("final_report.json")
        self.agreement = self._load_json("agreement_analysis.json")
        self.distribution = self._load_json("distribution_analysis.json")
        self.comparative = self._load_json("comparative_analysis.json")
        self.gemini_results = self._load_json("gemini_results.json")
        self.gpt2_results = self._load_json("gpt2_results.json")
        
        print("Data loaded successfully!")
    
    def _load_json(self, filename):
        """Helper untuk load JSON"""
        filepath = self.data_dir / filename
        if not filepath.exists():
            print(f"Warning: {filename} not found")
            return None
        
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def create_all_visualizations(self):
        """Buat semua visualisasi"""
        print("\nCreating visualizations...\n")
        
        # 1. Confidence Distribution
        self.plot_confidence_distribution()
        
        # 2. Latency Distribution
        self.plot_latency_distribution()
        
        # 3. Agreement Analysis
        self.plot_agreement_analysis()
        
        # 4. Classification Comparison
        self.plot_classification_comparison()
        
        # 5. Complete Dashboard
        self.plot_complete_dashboard()
        
        # 6. Research Methodology Flow
        self.plot_research_flow()
        
        print("\nAll visualizations created!")
        print(f" Saved to: {self.output_dir}/")
    
    def plot_confidence_distribution(self):
        """Plot distribusi confidence score - comparison chart"""
        if not self.distribution:
            print("Skipping confidence distribution - no data")
            return
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        gemini_conf = self.distribution['confidence']['gemini']
        gpt2_conf = self.distribution['confidence']['gpt2']
        
        metrics = ['Mean', 'Median', 'Min', 'Max']
        gemini_values = [gemini_conf.get('mean', 0), gemini_conf.get('median', 0), 
                        gemini_conf.get('min', 0), gemini_conf.get('max', 0)]
        gpt2_values = [gpt2_conf.get('mean', 0), gpt2_conf.get('median', 0),
                       gpt2_conf.get('min', 0), gpt2_conf.get('max', 0)]
        
        x = np.arange(len(metrics))
        width = 0.35
        
        # Warna Konsisten
        gemini_color = '#4285F4'
        gpt2_color = '#34A853'
        
        # Plot Bars
        bars1 = ax.bar(x - width/2, gemini_values, width, label='Platform Gemini', 
                       color=gemini_color, alpha=0.8)
        bars2 = ax.bar(x + width/2, gpt2_values, width, label='Model LLM-Judol (GPT-2)', 
                       color=gpt2_color, alpha=0.8)

        ax.set_title('Confidence Score Distribution - Comparison', fontweight='bold', fontsize=14)
        ax.set_ylabel('Confidence Score', fontsize=11)
        ax.set_xlabel('Metrics', fontsize=11)
        ax.set_xticks(x)
        ax.set_xticklabels(metrics)
        
        # 1. Geser Legend ke Tengah Atas (Horizontal)
        ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.0), ncol=2, framealpha=0.9)
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        
        # 2. Tambah Label Angka & Ruang Kosong (Headroom)
        max_height = 0
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                if height > max_height: max_height = height
                
                # Tulis angka di atas batang
                ax.text(bar.get_x() + bar.get_width()/2., height,
                        f'{height:.3f}', ha='center', va='bottom', 
                        fontsize=9, fontweight='bold')
        
        # Set batas atas Y lebih tinggi 25% dari data tertinggi biar legend gak nabrak
        # Karena confidence max 1.0, kita set minimal 1.25
        limit_y = max(max_height * 1.25, 1.2) 
        ax.set_ylim(0, limit_y)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / '1_confidence_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("Created: 1_confidence_distribution.png")
    
    def plot_latency_distribution(self):
        """Plot distribusi latency - comparison chart"""
        if not self.distribution:
            print("Skipping latency distribution - no data")
            return
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        gemini_lat = self.distribution['latency']['gemini']
        gpt2_lat = self.distribution['latency']['gpt2']
        
        # Data untuk comparison
        metrics = ['Mean', 'Median', 'Min', 'Max']
        gemini_values = [gemini_lat['mean'], gemini_lat['median'],
                        gemini_lat['min'], gemini_lat['max']]
        gpt2_values = [gpt2_lat['mean'], gpt2_lat['median'],
                      gpt2_lat['min'], gpt2_lat['max']]
        
        x = np.arange(len(metrics))
        width = 0.35
        
        # Bars side by side - Gemini biru, GPT-2 hijau
        bars1 = ax.bar(x - width/2, gemini_values, width, label='Platform Gemini', 
                      color='#4285F4', alpha=0.8)
        bars2 = ax.bar(x + width/2, gpt2_values, width, label='Model LLM-Judol (GPT-2)', 
                      color='#34A853', alpha=0.8)
        
        ax.set_title('Latency Distribution - Comparison (ms)', fontweight='bold', fontsize=14)
        ax.set_ylabel('Latency (milliseconds)', fontsize=11)
        ax.set_xlabel('Metrics', fontsize=11)
        ax.set_xticks(x)
        ax.set_xticklabels(metrics)
        ax.legend(fontsize=10)
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        
        # Add value labels
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + max(gemini_values + gpt2_values) * 0.02,
                       f'{height:.1f}', ha='center', va='bottom', 
                       fontsize=9, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / '2_latency_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("Created: 2_latency_distribution.png")
    
    def plot_agreement_analysis(self):
        """Plot analisis kesepakatan"""
        if not self.agreement:
            print("Skipping agreement analysis - no data")
            return
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # Pie chart
        sizes = [self.agreement['agreements'], self.agreement['disagreements']]
        labels = [f"Setuju\n{self.agreement['agreements']} komentar",
                 f"Tidak Setuju\n{self.agreement['disagreements']} komentar"]
        colors = ['#34A853', '#EA4335']
        explode = (0.05, 0.05)
        
        ax1.pie(sizes, explode=explode, labels=labels, colors=colors,
               autopct='%1.1f%%', shadow=True, startangle=90,
               textprops={'fontsize': 11, 'fontweight': 'bold'})
        ax1.set_title(f'Agreement Rate: {self.agreement["agreement_rate"]:.2f}%',
                     fontsize=14, fontweight='bold', pad=20)
        
        # Bar chart details
        if 'details' in self.agreement and self.agreement['details']:
            details = self.agreement['details']
            agreed_count = sum(1 for d in details if d['agreed'])
            disagreed_count = len(details) - agreed_count
            
            categories = ['Total\nKlasifikasi', 'Kesepakatan\n(Agreement)', 'Ketidaksepakatan\n(Disagreement)']
            values = [len(details), agreed_count, disagreed_count]
            colors = ['#4285F4', '#34A853', '#EA4335']
            
            bars = ax2.bar(categories, values, color=colors)
            ax2.set_title('Breakdown Analisis Kesepakatan', fontweight='bold')
            ax2.set_ylabel('Jumlah Komentar')
            ax2.grid(axis='y', alpha=0.3)
            
            for bar, val in zip(bars, values):
                height = bar.get_height()
                ax2.text(bar.get_x() + bar.get_width()/2., height + max(values) * 0.02,
                        f'{val}', ha='center', va='bottom', fontweight='bold', fontsize=11)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / '3_agreement_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("Created: 3_agreement_analysis.png")
    
    def plot_classification_comparison(self):
        """Plot perbandingan klasifikasi"""
        if not self.comparative:
            print("Skipping classification comparison - no data")
            return
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))
        
        # Color scheme
        gemini_color = '#4285F4'
        gpt2_color = '#34A853'

        gemini_data = self.final_report.get('comparison', {}).get('gemini', {})
        gpt2_data = self.final_report.get('comparison', {}).get('gpt2', {})

        total_gemini = sum(1 for r in self.gemini_results if r.get('success', False))
        total_gpt2 = sum(1 for r in self.gpt2_results if r.get('success', False))

        gemini_judi = gemini_data.get('classification', {}).get('judi_online', 0)
        gpt2_judi = gpt2_data.get('classification', {}).get('judi_online', 0)

        gemini_bukan = total_gemini - gemini_judi
        gpt2_bukan = total_gpt2 - gpt2_judi

        def add_labels_and_space(ax, bars_list):
            max_height = 0
            for bars in bars_list:
                for bar in bars:
                    height = bar.get_height()
                    if height > max_height: max_height = height
                    ax.text(bar.get_x() + bar.get_width()/2., height,
                            f'{float(height):.2f}' if height % 1 != 0 else f'{int(height)}',
                            ha='center', va='bottom', fontsize=9, fontweight='bold')
            # Tambah ruang kosong 20% di atas biar legend gak nabrak
            ax.set_ylim(0, max_height * 1.25)

        # 1. Classification counts
        x = np.arange(2)
        width = 0.35
        
        bars1 = ax1.bar(x - width/2, [gemini_judi, gemini_bukan],
                       width, label='Platform Gemini', color=gemini_color, alpha=0.8)
        bars2 = ax1.bar(x + width/2, [gpt2_judi, gpt2_bukan],
                       width, label='Model LLM-Judol', color=gpt2_color, alpha=0.8)
        
        ax1.set_title('Perbandingan Hasil Klasifikasi', fontweight='bold', fontsize=12)
        ax1.set_ylabel('Jumlah Komentar')
        ax1.set_xticks(x)
        ax1.set_xticklabels(['Judi Online', 'Bukan Judi Online'])
        ax1.legend(loc='upper right', framealpha=0.9) # Legend transparan dikit
        ax1.grid(axis='y', alpha=0.3, linestyle='--')
        add_labels_and_space(ax1, [bars1, bars2])
        
        # 2. Detection Rate Comparison
        gemini_rate = gemini_data.get('classification', {}).get('detection_rate', 0)
        gpt2_rate = gpt2_data.get('classification', {}).get('detection_rate', 0)
        
        detection_rates = [gemini_rate, gpt2_rate]
        bars = ax2.bar(['Platform Gemini', 'Model LLM-Judol'], detection_rates, 
                       color=[gemini_color, gpt2_color], alpha=0.8)
        ax2.set_title('Detection Rate Comparison', fontweight='bold', fontsize=12)
        ax2.set_ylabel('Detection Rate (%)')
        ax2.grid(axis='y', alpha=0.3, linestyle='--')
        add_labels_and_space(ax2, [bars])
        
        # 3. Confidence Comparison
        gemini_conf_mean = gemini_data.get('confidence', {}).get('mean', 0)
        gpt2_conf_mean = gpt2_data.get('confidence', {}).get('mean', 0)
        
        bars = ax3.bar(['Platform Gemini', 'Model LLM-Judol'], [gemini_conf_mean, gpt2_conf_mean],
                      color=[gemini_color, gpt2_color], alpha=0.8)
        ax3.set_title('Average Confidence Score', fontweight='bold', fontsize=12)
        ax3.set_ylabel('Confidence Score')
        ax3.grid(axis='y', alpha=0.3, linestyle='--')
        add_labels_and_space(ax3, [bars])
        
        # 4. Latency Comparison
        gemini_lat_mean = gemini_data.get('latency', {}).get('mean', 0)
        gpt2_lat_mean = gpt2_data.get('latency', {}).get('mean', 0)
        
        bars = ax4.bar(['Platform Gemini', 'Model LLM-Judol'], [gemini_lat_mean, gpt2_lat_mean],
                      color=[gemini_color, gpt2_color], alpha=0.8)
        ax4.set_title('Average Latency (Lower is Better)', fontweight='bold', fontsize=12)
        ax4.set_ylabel('Latency (ms)')
        ax4.grid(axis='y', alpha=0.3, linestyle='--')
        add_labels_and_space(ax4, [bars])
        
        plt.tight_layout()
        plt.savefig(self.output_dir / '4_classification_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("Created: 4_classification_comparison.png")
    
    def plot_complete_dashboard(self):
        """Dashboard komprehensif"""
        if not self.final_report:
            print("Skipping dashboard - no data")
            return
        
        fig = plt.figure(figsize=(16, 12))
        gs = fig.add_gridspec(3, 3, hspace=0.4, wspace=0.3)
        
        # Title
        fig.suptitle('DASHBOARD ANALISIS KOMPARASI: GEMINI vs GPT-2 (LLM-JUDOL)\nDeteksi Konten Judi Online pada Komentar YouTube',
                    fontsize=16, fontweight='bold', y=0.98)

        gemini_data = self.final_report.get('comparison', {}).get('gemini', {})
        gpt2_data = self.final_report.get('comparison', {}).get('gpt2', {})

        total_gemini = sum(1 for r in self.gemini_results if r.get('success', False))
        total_gpt2 = sum(1 for r in self.gpt2_results if r.get('success', False))

        gemini_judi = gemini_data.get('classification', {}).get('judi_online', 0)
        gpt2_judi = gpt2_data.get('classification', {}).get('judi_online', 0)

        gemini_bukan = total_gemini - gemini_judi
        gpt2_bukan = total_gpt2 - gpt2_judi

        # Helper function untuk label + space (REUSABLE)
        def add_labels_and_space(ax, bars_list):
            max_height = 0
            for bars in bars_list:
                for bar in bars:
                    height = bar.get_height()
                    if height > max_height: max_height = height
                    ax.text(bar.get_x() + bar.get_width()/2., height,
                            f'{float(height):.2f}' if height % 1 != 0 else f'{int(height)}',
                            ha='center', va='bottom', fontsize=9, fontweight='bold')
            ax.set_ylim(0, max_height * 1.35) # Kasih space 35% di atas biar lega

        # 1. Rule-based Statistics
        ax1 = fig.add_subplot(gs[0, 0])
        stats = self.final_report.get('statistics', {}).get('rule_based', {})
        labels = ['Judi\nOnline', 'Bukan\nJudi', 'Ambigu']
        values = [stats.get('judi_online', 0), stats.get('bukan_judi_online', 0), stats.get('ambigu', 0)]
        colors = ['#EA4335', '#34A853', '#FBBC04']
        
        bars = ax1.bar(labels, values, color=colors)
        ax1.set_title('Rule-Based Classification', fontweight='bold')
        ax1.set_ylabel('Jumlah Komentar')
        ax1.grid(axis='y', alpha=0.3)
        add_labels_and_space(ax1, [bars])
        
        # 2. Agreement Rate
        ax2 = fig.add_subplot(gs[0, 1])
        agreement_data = self.final_report.get('agreement', {})
        agreement_rate = agreement_data.get('agreement_rate', 0)
        
        sizes = [agreement_rate, 100 - agreement_rate]
        colors = ['#34A853', '#E8E8E8']
        wedges, texts, autotexts = ax2.pie(sizes, colors=colors, autopct='%1.1f%%',
                                          startangle=90, textprops={'fontweight': 'bold'})
        ax2.set_title(f'Agreement Rate\n{agreement_rate:.1f}%', fontweight='bold')
        
        # 3. Model Performance Summary
        ax3 = fig.add_subplot(gs[0, 2])
        ax3.axis('off')
        
        g_lat = gemini_data.get('latency', {}).get('mean', 0)
        gpt_lat = gpt2_data.get('latency', {}).get('mean', 0)
        faster = "GPT-2 (Lokal)" if gpt_lat < g_lat else "Gemini (Cloud)"
        
        g_conf = gemini_data.get('confidence', {}).get('mean', 0)
        gpt_conf = gpt2_data.get('confidence', {}).get('mean', 0)
        confident = "GPT-2" if gpt_conf > g_conf else "Gemini"
        
        summary_text = f"""
PERFORMA MODEL SUMMARY

Platform Gemini (Blue):
   Judi Online: {gemini_judi}
   Bukan Judi: {gemini_bukan}
   Avg Conf: {g_conf:.3f}
   Avg Latency: {g_lat:.1f}ms

Model LLM-Judol (Green):
   Judi Online: {gpt2_judi}
   Bukan Judi: {gpt2_bukan}
   Avg Conf: {gpt_conf:.3f}
   Avg Latency: {gpt_lat:.1f}ms

KESIMPULAN:
✓ Tercepat: {faster}
✓ Confidence: {confident}
✓ Konsistensi: {agreement_rate:.1f}%
        """
        ax3.text(0.1, 0.5, summary_text.strip(), transform=ax3.transAxes,
                fontsize=9, verticalalignment='center', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
        
        # 4. Classification Distribution (Legend di sini)
        ax4 = fig.add_subplot(gs[1, :])
        gemini_color = '#4285F4'
        gpt2_color = '#34A853'
        
        x = np.arange(2)
        width = 0.35
        gemini_vals = [gemini_judi, gemini_bukan]
        gpt2_vals = [gpt2_judi, gpt2_bukan]
        
        bars1 = ax4.bar(x - width/2, gemini_vals, width, label='Gemini', color=gemini_color, alpha=0.8)
        bars2 = ax4.bar(x + width/2, gpt2_vals, width, label='GPT-2 (Judol)', color=gpt2_color, alpha=0.8)
        
        ax4.set_title('Perbandingan Klasifikasi LLM (Data Ambigu)', fontweight='bold', fontsize=12)
        ax4.set_ylabel('Jumlah Komentar')
        ax4.set_xticks(x)
        ax4.set_xticklabels(['Judi Online', 'Bukan Judi Online'])
        # Legend dipindah ke atas tengah
        ax4.legend(loc='upper center', bbox_to_anchor=(0.5, 1.0), ncol=2, framealpha=0.9)
        ax4.grid(axis='y', alpha=0.3)
        add_labels_and_space(ax4, [bars1, bars2])
        
        # 5. Confidence
        ax5 = fig.add_subplot(gs[2, 0])
        g_conf_d = gemini_data.get('confidence', {})
        gpt_conf_d = gpt2_data.get('confidence', {})
        
        vals_g = [g_conf_d.get('mean', 0), g_conf_d.get('median', 0)]
        vals_gpt = [gpt_conf_d.get('mean', 0), gpt_conf_d.get('median', 0)]
        x = np.arange(2)
        
        bars1 = ax5.bar(x - width/2, vals_g, width, label='Gemini', color=gemini_color, alpha=0.8)
        bars2 = ax5.bar(x + width/2, vals_gpt, width, label='GPT-2', color=gpt2_color, alpha=0.8)
        
        ax5.set_title('Confidence Score', fontweight='bold')
        ax5.set_xticks(x)
        ax5.set_xticklabels(['Mean', 'Median'])
        ax5.legend(loc='upper right', fontsize=8) # Legend kecil di pojok
        ax5.grid(axis='y', alpha=0.3)
        add_labels_and_space(ax5, [bars1, bars2])
        
        # 6. Latency
        ax6 = fig.add_subplot(gs[2, 1])
        g_lat_d = gemini_data.get('latency', {})
        gpt_lat_d = gpt2_data.get('latency', {})
        
        vals_g = [g_lat_d.get('mean', 0), g_lat_d.get('median', 0)]
        vals_gpt = [gpt_lat_d.get('mean', 0), gpt_lat_d.get('median', 0)]
        
        bars1 = ax6.bar(x - width/2, vals_g, width, label='Gemini', color=gemini_color, alpha=0.8)
        bars2 = ax6.bar(x + width/2, vals_gpt, width, label='GPT-2', color=gpt2_color, alpha=0.8)
        
        ax6.set_title('Latency (ms)', fontweight='bold')
        ax6.set_xticks(x)
        ax6.set_xticklabels(['Mean', 'Median'])
        ax6.grid(axis='y', alpha=0.3)
        add_labels_and_space(ax6, [bars1, bars2])
        
        # 7. Metadata
        ax7 = fig.add_subplot(gs[2, 2])
        ax7.axis('off')
        metadata_text = f"""
METADATA
Total Data: {self.final_report.get('statistics', {}).get('total', 0)}
Rule-Based:
  Judi: {stats.get('judi_online', 0)}
  Bukan: {stats.get('bukan_judi_online', 0)}
  Ambigu: {stats.get('ambigu', 0)}
  
Date: {self.final_report.get('metadata', {}).get('generated_at', '')[:10]}
        """
        ax7.text(0.1, 0.5, metadata_text.strip(), transform=ax7.transAxes,
                fontsize=9, verticalalignment='center', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.2))
        
        plt.savefig(self.output_dir / '5_complete_dashboard.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("Created: 5_complete_dashboard.png")
    
    def plot_research_flow(self):
        """Diagram alur penelitian"""
        fig, ax = plt.subplots(figsize=(14, 10))
        ax.axis('off')
        
        # Title
        ax.text(0.5, 0.95, 'ALUR METODOLOGI PENELITIAN', 
               ha='center', va='top', fontsize=16, fontweight='bold')
        ax.text(0.5, 0.92, 'Deteksi Konten Judi Online: Cascading Filtering Approach',
               ha='center', va='top', fontsize=11, style='italic')
        
        # Define stages
        stages = [
            {
                'y': 0.85,
                'title': '1. INPUT DATA',
                'text': 'Komentar YouTube\n(JSON/CSV/Text)',
                'color': '#E3F2FD',
                'border': '#2196F3'
            },
            {
                'y': 0.75,
                'title': '2. PRA-PEMROSESAN',
                'text': 'Case Folding → Leet Speak Conversion\n→ Tokenisasi → Stopword Removal',
                'color': '#E8F5E9',
                'border': '#4CAF50'
            },
            {
                'y': 0.65,
                'title': '3. RULE-BASED FILTERING',
                'text': 'Regex Pattern Matching + Fuzzy Matching',
                'color': '#FFF3E0',
                'border': '#FF9800'
            },
            {
                'y': 0.52,
                'title': '   KLASIFIKASI AWAL',
                'text': '   Skor Tinggi → JUDI ONLINE\n   Skor Rendah → BUKAN JUDI\n   Skor Menengah → DATA AMBIGU',
                'color': '#FCE4EC',
                'border': '#E91E63'
            },
            {
                'y': 0.38,
                'title': '4. LLM CLASSIFICATION (Data Ambigu)',
                'text': 'Platform Gemini (Prompt Engineering)\nvs\nModel LLM-Judol (GPT-2 Fine-Tuned)',
                'color': '#F3E5F5',
                'border': '#9C27B0'
            },
            {
                'y': 0.25,
                'title': '5. ANALISIS PERFORMA',
                'text': 'Agreement Rate | Confidence Distribution | Latency Analysis',
                'color': '#E0F2F1',
                'border': '#009688'
            },
            {
                'y': 0.15,
                'title': '6. PELAPORAN',
                'text': 'Grafik Visualisasi + Tabel Komparasi + Kesimpulan',
                'color': '#FFF9C4',
                'border': '#FBC02D'
            }
        ]
        
        # Draw stages
        for stage in stages:
            # Box
            rect = mpatches.FancyBboxPatch(
                (0.15, stage['y'] - 0.05), 0.7, 0.08,
                boxstyle="round,pad=0.01",
                facecolor=stage['color'],
                edgecolor=stage['border'],
                linewidth=2
            )
            ax.add_patch(rect)
            
            # Title
            ax.text(0.5, stage['y'] + 0.015, stage['title'],
                   ha='center', va='center', fontsize=11, fontweight='bold')
            
            # Text
            ax.text(0.5, stage['y'] - 0.015, stage['text'],
                   ha='center', va='center', fontsize=9, style='italic')
            
            # Arrow (except last)
            if stage['y'] > 0.2:
                ax.annotate('', xy=(0.5, stage['y'] - 0.06), xytext=(0.5, stage['y'] - 0.04),
                           arrowprops=dict(arrowstyle='->', lw=2, color='#424242'))
        
        # Add legend for cascading
        legend_y = 0.05
        ax.text(0.15, legend_y, 'KEY INSIGHT:', fontsize=10, fontweight='bold')
        ax.text(0.15, legend_y - 0.02, 
               'Cascading approach: Rule-based handles simple cases efficiently,\nLLM only processes ambiguous cases to minimize API costs and latency.',
               fontsize=8, style='italic')
        
        plt.savefig(self.output_dir / '6_research_flow_diagram.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("Created: 6_research_flow_diagram.png")


def main():
    """Main function"""
    print("=" * 70)
    print("RESEARCH RESULTS VISUALIZER")
    print("=" * 70)
    
    visualizer = ResearchVisualizer()
    visualizer.create_all_visualizations()
    
    print("\n" + "=" * 70)
    print("VISUALIZATION COMPLETE!")
    print("=" * 70)
    print("\nGenerated files:")
    print("  1. 1_confidence_distribution.png")
    print("  2. 2_latency_distribution.png")
    print("  3. 3_agreement_analysis.png")
    print("  4. 4_classification_comparison.png")
    print("  5. 5_complete_dashboard.png")
    print("  6. 6_research_flow_diagram.png")
    print("\n Location: visualizations/")
    print("=" * 70)


if __name__ == "__main__":
    main()
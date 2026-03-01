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
    """Visualizer untuk hasil penelitian - membaca dari results/aggregate_results.json"""

    def __init__(self, results_dir="results", output_dir="visualizations"):
        self.results_dir = Path(results_dir)
        self.output_dir  = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.load_data()

    # ------------------------------------------------------------------ #
    # DATA LOADING                                                         #
    # ------------------------------------------------------------------ #

    def load_data(self):
        """Load aggregate_results.json yang dihasilkan detector.py"""
        print("Loading data from aggregate_results.json ...")

        agg_path = self.results_dir / "aggregate_results.json"
        if not agg_path.exists():
            print(f"[ERROR] File tidak ditemukan: {agg_path}")
            print("        Pastikan detector.py sudah dijalankan terlebih dahulu.")
            self.aggregate = None
            self.videos    = []
            self.summary   = {}
            self.averages  = {}
            return

        with open(agg_path, 'r', encoding='utf-8') as f:
            self.aggregate = json.load(f)

        self.videos   = self.aggregate.get('per_video_results', [])
        self.summary  = self.aggregate.get('summary', {})
        self.averages = self.aggregate.get('averages', {})

        print(f"  Loaded {len(self.videos)} video(s).")
        print("Data loaded successfully!")

    # ------------------------------------------------------------------ #
    # HELPERS                                                              #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _add_bar_labels(ax, bars_list, fmt=None):
        """Tulis angka di atas setiap batang dan tambah headroom."""
        max_h = 0
        for bars in bars_list:
            for bar in bars:
                h = bar.get_height()
                if h > max_h:
                    max_h = h
                if fmt:
                    label = fmt(h)
                elif h != 0 and h % 1 != 0:
                    label = f'{h:.4f}'
                else:
                    label = f'{int(h)}'
                ax.text(bar.get_x() + bar.get_width() / 2., h,
                        label, ha='center', va='bottom',
                        fontsize=9, fontweight='bold')
        ax.set_ylim(0, max(max_h * 1.3, 0.001))

    # ------------------------------------------------------------------ #
    # 1. CONFIDENCE DISTRIBUTION                                           #
    # ------------------------------------------------------------------ #

    def plot_confidence_distribution(self):
        """Bar chart: confidence score Gemini vs GPT per video."""
        if not self.videos:
            print("Skipping confidence distribution - no data")
            return

        video_ids   = [v['video_id'] for v in self.videos]
        gemini_vals = [v.get('gemini_avg_confidence', 0) for v in self.videos]
        gpt_vals    = [v.get('gpt_avg_confidence', 0) for v in self.videos]

        x     = np.arange(len(video_ids))
        width = 0.35

        fig, ax = plt.subplots(figsize=(max(10, len(video_ids) * 2), 6))
        bars1 = ax.bar(x - width / 2, gemini_vals, width,
                       label='Platform Gemini', color='#4285F4', alpha=0.8)
        bars2 = ax.bar(x + width / 2, gpt_vals, width,
                       label='Platform GPT (gpt-4o-mini)', color='#34A853', alpha=0.8)

        ax.set_title('Rata-rata Confidence Score per Video', fontweight='bold', fontsize=14)
        ax.set_ylabel('Confidence Score (0-1)')
        ax.set_xlabel('Video ID')
        ax.set_xticks(x)
        ax.set_xticklabels(video_ids, rotation=15, ha='right')
        ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.0), ncol=2, framealpha=0.9)
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        self._add_bar_labels(ax, [bars1, bars2], fmt=lambda h: f'{h:.3f}')

        plt.tight_layout()
        plt.savefig(self.output_dir / '1_confidence_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("Created: 1_confidence_distribution.png")

    # ------------------------------------------------------------------ #
    # 2. LATENCY DISTRIBUTION                                              #
    # ------------------------------------------------------------------ #

    def plot_latency_distribution(self):
        """Bar chart: latency Gemini vs GPT per video."""
        if not self.videos:
            print("Skipping latency distribution - no data")
            return

        video_ids   = [v['video_id'] for v in self.videos]
        gemini_vals = [v.get('gemini_avg_latency', 0) for v in self.videos]
        gpt_vals    = [v.get('gpt_avg_latency', 0) for v in self.videos]

        x     = np.arange(len(video_ids))
        width = 0.35

        fig, ax = plt.subplots(figsize=(max(10, len(video_ids) * 2), 6))
        bars1 = ax.bar(x - width / 2, gemini_vals, width,
                       label='Platform Gemini', color='#4285F4', alpha=0.8)
        bars2 = ax.bar(x + width / 2, gpt_vals, width,
                       label='Platform GPT (gpt-4o-mini)', color='#34A853', alpha=0.8)

        ax.set_title('Rata-rata Latency per Video (Lower is Better)', fontweight='bold', fontsize=14)
        ax.set_ylabel('Latency (ms)')
        ax.set_xlabel('Video ID')
        ax.set_xticks(x)
        ax.set_xticklabels(video_ids, rotation=15, ha='right')
        ax.legend(loc='upper left', fontsize=10)
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        self._add_bar_labels(ax, [bars1, bars2], fmt=lambda h: f'{h:.1f}ms')

        plt.tight_layout()
        plt.savefig(self.output_dir / '2_latency_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("Created: 2_latency_distribution.png")

    # ------------------------------------------------------------------ #
    # 3. AGREEMENT ANALYSIS                                                #
    # ------------------------------------------------------------------ #

    def plot_agreement_analysis(self):
        """Pie chart agreement + bar breakdown per video."""
        if not self.videos:
            print("Skipping agreement analysis - no data")
            return

        total_agree    = sum(v.get('agreement_count', 0) for v in self.videos)
        total_disagree = sum(v.get('disagreement_count', 0) for v in self.videos)
        total_llm      = total_agree + total_disagree
        rate           = (total_agree / total_llm * 100) if total_llm else 0

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

        # Pie
        sizes   = [total_agree, total_disagree]
        labels  = [f'Setuju\n{total_agree} komentar', f'Tidak Setuju\n{total_disagree} komentar']
        colors  = ['#34A853', '#EA4335']
        explode = (0.05, 0.05)
        ax1.pie(sizes, explode=explode, labels=labels, colors=colors,
                autopct='%1.1f%%', shadow=False, startangle=90,
                textprops={'fontsize': 11, 'fontweight': 'bold'})
        ax1.set_title(f'Agreement Rate: {rate:.2f}%', fontsize=14, fontweight='bold', pad=20)

        # Bar per video
        video_ids     = [v['video_id'] for v in self.videos]
        agree_vals    = [v.get('agreement_count', 0) for v in self.videos]
        disagree_vals = [v.get('disagreement_count', 0) for v in self.videos]
        x     = np.arange(len(video_ids))
        width = 0.35

        bars1 = ax2.bar(x - width / 2, agree_vals, width, label='Sepakat', color='#34A853', alpha=0.8)
        bars2 = ax2.bar(x + width / 2, disagree_vals, width, label='Tidak Sepakat', color='#EA4335', alpha=0.8)
        ax2.set_title('Breakdown Agreement per Video', fontweight='bold')
        ax2.set_ylabel('Jumlah Komentar')
        ax2.set_xticks(x)
        ax2.set_xticklabels(video_ids, rotation=15, ha='right')
        ax2.legend()
        ax2.grid(axis='y', alpha=0.3)
        self._add_bar_labels(ax2, [bars1, bars2])

        plt.tight_layout()
        plt.savefig(self.output_dir / '3_agreement_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("Created: 3_agreement_analysis.png")

    # ------------------------------------------------------------------ #
    # 4. CLASSIFICATION COMPARISON                                         #
    # ------------------------------------------------------------------ #

    def plot_classification_comparison(self):
        """2x2 chart: klasifikasi, detection rate, confidence, latency."""
        if not self.videos:
            print("Skipping classification comparison - no data")
            return

        gemini_color = '#4285F4'
        gpt_color    = '#34A853'

        gemini_judi  = sum(v.get('gemini_judi', 0) for v in self.videos)
        gemini_bukan = sum(v.get('gemini_bukan_judi', 0) for v in self.videos)
        gpt_judi     = sum(v.get('gpt_judi', 0) for v in self.videos)
        gpt_bukan    = sum(v.get('gpt_bukan_judi', 0) for v in self.videos)

        total_gemini = gemini_judi + gemini_bukan
        total_gpt    = gpt_judi + gpt_bukan
        gemini_rate  = (gemini_judi / total_gemini * 100) if total_gemini else 0
        gpt_rate     = (gpt_judi / total_gpt * 100) if total_gpt else 0

        g_conf = self.averages.get('gemini_confidence', 0)
        p_conf = self.averages.get('gpt_confidence', 0)
        g_lat  = self.averages.get('gemini_latency_ms', 0)
        p_lat  = self.averages.get('gpt_latency_ms', 0)

        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))

        def add_labels(ax, bars_list, fmt=None):
            max_h = 0
            for bars in bars_list:
                for bar in bars:
                    h = bar.get_height()
                    if h > max_h: max_h = h
                    label = fmt(h) if fmt else (f'{h:.2f}' if h % 1 != 0 else f'{int(h)}')
                    ax.text(bar.get_x() + bar.get_width() / 2., h,
                            label, ha='center', va='bottom', fontsize=9, fontweight='bold')
            ax.set_ylim(0, max(max_h * 1.25, 0.001))

        # 1. Classification counts
        x = np.arange(2)
        width = 0.35
        bars1 = ax1.bar(x - width/2, [gemini_judi, gemini_bukan], width,
                        label='Platform Gemini', color=gemini_color, alpha=0.8)
        bars2 = ax1.bar(x + width/2, [gpt_judi, gpt_bukan], width,
                        label='Platform GPT (gpt-4o-mini)', color=gpt_color, alpha=0.8)
        ax1.set_title('Perbandingan Hasil Klasifikasi', fontweight='bold', fontsize=12)
        ax1.set_ylabel('Jumlah Komentar')
        ax1.set_xticks(x)
        ax1.set_xticklabels(['Judi Online', 'Bukan Judi Online'])
        ax1.legend(loc='upper right', framealpha=0.9)
        ax1.grid(axis='y', alpha=0.3, linestyle='--')
        add_labels(ax1, [bars1, bars2])

        # 2. Detection rate
        bars = ax2.bar(['Platform Gemini', 'Platform GPT\n(gpt-4o-mini)'],
                       [gemini_rate, gpt_rate], color=[gemini_color, gpt_color], alpha=0.8)
        ax2.set_title('Tingkat Deteksi Judi (Detection Rate)', fontweight='bold', fontsize=12)
        ax2.set_ylabel('Detection Rate (%)')
        ax2.grid(axis='y', alpha=0.3, linestyle='--')
        add_labels(ax2, [bars], fmt=lambda h: f'{h:.1f}%')

        # 3. Confidence comparison
        bars = ax3.bar(['Platform Gemini', 'Platform GPT\n(gpt-4o-mini)'],
                       [g_conf, p_conf], color=[gemini_color, gpt_color], alpha=0.8)
        ax3.set_title('Average Confidence Score', fontweight='bold', fontsize=12)
        ax3.set_ylabel('Confidence Score')
        ax3.grid(axis='y', alpha=0.3, linestyle='--')
        add_labels(ax3, [bars], fmt=lambda h: f'{h:.4f}')

        # 4. Latency comparison
        bars = ax4.bar(['Platform Gemini', 'Platform GPT\n(gpt-4o-mini)'],
                       [g_lat, p_lat], color=[gemini_color, gpt_color], alpha=0.8)
        ax4.set_title('Average Latency (Lower is Better)', fontweight='bold', fontsize=12)
        ax4.set_ylabel('Latency (ms)')
        ax4.grid(axis='y', alpha=0.3, linestyle='--')
        add_labels(ax4, [bars], fmt=lambda h: f'{h:.1f}ms')

        plt.tight_layout()
        plt.savefig(self.output_dir / '4_classification_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("Created: 4_classification_comparison.png")

    # ------------------------------------------------------------------ #
    # 5. COST ANALYSIS  (BARU)                                             #
    # ------------------------------------------------------------------ #

    def plot_cost_analysis(self):
        """
        Visualisasi biaya API - 4 panel:
          [1] Grouped bar: cost Gemini / GPT / Total per video
          [2] Pie chart: proporsi total cost Gemini vs GPT
          [3] Bar chart: total token usage (input vs output) per model
          [4] Stacked bar: cost breakdown (input cost vs output cost) per model
        """
        if not self.videos:
            print("Skipping cost analysis - no data")
            return

        gemini_color = '#4285F4'
        gpt_color    = '#34A853'

        video_ids      = [v['video_id'] for v in self.videos]
        gemini_costs   = [v.get('gemini_total_cost_usd', 0.0) for v in self.videos]
        gpt_costs      = [v.get('gpt_total_cost_usd', 0.0) for v in self.videos]
        total_costs    = [v.get('total_cost_usd', 0.0) for v in self.videos]

        gemini_in_tok  = [v.get('gemini_total_input_tokens', 0) for v in self.videos]
        gemini_out_tok = [v.get('gemini_total_output_tokens', 0) for v in self.videos]
        gpt_in_tok     = [v.get('gpt_total_input_tokens', 0) for v in self.videos]
        gpt_out_tok    = [v.get('gpt_total_output_tokens', 0) for v in self.videos]

        total_gemini_cost = sum(gemini_costs)
        total_gpt_cost    = sum(gpt_costs)
        grand_total_cost  = total_gemini_cost + total_gpt_cost

        total_gemini_in   = sum(gemini_in_tok)
        total_gemini_out  = sum(gemini_out_tok)
        total_gpt_in      = sum(gpt_in_tok)
        total_gpt_out     = sum(gpt_out_tok)

        # Cost breakdown: input cost vs output cost per model
        # Price: input $0.15/1M, output $0.60/1M
        g_in_cost  = (total_gemini_in  / 1_000_000) * 0.15
        g_out_cost = (total_gemini_out / 1_000_000) * 0.60
        p_in_cost  = (total_gpt_in     / 1_000_000) * 0.15
        p_out_cost = (total_gpt_out    / 1_000_000) * 0.60

        fig = plt.figure(figsize=(16, 13))
        gs  = fig.add_gridspec(2, 2, hspace=0.45, wspace=0.35)
        fig.suptitle(
            'ANALISIS BIAYA API: Gemini (gemini-2.5-flash) vs GPT (gpt-4o-mini)\n'
            'Harga: $0.15 / 1M input tokens  |  $0.60 / 1M output tokens',
            fontsize=14, fontweight='bold', y=0.99
        )

        # ---- Panel 1: Cost per video (grouped bar) -------------------
        ax1   = fig.add_subplot(gs[0, :])
        x     = np.arange(len(video_ids))
        width = 0.25

        bars_g = ax1.bar(x - width, gemini_costs, width,
                         label='Gemini (gemini-2.5-flash)', color=gemini_color, alpha=0.85)
        bars_p = ax1.bar(x,         gpt_costs,    width,
                         label='GPT (gpt-4o-mini)',         color=gpt_color,    alpha=0.85)
        bars_t = ax1.bar(x + width, total_costs,  width,
                         label='Total',                     color='#9E9E9E',    alpha=0.85)

        ax1.set_title('Biaya API per Video (USD)', fontweight='bold', fontsize=13)
        ax1.set_ylabel('Biaya (USD)')
        ax1.set_xlabel('Video ID')
        ax1.set_xticks(x)
        ax1.set_xticklabels(video_ids, rotation=15, ha='right')
        ax1.legend(loc='upper right')
        ax1.grid(axis='y', alpha=0.3, linestyle='--')

        for bars in [bars_g, bars_p, bars_t]:
            for bar in bars:
                h = bar.get_height()
                if h > 0:
                    ax1.text(bar.get_x() + bar.get_width() / 2., h,
                             f'${h:.5f}', ha='center', va='bottom',
                             fontsize=7.5, fontweight='bold')
        max_h = max(total_costs) if total_costs else 0.001
        ax1.set_ylim(0, max_h * 1.4)

        # ---- Panel 2: Pie chart proporsi cost ------------------------
        ax2 = fig.add_subplot(gs[1, 0])
        if grand_total_cost > 0:
            pie_sizes  = [total_gemini_cost, total_gpt_cost]
            pie_labels = [
                f'Gemini\n${total_gemini_cost:.5f}',
                f'GPT\n${total_gpt_cost:.5f}'
            ]
            pie_colors = [gemini_color, gpt_color]
            ax2.pie(
                pie_sizes, labels=pie_labels, colors=pie_colors,
                autopct='%1.1f%%', startangle=90, shadow=False,
                explode=(0.04, 0.04),
                textprops={'fontsize': 10, 'fontweight': 'bold'}
            )
            ax2.set_title(
                f'Proporsi Total Cost\n(Grand Total: ${grand_total_cost:.5f} USD)',
                fontweight='bold'
            )
        else:
            ax2.text(0.5, 0.5, 'Data cost belum tersedia\n(token count = 0)',
                     ha='center', va='center', fontsize=11, transform=ax2.transAxes)
            ax2.set_title('Proporsi Total Cost', fontweight='bold')
            ax2.axis('off')

        # ---- Panel 3: Token usage ------------------------------------
        ax3 = fig.add_subplot(gs[1, 1])
        models       = ['Gemini\nInput', 'Gemini\nOutput', 'GPT\nInput', 'GPT\nOutput']
        token_values = [total_gemini_in, total_gemini_out, total_gpt_in, total_gpt_out]
        bar_colors   = [gemini_color, '#7BB3F7', gpt_color, '#7BC89E']

        bars_tok = ax3.bar(models, token_values, color=bar_colors, alpha=0.85)
        ax3.set_title('Total Token Usage (semua video)', fontweight='bold')
        ax3.set_ylabel('Jumlah Token')
        ax3.grid(axis='y', alpha=0.3, linestyle='--')
        max_tok = max(token_values) if any(token_values) else 1
        for bar, val in zip(bars_tok, token_values):
            ax3.text(bar.get_x() + bar.get_width() / 2., bar.get_height(),
                     f'{val:,}', ha='center', va='bottom', fontsize=9, fontweight='bold')
        ax3.set_ylim(0, max_tok * 1.3)

        # ---- Footnote ------------------------------------------------
        fig.text(0.5, 0.01,
                 f'Total Gemini: ${total_gemini_cost:.6f} USD  |  Total GPT: ${total_gpt_cost:.6f} USD  |  '
                 f'GRAND TOTAL: ${grand_total_cost:.6f} USD',
                 ha='center', fontsize=10, fontweight='bold',
                 bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

        plt.savefig(self.output_dir / '5_cost_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("Created: 5_cost_analysis.png")

    # ------------------------------------------------------------------ #
    # 6. COMPLETE DASHBOARD                                                #
    # ------------------------------------------------------------------ #

    def plot_complete_dashboard(self):
        """Dashboard komprehensif termasuk panel cost."""
        if not self.videos:
            print("Skipping dashboard - no data")
            return

        gemini_color = '#4285F4'
        gpt_color    = '#34A853'

        gemini_judi  = sum(v.get('gemini_judi', 0) for v in self.videos)
        gemini_bukan = sum(v.get('gemini_bukan_judi', 0) for v in self.videos)
        gpt_judi     = sum(v.get('gpt_judi', 0) for v in self.videos)
        gpt_bukan    = sum(v.get('gpt_bukan_judi', 0) for v in self.videos)

        total_judi_rb   = self.summary.get('total_judi_online', 0)
        total_bukan_rb  = self.summary.get('total_bukan_judi', 0)
        total_ambigu_rb = self.summary.get('total_ambigu', 0)

        total_agree    = sum(v.get('agreement_count', 0) for v in self.videos)
        total_disagree = sum(v.get('disagreement_count', 0) for v in self.videos)
        total_llm      = total_agree + total_disagree
        agreement_rate = (total_agree / total_llm * 100) if total_llm else 0

        g_conf = self.averages.get('gemini_confidence', 0)
        p_conf = self.averages.get('gpt_confidence', 0)
        g_lat  = self.averages.get('gemini_latency_ms', 0)
        p_lat  = self.averages.get('gpt_latency_ms', 0)

        total_gemini_cost = self.summary.get('gemini_total_cost_usd', 0.0)
        total_gpt_cost    = self.summary.get('gpt_total_cost_usd', 0.0)
        grand_total_cost  = self.summary.get('total_cost_usd', 0.0)

        def add_labels(ax, bars_list, fmt=None):
            max_h = 0
            for bars in bars_list:
                for bar in bars:
                    h = bar.get_height()
                    if h > max_h: max_h = h
                    label = fmt(h) if fmt else (f'{h:.3f}' if h % 1 != 0 else f'{int(h)}')
                    ax.text(bar.get_x() + bar.get_width() / 2., h,
                            label, ha='center', va='bottom', fontsize=8, fontweight='bold')
            ax.set_ylim(0, max(max_h * 1.35, 0.001))

        fig = plt.figure(figsize=(18, 14))
        gs  = fig.add_gridspec(3, 3, hspace=0.5, wspace=0.35)
        fig.suptitle(
            'DASHBOARD ANALISIS KOMPARASI: Gemini vs GPT (gpt-4o-mini)\n'
            'Deteksi Konten Judi Online pada Komentar YouTube',
            fontsize=16, fontweight='bold', y=0.99
        )

        # --- Panel 1: Rule-based stats --------------------------------
        ax1 = fig.add_subplot(gs[0, 0])
        bars = ax1.bar(['Judi\nOnline', 'Bukan\nJudi', 'Ambigu'],
                       [total_judi_rb, total_bukan_rb, total_ambigu_rb],
                       color=['#EA4335', '#34A853', '#FBBC04'])
        ax1.set_title('Pencocokan Karakter\n(Regex & Fuzzy)', fontweight='bold')
        ax1.set_ylabel('Jumlah Komentar')
        ax1.grid(axis='y', alpha=0.3)
        add_labels(ax1, [bars])

        # --- Panel 2: Agreement pie -----------------------------------
        ax2 = fig.add_subplot(gs[0, 1])
        ax2.pie([agreement_rate, 100 - agreement_rate],
                colors=['#34A853', '#E8E8E8'],
                autopct='%1.1f%%', startangle=90, shadow=False,
                textprops={'fontweight': 'bold'})
        ax2.set_title(f'Agreement Rate\n{agreement_rate:.1f}%', fontweight='bold')

        # --- Panel 3: Summary text box --------------------------------
        ax3 = fig.add_subplot(gs[0, 2])
        ax3.axis('off')
        faster    = 'GPT' if p_lat < g_lat else 'Gemini'
        confident = 'GPT' if p_conf > g_conf else 'Gemini'
        cheaper   = 'GPT' if total_gpt_cost <= total_gemini_cost else 'Gemini'
        summary_text = (
            "PERFORMA MODEL SUMMARY\n\n"
            f"Platform Gemini:\n"
            f"  Judi Online : {gemini_judi}\n"
            f"  Bukan Judi  : {gemini_bukan}\n"
            f"  Avg Conf    : {g_conf:.3f}\n"
            f"  Avg Latency : {g_lat:.1f}ms\n"
            f"  Total Cost  : ${total_gemini_cost:.5f}\n\n"
            f"Platform GPT (gpt-4o-mini):\n"
            f"  Judi Online : {gpt_judi}\n"
            f"  Bukan Judi  : {gpt_bukan}\n"
            f"  Avg Conf    : {p_conf:.3f}\n"
            f"  Avg Latency : {p_lat:.1f}ms\n"
            f"  Total Cost  : ${total_gpt_cost:.5f}\n\n"
            f"KESIMPULAN:\n"
            f"  Tercepat       : {faster}\n"
            f"  Conf Tertinggi : {confident}\n"
            f"  Termurah       : {cheaper}\n"
            f"  Konsistensi    : {agreement_rate:.1f}%"
        )
        ax3.text(0.05, 0.98, summary_text, transform=ax3.transAxes,
                 fontsize=8.5, verticalalignment='top', fontfamily='monospace',
                 bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

        # --- Panel 4: LLM classification comparison -------------------
        ax4 = fig.add_subplot(gs[1, :])
        x     = np.arange(2)
        width = 0.35
        bars1 = ax4.bar(x - width/2, [gemini_judi, gemini_bukan], width,
                        label='Platform Gemini', color=gemini_color, alpha=0.8)
        bars2 = ax4.bar(x + width/2, [gpt_judi, gpt_bukan], width,
                        label='Platform GPT (gpt-4o-mini)', color=gpt_color, alpha=0.8)
        ax4.set_title('Perbandingan Klasifikasi LLM (Data Ambigu)', fontweight='bold', fontsize=12)
        ax4.set_ylabel('Jumlah Komentar')
        ax4.set_xticks(x)
        ax4.set_xticklabels(['Judi Online', 'Bukan Judi Online'])
        ax4.legend(loc='upper center', bbox_to_anchor=(0.5, 1.0), ncol=2, framealpha=0.9)
        ax4.grid(axis='y', alpha=0.3)
        add_labels(ax4, [bars1, bars2])

        # --- Panel 5: Confidence comparison ---------------------------
        ax5 = fig.add_subplot(gs[2, 0])
        bars_c = ax5.bar(['Gemini', 'GPT'], [g_conf, p_conf],
                         color=[gemini_color, gpt_color], alpha=0.8)
        ax5.set_title('Avg Confidence Score', fontweight='bold')
        ax5.grid(axis='y', alpha=0.3)
        add_labels(ax5, [bars_c], fmt=lambda h: f'{h:.4f}')

        # --- Panel 6: Latency comparison ------------------------------
        ax6 = fig.add_subplot(gs[2, 1])
        bars_l = ax6.bar(['Gemini', 'GPT'], [g_lat, p_lat],
                         color=[gemini_color, gpt_color], alpha=0.8)
        ax6.set_title('Avg Latency (ms)', fontweight='bold')
        ax6.grid(axis='y', alpha=0.3)
        add_labels(ax6, [bars_l], fmt=lambda h: f'{h:.1f}ms')

        # --- Panel 7: Cost comparison ---------------------------------
        ax7 = fig.add_subplot(gs[2, 2])
        bars_cost = ax7.bar(['Gemini', 'GPT', 'Total'],
                            [total_gemini_cost, total_gpt_cost, grand_total_cost],
                            color=[gemini_color, gpt_color, '#9E9E9E'], alpha=0.8)
        ax7.set_title('Total Biaya API (USD)', fontweight='bold')
        ax7.set_ylabel('USD')
        ax7.grid(axis='y', alpha=0.3)
        for bar in bars_cost:
            h = bar.get_height()
            ax7.text(bar.get_x() + bar.get_width() / 2., h,
                     f'${h:.5f}', ha='center', va='bottom', fontsize=8, fontweight='bold')
        ax7.set_ylim(0, max(grand_total_cost * 1.35, 0.001))

        plt.savefig(self.output_dir / '6_complete_dashboard.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("Created: 6_complete_dashboard.png")

    # ------------------------------------------------------------------ #
    # 7. RESEARCH FLOW DIAGRAM                                             #
    # ------------------------------------------------------------------ #

    def plot_research_flow(self):
        """Diagram alur penelitian."""
        fig, ax = plt.subplots(figsize=(14, 10))
        ax.axis('off')

        ax.text(0.5, 0.95, 'ALUR METODOLOGI PENELITIAN',
                ha='center', va='top', fontsize=16, fontweight='bold')
        ax.text(0.5, 0.92, 'Deteksi Konten Judi Online: Cascading Filtering Approach',
                ha='center', va='top', fontsize=11, style='italic')

        stages = [
            {'y': 0.85, 'title': '1. INPUT DATA',
             'text': 'Komentar YouTube (JSON)',
             'color': '#E3F2FD', 'border': '#2196F3'},
            {'y': 0.75, 'title': '2. PRA-PEMROSESAN',
             'text': 'Case Folding -> Leet Speak Conversion -> Tokenisasi -> Pembersihan Tanda Baca',
             'color': '#E8F5E9', 'border': '#4CAF50'},
            {'y': 0.65, 'title': '3. PENCOCOKAN KARAKTER (Regex & Fuzzy)',
             'text': 'Pattern Matching Sederhana Tanpa AI',
             'color': '#FFF3E0', 'border': '#FF9800'},
            {'y': 0.52, 'title': '   HASIL PENCOCOKAN AWAL',
             'text': '   Sangat Mirip -> JUDI ONLINE  |  Tidak Mirip -> BUKAN JUDI  |  Samar-samar -> DATA AMBIGU',
             'color': '#FCE4EC', 'border': '#E91E63'},
            {'y': 0.38, 'title': '4. KLASIFIKASI AI (Untuk Data Ambigu)',
             'text': 'Platform Gemini (gemini-2.5-flash)  vs  Platform GPT (gpt-4o-mini)',
             'color': '#F3E5F5', 'border': '#9C27B0'},
            {'y': 0.25, 'title': '5. ANALISIS PERFORMA & BIAYA',
             'text': 'Agreement Rate | Confidence Distribution | Latency | Cost per Token (USD)',
             'color': '#E0F2F1', 'border': '#009688'},
            {'y': 0.15, 'title': '6. PELAPORAN',
             'text': 'Grafik Visualisasi + Tabel Komparasi + Report Biaya API + Kesimpulan',
             'color': '#FFF9C4', 'border': '#FBC02D'},
        ]

        for stage in stages:
            rect = mpatches.FancyBboxPatch(
                (0.15, stage['y'] - 0.05), 0.7, 0.08,
                boxstyle="round,pad=0.01",
                facecolor=stage['color'],
                edgecolor=stage['border'],
                linewidth=2
            )
            ax.add_patch(rect)
            ax.text(0.5, stage['y'] + 0.015, stage['title'],
                    ha='center', va='center', fontsize=11, fontweight='bold')
            ax.text(0.5, stage['y'] - 0.015, stage['text'],
                    ha='center', va='center', fontsize=9, style='italic')
            if stage['y'] > 0.2:
                ax.annotate('', xy=(0.5, stage['y'] - 0.06), xytext=(0.5, stage['y'] - 0.04),
                            arrowprops=dict(arrowstyle='->', lw=2, color='#424242'))

        ax.text(0.15, 0.05, 'KEY INSIGHT:', fontsize=10, fontweight='bold')
        ax.text(0.15, 0.03,
                'Cascading approach: Rule-Based menangani kasus yang jelas, '
                'AI hanya memproses data ambigu -> hemat biaya & waktu.',
                fontsize=8, style='italic')

        plt.savefig(self.output_dir / '7_research_flow_diagram.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("Created: 7_research_flow_diagram.png")

    # ------------------------------------------------------------------ #
    # MAIN                                                                 #
    # ------------------------------------------------------------------ #

    def create_all_visualizations(self):
        """Buat semua visualisasi."""
        print("\nCreating visualizations...\n")

        self.plot_confidence_distribution()     # 1
        self.plot_latency_distribution()        # 2
        self.plot_agreement_analysis()          # 3
        self.plot_classification_comparison()   # 4
        self.plot_cost_analysis()               # 5 (BARU)
        self.plot_complete_dashboard()          # 6
        self.plot_research_flow()               # 7

        print("\nAll visualizations created!")
        print(f"  Saved to: {self.output_dir}/")


def main():
    """Main function"""
    print("=" * 70)
    print("RESEARCH RESULTS VISUALIZER")
    print("  Membaca dari: results/aggregate_results.json")
    print("=" * 70)

    visualizer = ResearchVisualizer(results_dir="results", output_dir="visualizations")

    if not visualizer.videos:
        print("\n[ERROR] Tidak ada data untuk divisualisasikan.")
        print("        Pastikan detector.py sudah dijalankan dan menghasilkan")
        print("        results/aggregate_results.json terlebih dahulu.")
        return

    visualizer.create_all_visualizations()

    print("\n" + "=" * 70)
    print("VISUALIZATION COMPLETE!")
    print("=" * 70)
    print("\nGenerated files:")
    print("  1. 1_confidence_distribution.png")
    print("  2. 2_latency_distribution.png")
    print("  3. 3_agreement_analysis.png")
    print("  4. 4_classification_comparison.png")
    print("  5. 5_cost_analysis.png          <- BARU")
    print("  6. 6_complete_dashboard.png")
    print("  7. 7_research_flow_diagram.png")
    print("\n  Location: visualizations/")
    print("=" * 70)


if __name__ == "__main__":
    main()
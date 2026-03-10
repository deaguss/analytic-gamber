# -*- coding: utf-8 -*-
import json
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
import numpy as np
import seaborn as sns

sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 12
plt.rcParams['axes.labelsize'] = 13
plt.rcParams['axes.titlesize'] = 15
plt.rcParams['xtick.labelsize'] = 11
plt.rcParams['ytick.labelsize'] = 11
plt.rcParams['legend.fontsize'] = 11

# ------------------------------------------------------------------ #
# Konstanta kategori sampel                                           #
# ------------------------------------------------------------------ #
CAT_COLOR = {
    'JUDI':       '#FF6B6B',
    'TIDAK_JUDI': '#51CF66',
    'AMBIGU':     '#FFD43B',
    'UNKNOWN':    '#ADB5BD',
}
CAT_LABEL = {
    'JUDI':       'Berindikasi Judi',
    'TIDAK_JUDI': 'Tidak Judi',
    'AMBIGU':     'Ambigu',
}
CAT_ORDER  = ['JUDI', 'TIDAK_JUDI', 'AMBIGU']
CAT_LABELS = [CAT_LABEL[c] for c in CAT_ORDER]
CAT_COLORS = [CAT_COLOR[c] for c in CAT_ORDER]

GEMINI_COLOR = '#4285F4'
GPT_COLOR    = '#34A853'


def bar_labels(ax, bars, fmt=None):
    """Tulis nilai di atas setiap bar."""
    for bar in bars:
        h = bar.get_height()
        if h == 0:
            continue
        label = fmt(h) if fmt else (f'{h:.4f}' if h % 1 != 0 else f'{int(h)}')
        ax.text(bar.get_x() + bar.get_width() / 2., h,
                label, ha='center', va='bottom', fontsize=10, fontweight='bold')


def headroom(ax, factor=1.3):
    """Tambahkan ruang di atas bar tertinggi."""
    ymax = ax.get_ylim()[1]
    ax.set_ylim(0, max(ymax * factor, 0.001))

def set_cat_xticks(ax, x_positions, labels, colors, fontsize=11):
    """X-tick label dengan background warna kategori sebagai keterangan visual."""
    ax.set_xticks(x_positions)
    ax.set_xticklabels(labels, fontsize=fontsize)
    for tick, col in zip(ax.get_xticklabels(), colors):
        tick.set_bbox(dict(
            boxstyle='round,pad=0.35',
            facecolor=col,
            alpha=0.45,
            edgecolor=col,
            linewidth=0.8
        ))


def legend_above(ax, ncol=2, fontsize=10, **kwargs):
    """Legend compact di pojok kanan atas LUAR area plot — tidak pernah timpa bar/value."""
    defaults = dict(
        loc='upper left',
        bbox_to_anchor=(0, 1.18),
        ncol=ncol,
        borderaxespad=0,
        framealpha=0.92,
        fontsize=fontsize,
        handlelength=1.6,
        handleheight=0.9,
        handletextpad=0.5,
        columnspacing=1.0,
        borderpad=0.5,
    )
    defaults.update(kwargs)
    return ax.legend(**defaults)




# ------------------------------------------------------------------ #
# DATA LOADER                                                          #
# ------------------------------------------------------------------ #

class ResearchVisualizer:

    def __init__(self, results_dir="results", output_dir="visualizations"):
        self.results_dir = Path(results_dir)
        self.output_dir  = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.load_data()

    def load_data(self):
        print("Loading data from aggregate_results.json ...")
        agg_path = self.results_dir / "aggregate_results.json"
        if not agg_path.exists():
            print(f"[ERROR] File tidak ditemukan: {agg_path}")
            self.aggregate = None
            self.videos    = []
            self.summary   = {}
            self.averages  = {}
            self.by_cat    = {}
            return

        with open(agg_path, 'r', encoding='utf-8') as f:
            self.aggregate = json.load(f)

        self.videos   = self.aggregate.get('per_video_results', [])
        self.summary  = self.aggregate.get('summary', {})
        self.averages = self.aggregate.get('averages', {})

        # Kelompokkan video per kategori sampel
        self.by_cat = {c: [] for c in CAT_ORDER}
        for v in self.videos:
            cat = v.get('sample_category', 'UNKNOWN')
            if cat in self.by_cat:
                self.by_cat[cat].append(v)

        n = {c: len(self.by_cat[c]) for c in CAT_ORDER}
        print(f"  Loaded {len(self.videos)} video(s): "
              f"JUDI={n['JUDI']}, TIDAK_JUDI={n['TIDAK_JUDI']}, AMBIGU={n['AMBIGU']}")
        print("Data loaded successfully!")

    # ---------------------------------------------------------------- #
    # HELPER: rata-rata field dari list video                           #
    # ---------------------------------------------------------------- #
    def _avg(self, videos, field):
        vals = [v.get(field, 0) for v in videos if v.get(field) is not None]
        return sum(vals) / len(vals) if vals else 0.0

    def _sum(self, videos, field):
        return sum(v.get(field, 0) for v in videos)

    # ================================================================ #
    # 1. CONFIDENCE  -  Rata-rata per kategori sampel                     #
    # ================================================================ #
    def plot_confidence_distribution(self):
        if not self.videos:
            print("Skipping confidence - no data"); return

        gemini_vals = [self._avg(self.by_cat[c], 'gemini_avg_confidence') for c in CAT_ORDER]
        gpt_vals    = [self._avg(self.by_cat[c], 'gpt_avg_confidence')    for c in CAT_ORDER]

        x, w = np.arange(3), 0.32
        fig, ax = plt.subplots(figsize=(14, 8))

        b1 = ax.bar(x - w/2, gemini_vals, w, label='Gemini (gemini-2.5-flash)',
                    color=GEMINI_COLOR, alpha=0.88, edgecolor='white')
        b2 = ax.bar(x + w/2, gpt_vals,    w, label='GPT (gpt-4o-mini)',
                    color=GPT_COLOR,    alpha=0.88, edgecolor='white')

        bar_labels(ax, b1, fmt=lambda h: f'{h:.3f}')
        bar_labels(ax, b2, fmt=lambda h: f'{h:.3f}')

        ax.set_title('Rata-rata Confidence Score per Kategori Sampel\n'
                     '(Seberapa yakin model dalam mengklasifikasikan komentar)',
                     fontweight='bold', fontsize=13)
        ax.set_ylabel('Avg Confidence Score (0 = tidak yakin, 1 = sangat yakin)')
        ax.set_xlabel('Kategori Sampel Video')
        ax.set_ylim(0, 1.15)
        legend_above(ax)
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        set_cat_xticks(ax, x, CAT_LABELS, CAT_COLORS, fontsize=11)

        fig.text(0.5, 0.01,
                 'Confidence score tinggi = model lebih yakin. '
                 'Semakin mendekati 1.0 semakin baik.',
                 ha='center', fontsize=11, style='italic', color='gray')
        plt.tight_layout(rect=[0, 0.04, 1, 0.88])
        plt.savefig(self.output_dir / '1_confidence_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("Created: 1_confidence_distribution.png")

    # ================================================================ #
    # 2. LATENCY  -  Rata-rata per kategori sampel                        #
    # ================================================================ #
    def plot_latency_distribution(self):
        if not self.videos:
            print("Skipping latency - no data"); return

        gemini_vals = [self._avg(self.by_cat[c], 'gemini_avg_latency') for c in CAT_ORDER]
        gpt_vals    = [self._avg(self.by_cat[c], 'gpt_avg_latency')    for c in CAT_ORDER]

        x, w = np.arange(3), 0.32
        fig, ax = plt.subplots(figsize=(14, 8))

        b1 = ax.bar(x - w/2, gemini_vals, w, label='Gemini (gemini-2.5-flash)',
                    color=GEMINI_COLOR, alpha=0.88, edgecolor='white')
        b2 = ax.bar(x + w/2, gpt_vals,    w, label='GPT (gpt-4o-mini)',
                    color=GPT_COLOR,    alpha=0.88, edgecolor='white')

        bar_labels(ax, b1, fmt=lambda h: f'{h:.0f} ms')
        bar_labels(ax, b2, fmt=lambda h: f'{h:.0f} ms')

        ax.set_title('Rata-rata Latency Response per Kategori Sampel\n'
                     '(Waktu respons API per komentar  -  semakin rendah semakin baik)',
                     fontweight='bold', fontsize=13)
        ax.set_ylabel('Avg Latency (ms)')
        ax.set_xlabel('Kategori Sampel Video')
        legend_above(ax)
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        headroom(ax, 1.35)
        set_cat_xticks(ax, x, CAT_LABELS, CAT_COLORS, fontsize=11)

        fig.text(0.5, 0.01,
                 'Latency diukur dari pengiriman request sampai respons diterima (dalam milidetik).',
                 ha='center', fontsize=11, style='italic', color='gray')
        plt.tight_layout(rect=[0, 0.04, 1, 0.88])
        plt.savefig(self.output_dir / '2_latency_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("Created: 2_latency_distribution.png")

    # ================================================================ #
    # 3. AGREEMENT  -  Total & per kategori sampel                        #
    # ================================================================ #
    def plot_agreement_analysis(self):
        if not self.videos:
            print("Skipping agreement - no data"); return

        total_agree    = self._sum(self.videos, 'agreement_count')
        total_disagree = self._sum(self.videos, 'disagreement_count')
        total_llm      = total_agree + total_disagree
        rate_total     = (total_agree / total_llm * 100) if total_llm else 0

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

        # --- Pie: agreement rate keseluruhan ---
        ax1.pie(
            [total_agree, total_disagree],
            labels=[f'Sepakat\n{total_agree:,} komentar\n({rate_total:.1f}%)',
                    f'Tidak Sepakat\n{total_disagree:,} komentar\n({100-rate_total:.1f}%)'],
            colors=['#34A853', '#EA4335'],
            explode=(0.05, 0.05), startangle=90,
            textprops={'fontsize': 11, 'fontweight': 'bold'}
        )
        ax1.set_title(
            f'Agreement Rate Keseluruhan\n'
            f'Total {total_llm:,} komentar diproses LLM',
            fontweight='bold', fontsize=13
        )

        # --- Bar: agreement rate per kategori ---
        rates = []
        agrees = []
        disagrees = []
        for c in CAT_ORDER:
            vids = self.by_cat[c]
            ag   = self._sum(vids, 'agreement_count')
            dag  = self._sum(vids, 'disagreement_count')
            tot  = ag + dag
            rates.append((ag / tot * 100) if tot else 0)
            agrees.append(ag)
            disagrees.append(dag)

        x, w = np.arange(3), 0.32
        b1 = ax2.bar(x - w/2, agrees,    w, label='Sepakat',       color='#34A853', alpha=0.88)
        b2 = ax2.bar(x + w/2, disagrees, w, label='Tidak Sepakat', color='#EA4335', alpha=0.88)

        bar_labels(ax2, b1)
        bar_labels(ax2, b2)

        # Tampilkan agreement rate di atas tiap pasang bar
        for i, (xi, rate) in enumerate(zip(x, rates)):
            ax2.text(xi, max(agrees[i], disagrees[i]) * 1.18,
                     f'Rate: {rate:.1f}%',
                     ha='center', fontsize=11, fontweight='bold', color='#1a1a1a')

        ax2.set_title('Jumlah Komentar Sepakat vs Tidak Sepakat\nper Kategori Sampel',
                      fontweight='bold', fontsize=13)
        ax2.set_ylabel('Jumlah Komentar')
        ax2.set_xlabel('Kategori Sampel Video')
        legend_above(ax2)
        ax2.grid(axis='y', alpha=0.3, linestyle='--')
        headroom(ax2, 1.45)
        set_cat_xticks(ax2, x, CAT_LABELS, CAT_COLORS, fontsize=11)

        fig.text(0.5, 0.01,
                 'Agreement = Gemini dan GPT menghasilkan klasifikasi yang sama pada komentar ambigu.',
                 ha='center', fontsize=11, style='italic', color='gray')
        plt.tight_layout(rect=[0, 0.04, 1, 0.88])
        plt.savefig(self.output_dir / '3_agreement_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("Created: 3_agreement_analysis.png")

    # ================================================================ #
    # 4. CLASSIFICATION  -  Total deteksi per kategori sampel             #
    # ================================================================ #
    def plot_classification_comparison(self):
        if not self.videos:
            print("Skipping classification - no data"); return

        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Perbandingan Hasil Klasifikasi LLM per Kategori Sampel\n'
                     '(Hanya komentar ambigu yang dikirim ke LLM)',
                     fontweight='bold', fontsize=14, y=1.01)

        # Panel 1 & 2: Total deteksi judi & bukan judi per kategori
        for idx, (model, color, field_j, field_b, title) in enumerate([
            ('Gemini (gemini-2.5-flash)', GEMINI_COLOR, 'gemini_judi', 'gemini_bukan_judi',
             'Gemini: Total Deteksi per Kategori Sampel'),
            ('GPT (gpt-4o-mini)',          GPT_COLOR,   'gpt_judi',    'gpt_bukan_judi',
             'GPT: Total Deteksi per Kategori Sampel'),
        ]):
            ax  = axes[0][idx]
            judi  = [self._sum(self.by_cat[c], field_j) for c in CAT_ORDER]
            bukan = [self._sum(self.by_cat[c], field_b) for c in CAT_ORDER]
            x, w  = np.arange(3), 0.32

            b1 = ax.bar(x - w/2, judi,  w, label='Terdeteksi Judi',  color='#EA4335', alpha=0.88)
            b2 = ax.bar(x + w/2, bukan, w, label='Bukan Judi',        color='#34A853', alpha=0.88)
            bar_labels(ax, b1)
            bar_labels(ax, b2)

            ax.set_title(title, fontweight='bold', fontsize=11)
            ax.set_ylabel('Jumlah Komentar')
            legend_above(ax, fontsize=10)
            ax.grid(axis='y', alpha=0.3, linestyle='--')
            headroom(ax, 1.35)
            set_cat_xticks(ax, x, CAT_LABELS, CAT_COLORS, fontsize=10)

        # Panel 3: Detection rate (%) per kategori per model
        ax3 = axes[1][0]
        x, w = np.arange(3), 0.32
        for shift, (model, color, fj, fb) in enumerate([
            ('Gemini', GEMINI_COLOR, 'gemini_judi', 'gemini_bukan_judi'),
            ('GPT',    GPT_COLOR,    'gpt_judi',    'gpt_bukan_judi'),
        ]):
            rates = []
            for c in CAT_ORDER:
                j = self._sum(self.by_cat[c], fj)
                b = self._sum(self.by_cat[c], fb)
                rates.append((j / (j + b) * 100) if (j + b) > 0 else 0)
            off = -w/2 if shift == 0 else w/2
            bars = ax3.bar(x + off, rates, w, label=model, color=color, alpha=0.88)
            bar_labels(ax3, bars, fmt=lambda h: f'{h:.1f}%')

        ax3.set_title('Detection Rate per Kategori Sampel\n'
                      '(% komentar ambigu yang terdeteksi sebagai judi)',
                      fontweight='bold', fontsize=11)
        ax3.set_ylabel('Detection Rate (%)')
        ax3.set_xlabel('Kategori Sampel Video')
        ax3.set_ylim(0, 120)
        legend_above(ax3, fontsize=10)
        ax3.grid(axis='y', alpha=0.3, linestyle='--')
        set_cat_xticks(ax3, x, CAT_LABELS, CAT_COLORS, fontsize=10)

        # Panel 4: Total keseluruhan Gemini vs GPT (ringkasan)
        ax4 = axes[1][1]
        labels_x = ['Judi\n(Gemini)', 'Bukan Judi\n(Gemini)', 'Judi\n(GPT)', 'Bukan Judi\n(GPT)']
        vals = [
            self._sum(self.videos, 'gemini_judi'),
            self._sum(self.videos, 'gemini_bukan_judi'),
            self._sum(self.videos, 'gpt_judi'),
            self._sum(self.videos, 'gpt_bukan_judi'),
        ]
        colors_bar = ['#EA4335', '#34A853', '#EA4335', '#34A853']
        alphas     = [0.9, 0.9, 0.6, 0.6]
        x4 = np.arange(4)
        for xi, val, col, alp in zip(x4, vals, colors_bar, alphas):
            b = ax4.bar(xi, val, 0.55, color=col, alpha=alp, edgecolor='white')
            ax4.text(xi, val, f'{val:,}', ha='center', va='bottom',
                     fontsize=11, fontweight='bold')

        # Garis pemisah Gemini vs GPT
        ax4.axvline(1.5, color='gray', linestyle='--', linewidth=1.2, alpha=0.6)
        ax4.text(0.75,  max(vals)*1.08, 'Gemini', ha='center', fontsize=10,
                 fontweight='bold', color=GEMINI_COLOR)
        ax4.text(2.5, max(vals)*1.08, 'GPT', ha='center', fontsize=10,
                 fontweight='bold', color=GPT_COLOR)

        ax4.set_title('Total Klasifikasi LLM  -  Seluruh Data\n(30 video, semua kategori)',
                      fontweight='bold', fontsize=11)
        ax4.set_ylabel('Jumlah Komentar')
        ax4.set_xticks(x4); ax4.set_xticklabels(labels_x, fontsize=11)
        ax4.grid(axis='y', alpha=0.3, linestyle='--')
        headroom(ax4, 1.25)

        # Legend warna arsir
        leg = [mpatches.Patch(color='#EA4335', label='Terdeteksi Judi'),
               mpatches.Patch(color='#34A853', label='Bukan Judi')]
        legend_above(ax4, handles=leg, fontsize=11)

        plt.tight_layout(rect=[0, 0, 1, 0.92])
        plt.savefig(self.output_dir / '4_classification_comparison.png',
                    dpi=300, bbox_inches='tight')
        plt.close()
        print("Created: 4_classification_comparison.png")

    # ================================================================ #
    # 5. COST ANALYSIS  -  Total & breakdown agregat                      #
    # ================================================================ #
    def plot_cost_analysis(self):
        if not self.videos:
            print("Skipping cost - no data"); return

        # Agregat keseluruhan
        total_gemini_cost = self._sum(self.videos, 'gemini_total_cost_usd')
        total_gpt_cost    = self._sum(self.videos, 'gpt_total_cost_usd')
        grand_total       = total_gemini_cost + total_gpt_cost

        total_g_in  = self._sum(self.videos, 'gemini_total_input_tokens')
        total_g_out = self._sum(self.videos, 'gemini_total_output_tokens')
        total_p_in  = self._sum(self.videos, 'gpt_total_input_tokens')
        total_p_out = self._sum(self.videos, 'gpt_total_output_tokens')

        # Cost per kategori
        cost_by_cat_g = [self._sum(self.by_cat[c], 'gemini_total_cost_usd') for c in CAT_ORDER]
        cost_by_cat_p = [self._sum(self.by_cat[c], 'gpt_total_cost_usd')    for c in CAT_ORDER]
        cost_by_cat_t = [g + p for g, p in zip(cost_by_cat_g, cost_by_cat_p)]

        # Input/output cost breakdown
        g_in_cost  = (total_g_in  / 1_000_000) * 0.30   # Gemini $0.30/1M
        g_out_cost = (total_g_out / 1_000_000) * 2.50   # Gemini $2.50/1M
        p_in_cost  = (total_p_in  / 1_000_000) * 0.15   # GPT    $0.15/1M
        p_out_cost = (total_p_out / 1_000_000) * 0.60   # GPT    $0.60/1M

        fig = plt.figure(figsize=(22, 18))
        gs  = fig.add_gridspec(2, 2, hspace=0.5, wspace=0.38)
        fig.suptitle(
            'ANALISIS BIAYA API  -  Gemini (gemini-2.5-flash) vs GPT (gpt-4o-mini)\n'
            'Gemini: $0.30/1M input token, $2.50/1M output token  |  '
            'GPT: $0.15/1M input token, $0.60/1M output token',
            fontsize=14, fontweight='bold', y=1.01
        )

        # --- Panel 1: Total cost per model (bar besar) ---
        ax1 = fig.add_subplot(gs[0, 0])
        models = ['Gemini\n(gemini-2.5-flash)', 'GPT\n(gpt-4o-mini)', 'Total\nKeduanya']
        costs  = [total_gemini_cost, total_gpt_cost, grand_total]
        colors = [GEMINI_COLOR, GPT_COLOR, '#9E9E9E']
        bars   = ax1.bar(models, costs, color=colors, alpha=0.88, width=0.5, edgecolor='white')
        for bar, val in zip(bars, costs):
            ax1.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
                     f'${val:.5f}\n(Rp {val*16300:,.0f})',
                     ha='center', va='bottom', fontsize=11, fontweight='bold')
        ax1.set_title('Total Biaya API Keseluruhan\n(30 video, semua komentar ambigu)',
                      fontweight='bold')
        ax1.set_ylabel('Biaya (USD)')
        ax1.grid(axis='y', alpha=0.3, linestyle='--')
        headroom(ax1, 1.55)

        # --- Panel 2: Pie proporsi cost ---
        ax2 = fig.add_subplot(gs[0, 1])
        if grand_total > 0:
            ax2.pie(
                [total_gemini_cost, total_gpt_cost],
                labels=[f'Gemini\n${total_gemini_cost:.5f}\n({total_gemini_cost/grand_total*100:.1f}%)',
                        f'GPT\n${total_gpt_cost:.5f}\n({total_gpt_cost/grand_total*100:.1f}%)'],
                colors=[GEMINI_COLOR, GPT_COLOR],
                explode=(0.05, 0.05), startangle=90,
                textprops={'fontsize': 11, 'fontweight': 'bold'}
            )
            ax2.set_title(f'Proporsi Biaya API\nGrand Total: ${grand_total:.5f} USD '
                          f'(~ Rp {grand_total*16300:,.0f})',
                          fontweight='bold')
        else:
            ax2.text(0.5, 0.5, 'Data cost belum tersedia', ha='center', va='center')
            ax2.axis('off')

        # --- Panel 3: Cost per kategori sampel ---
        ax3 = fig.add_subplot(gs[1, 0])
        x, w = np.arange(3), 0.32
        b1 = ax3.bar(x - w/2, cost_by_cat_g, w, label='Gemini', color=GEMINI_COLOR, alpha=0.88)
        b2 = ax3.bar(x + w/2, cost_by_cat_p, w, label='GPT',    color=GPT_COLOR,    alpha=0.88)
        bar_labels(ax3, b1, fmt=lambda h: f'${h:.4f}')
        bar_labels(ax3, b2, fmt=lambda h: f'${h:.4f}')
        ax3.set_title('Biaya API per Kategori Sampel\n(Total 10 video per kategori)',
                      fontweight='bold')
        ax3.set_ylabel('Biaya (USD)')
        ax3.set_xlabel('Kategori Sampel Video')
        legend_above(ax3)
        ax3.grid(axis='y', alpha=0.3, linestyle='--')
        headroom(ax3, 1.45)
        set_cat_xticks(ax3, x, CAT_LABELS, CAT_COLORS, fontsize=10)

        # --- Panel 4: Breakdown input vs output cost per model ---
        ax4 = fig.add_subplot(gs[1, 1])
        x4      = np.arange(2)
        in_c    = [g_in_cost,  p_in_cost]
        out_c   = [g_out_cost, p_out_cost]
        b_in  = ax4.bar(x4, in_c,  0.45, label='Input token cost',  color='#74C0FC', alpha=0.9)
        b_out = ax4.bar(x4, out_c, 0.45, label='Output token cost', color='#FF8787', alpha=0.9,
                        bottom=in_c)
        # Label total
        for xi, ic, oc in zip(x4, in_c, out_c):
            total = ic + oc
            ax4.text(xi, total, f'${total:.5f}', ha='center', va='bottom',
                     fontsize=11, fontweight='bold')
            ax4.text(xi, ic/2, f'Input\n${ic:.5f}', ha='center', va='center',
                     fontsize=11, color='#1a5276')
            if oc > 0:
                ax4.text(xi, ic + oc/2, f'Output\n${oc:.5f}', ha='center', va='center',
                         fontsize=11, color='#7b241c')

        ax4.set_title('Breakdown: Input Token vs Output Token Cost\n'
                      '(Output lebih mahal karena harga per-token lebih tinggi)',
                      fontweight='bold')
        ax4.set_ylabel('Biaya (USD)')
        ax4.set_xticks(x4)
        ax4.set_xticklabels(['Gemini\n(gemini-2.5-flash)', 'GPT\n(gpt-4o-mini)'], fontsize=10)
        legend_above(ax4)
        ax4.grid(axis='y', alpha=0.3, linestyle='--')
        headroom(ax4, 1.35)

        # Token usage footnote
        fig.text(0.5, 0.01,
                 f'Token Usage  -  Gemini: {total_g_in:,} input + {total_g_out:,} output  |  '
                 f'GPT: {total_p_in:,} input + {total_p_out:,} output',
                 ha='center', fontsize=11, fontweight='bold',
                 bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

        plt.savefig(self.output_dir / '5_cost_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("Created: 5_cost_analysis.png")

    # ================================================================ #
    # 6. COMPLETE DASHBOARD                                             #
    # ================================================================ #
    def plot_complete_dashboard(self):
        if not self.videos:
            print("Skipping dashboard - no data"); return

        # Hitung semua metrik agregat
        total_g_judi  = self._sum(self.videos, 'gemini_judi')
        total_g_bukan = self._sum(self.videos, 'gemini_bukan_judi')
        total_p_judi  = self._sum(self.videos, 'gpt_judi')
        total_p_bukan = self._sum(self.videos, 'gpt_bukan_judi')

        total_rb_judi  = self.summary.get('total_judi_online', 0)
        total_rb_bukan = self.summary.get('total_bukan_judi', 0)
        total_rb_ambi  = self.summary.get('total_ambigu', 0)
        total_comments = self.summary.get('total_comments', 0)

        total_agree    = self._sum(self.videos, 'agreement_count')
        total_disagree = self._sum(self.videos, 'disagreement_count')
        total_llm      = total_agree + total_disagree
        agr_rate       = (total_agree / total_llm * 100) if total_llm else 0

        g_conf = self.averages.get('gemini_confidence', 0)
        p_conf = self.averages.get('gpt_confidence', 0)
        g_lat  = self.averages.get('gemini_latency_ms', 0)
        p_lat  = self.averages.get('gpt_latency_ms', 0)

        total_g_cost = self.summary.get('gemini_total_cost_usd', 0.0)
        total_p_cost = self.summary.get('gpt_total_cost_usd', 0.0)
        grand_cost   = self.summary.get('total_cost_usd', 0.0)

        def lbl(ax, bars, fmt=None):
            for bar in bars:
                h = bar.get_height()
                if h == 0: continue
                label = fmt(h) if fmt else (f'{h:.3f}' if h % 1 != 0 else f'{int(h)}')
                ax.text(bar.get_x() + bar.get_width()/2., h,
                        label, ha='center', va='bottom', fontsize=10, fontweight='bold')
            ax.set_ylim(0, max(max(b.get_height() for b in bars) * 1.35, 0.001))

        fig = plt.figure(figsize=(22, 18))
        gs  = fig.add_gridspec(3, 3, hspace=0.75, wspace=0.40)
        fig.suptitle(
            'DASHBOARD RINGKASAN PENELITIAN\n'
            'Komparasi Gemini (gemini-2.5-flash) vs GPT (gpt-4o-mini)\n'
            'Deteksi Spam Komentar Judi Online pada YouTube',
            fontsize=16, fontweight='bold', y=1.01
        )

        # --- P1: Rule-based hasil per kategori ---
        ax1 = fig.add_subplot(gs[0, 0])
        cat_rb_judi  = [self._sum(self.by_cat[c], 'rule_based_judi')      for c in CAT_ORDER]
        cat_rb_bukan = [self._sum(self.by_cat[c], 'rule_based_bukan_judi') for c in CAT_ORDER]
        cat_rb_ambi  = [self._sum(self.by_cat[c], 'rule_based_ambigu')    for c in CAT_ORDER]
        x, w = np.arange(3), 0.25
        b1 = ax1.bar(x - w,   cat_rb_judi,  w, label='Judi',        color='#EA4335', alpha=0.85)
        b2 = ax1.bar(x,       cat_rb_bukan, w, label='Bukan Judi',  color='#34A853', alpha=0.85)
        b3 = ax1.bar(x + w,   cat_rb_ambi,  w, label='Ambigu->LLM', color='#FBBC04', alpha=0.85)
        for b in [b1, b2, b3]: lbl(ax1, b)
        ax1.set_title('Hasil Rule-Based\nper Kategori Sampel', fontweight='bold')
        ax1.set_ylabel('Jumlah Komentar')
        ax1.legend(fontsize=8, ncol=3, loc='upper center',
                   bbox_to_anchor=(0.5, -0.22), framealpha=0.9,
                   handlelength=1.2, borderpad=0.4, columnspacing=0.8)
        ax1.grid(axis='y', alpha=0.3)
        set_cat_xticks(ax1, x, CAT_LABELS, CAT_COLORS, fontsize=9)

        # --- P2: Agreement pie ---
        ax2 = fig.add_subplot(gs[0, 1])
        ax2.pie(
            [total_agree, total_disagree],
            labels=[f'Sepakat\n{total_agree:,}\n({agr_rate:.1f}%)',
                    f'Tidak Sepakat\n{total_disagree:,}\n({100-agr_rate:.1f}%)'],
            colors=['#34A853', '#EA4335'],
            explode=(0.05, 0.05), startangle=90,
            textprops={'fontsize': 10, 'fontweight': 'bold'}
        )
        ax2.set_title(f'Agreement Rate\n{agr_rate:.1f}% dari {total_llm:,} komentar',
                      fontweight='bold')

        # --- P3: Summary text ---
        ax3 = fig.add_subplot(gs[0, 2])
        ax3.axis('off')
        faster    = 'GPT' if p_lat  < g_lat  else 'Gemini'
        confident = 'GPT' if p_conf > g_conf else 'Gemini'
        cheaper   = 'GPT' if total_p_cost <= total_g_cost else 'Gemini'
        summary_text = (
            "RINGKASAN PERFORMA MODEL\n"
            "-----------------------------\n"
            f"Dataset : 30 video ({len(self.by_cat['JUDI'])} Judi | "
            f"{len(self.by_cat['TIDAK_JUDI'])} Tidak Judi | "
            f"{len(self.by_cat['AMBIGU'])} Ambigu)\n"
            f"Total komentar diproses : {total_comments:,}\n"
            f"Dikirim ke LLM (ambigu) : {total_llm:,}\n\n"
            f"Gemini (gemini-2.5-flash):\n"
            f"  Deteksi Judi   : {total_g_judi:,}\n"
            f"  Bukan Judi     : {total_g_bukan:,}\n"
            f"  Avg Confidence : {g_conf:.4f}\n"
            f"  Avg Latency    : {g_lat:.1f} ms\n"
            f"  Total Cost     : ${total_g_cost:.5f}\n\n"
            f"GPT (gpt-4o-mini):\n"
            f"  Deteksi Judi   : {total_p_judi:,}\n"
            f"  Bukan Judi     : {total_p_bukan:,}\n"
            f"  Avg Confidence : {p_conf:.4f}\n"
            f"  Avg Latency    : {p_lat:.1f} ms\n"
            f"  Total Cost     : ${total_p_cost:.5f}\n\n"
            f"-----------------------------\n"
            f"Tercepat (latency)    : {faster}\n"
            f"Paling yakin (conf.)  : {confident}\n"
            f"Paling murah          : {cheaper}\n"
            f"Agreement Rate        : {agr_rate:.1f}%\n"
            f"Grand Total Cost      : ${grand_cost:.5f}\n"
            f"                       (~ Rp {grand_cost*16300:,.0f})"
        )
        ax3.text(0.03, 0.98, summary_text, transform=ax3.transAxes,
                 fontsize=11, verticalalignment='top', fontfamily='monospace',
                 bbox=dict(boxstyle='round', facecolor='#f0f4ff', alpha=0.8))

        # --- P4: LLM classification per kategori (grouped) ---
        ax4 = fig.add_subplot(gs[1, :])
        x4, w4 = np.arange(3), 0.2
        g_judi_cat  = [self._sum(self.by_cat[c], 'gemini_judi')      for c in CAT_ORDER]
        g_bukan_cat = [self._sum(self.by_cat[c], 'gemini_bukan_judi') for c in CAT_ORDER]
        p_judi_cat  = [self._sum(self.by_cat[c], 'gpt_judi')          for c in CAT_ORDER]
        p_bukan_cat = [self._sum(self.by_cat[c], 'gpt_bukan_judi')    for c in CAT_ORDER]

        bg1 = ax4.bar(x4 - 1.5*w4, g_judi_cat,  w4, label='Gemini  -  Judi',       color=GEMINI_COLOR, alpha=0.9)
        bg2 = ax4.bar(x4 - 0.5*w4, g_bukan_cat, w4, label='Gemini  -  Bukan Judi', color=GEMINI_COLOR, alpha=0.45)
        bp1 = ax4.bar(x4 + 0.5*w4, p_judi_cat,  w4, label='GPT  -  Judi',          color=GPT_COLOR,    alpha=0.9)
        bp2 = ax4.bar(x4 + 1.5*w4, p_bukan_cat, w4, label='GPT  -  Bukan Judi',    color=GPT_COLOR,    alpha=0.45)
        for b in [bg1, bg2, bp1, bp2]: lbl(ax4, b)

        ax4.set_title('Hasil Klasifikasi LLM per Kategori Sampel\n'
                      '(Gemini solid = judi | Gemini transparan = bukan judi | '
                      'GPT solid = judi | GPT transparan = bukan judi)',
                      fontweight='bold', fontsize=11)
        ax4.set_ylabel('Jumlah Komentar')
        ax4.set_xlabel('Kategori Sampel Video')
        ax4.legend(fontsize=9, ncol=4, loc='upper center',
                   bbox_to_anchor=(0.5, -0.18), framealpha=0.9,
                   handlelength=1.2, borderpad=0.4, columnspacing=0.8)
        ax4.grid(axis='y', alpha=0.3, linestyle='--')
        set_cat_xticks(ax4, x4, CAT_LABELS, CAT_COLORS, fontsize=10)

        # --- P5: Avg Confidence ---
        ax5 = fig.add_subplot(gs[2, 0])
        g_conf_cat = [self._avg(self.by_cat[c], 'gemini_avg_confidence') for c in CAT_ORDER]
        p_conf_cat = [self._avg(self.by_cat[c], 'gpt_avg_confidence')    for c in CAT_ORDER]
        x5, w5 = np.arange(3), 0.32
        b1 = ax5.bar(x5 - w5/2, g_conf_cat, w5, label='Gemini', color=GEMINI_COLOR, alpha=0.88)
        b2 = ax5.bar(x5 + w5/2, p_conf_cat, w5, label='GPT',    color=GPT_COLOR,    alpha=0.88)
        bar_labels(ax5, b1, fmt=lambda h: f'{h:.3f}')
        bar_labels(ax5, b2, fmt=lambda h: f'{h:.3f}')
        ax5.set_title('Avg Confidence Score\nper Kategori Sampel', fontweight='bold')
        ax5.set_ylabel('Confidence (0-1)')
        set_cat_xticks(ax5, x5, CAT_LABELS, CAT_COLORS, fontsize=9)
        ax5.set_ylim(0, 1.25)
        ax5.legend(fontsize=8, ncol=2, loc='upper center',
                   bbox_to_anchor=(0.5, -0.22), framealpha=0.9,
                   handlelength=1.2, borderpad=0.4)
        ax5.grid(axis='y', alpha=0.3, linestyle='--')

        # --- P6: Avg Latency ---
        ax6 = fig.add_subplot(gs[2, 1])
        g_lat_cat = [self._avg(self.by_cat[c], 'gemini_avg_latency') for c in CAT_ORDER]
        p_lat_cat = [self._avg(self.by_cat[c], 'gpt_avg_latency')    for c in CAT_ORDER]
        b1 = ax6.bar(x5 - w5/2, g_lat_cat, w5, label='Gemini', color=GEMINI_COLOR, alpha=0.88)
        b2 = ax6.bar(x5 + w5/2, p_lat_cat, w5, label='GPT',    color=GPT_COLOR,    alpha=0.88)
        bar_labels(ax6, b1, fmt=lambda h: f'{h:.0f}ms')
        bar_labels(ax6, b2, fmt=lambda h: f'{h:.0f}ms')
        ax6.set_title('Avg Latency (ms)\nper Kategori Sampel', fontweight='bold')
        ax6.set_ylabel('Latency (ms)')
        set_cat_xticks(ax6, x5, CAT_LABELS, CAT_COLORS, fontsize=9)
        ax6.legend(fontsize=8, ncol=2, loc='upper center',
                   bbox_to_anchor=(0.5, -0.22), framealpha=0.9,
                   handlelength=1.2, borderpad=0.4)
        ax6.grid(axis='y', alpha=0.3, linestyle='--')
        headroom(ax6, 1.35)

        # --- P7: Cost comparison ---
        ax7 = fig.add_subplot(gs[2, 2])
        bars_c = ax7.bar(
            ['Gemini\n(gemini-2.5-flash)', 'GPT\n(gpt-4o-mini)', 'Total'],
            [total_g_cost, total_p_cost, grand_cost],
            color=[GEMINI_COLOR, GPT_COLOR, '#9E9E9E'], alpha=0.88, width=0.5
        )
        for bar, val in zip(bars_c, [total_g_cost, total_p_cost, grand_cost]):
            ax7.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
                     f'${val:.5f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
        ax7.set_title('Total Biaya API (USD)\n30 video keseluruhan', fontweight='bold')
        ax7.set_ylabel('USD')
        ax7.grid(axis='y', alpha=0.3, linestyle='--')
        headroom(ax7, 1.45)

        plt.savefig(self.output_dir / '6_complete_dashboard.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("Created: 6_complete_dashboard.png")

    # ================================================================ #
    # 7. RESEARCH FLOW                                                  #
    # ================================================================ #
    def plot_research_flow(self):
        fig, ax = plt.subplots(figsize=(16, 11))
        ax.axis('off')
        ax.text(0.5, 0.95, 'ALUR METODOLOGI PENELITIAN', ha='center', va='top',
                fontsize=16, fontweight='bold')
        ax.text(0.5, 0.92, 'Deteksi Spam Komentar Judi Online: Cascading Filtering Approach',
                ha='center', va='top', fontsize=11, style='italic')

        stages = [
            {'y': 0.85, 'title': '1. INPUT DATA',
             'text': '30 Video YouTube (10 Judi | 10 Tidak Judi | 10 Ambigu)  x  200 komentar/video  =  6.000 komentar',
             'color': '#E3F2FD', 'border': '#2196F3'},
            {'y': 0.75, 'title': '2. PRA-PEMROSESAN',
             'text': 'Case Folding -> Normalisasi Leet Speak -> Tokenisasi -> Penghapusan Stopword',
             'color': '#E8F5E9', 'border': '#4CAF50'},
            {'y': 0.65, 'title': '3. RULE-BASED CLASSIFICATION  (Regex + Fuzzy Matching)',
             'text': 'Threshold: similarity >= 0.75 -> Judi  |  < 0.50 -> Bukan Judi  |  0.50-0.75 -> Ambigu',
             'color': '#FFF3E0', 'border': '#FF9800'},
            {'y': 0.52, 'title': '   HASIL RULE-BASED',
             'text': '   Judi Online (langsung)  |  Bukan Judi (langsung)  |  Ambigu -> dikirim ke LLM (~52% komentar)',
             'color': '#FCE4EC', 'border': '#E91E63'},
            {'y': 0.38, 'title': '4. LLM CLASSIFICATION  (Hanya untuk komentar Ambigu)',
             'text': 'Gemini (gemini-2.5-flash) vs GPT (gpt-4o-mini)  -  klasifikasi paralel',
             'color': '#F3E5F5', 'border': '#9C27B0'},
            {'y': 0.25, 'title': '5. EVALUASI KOMPARASI',
             'text': 'Agreement Rate  |  Confidence Score  |  Latency  |  Biaya API (USD)',
             'color': '#E0F2F1', 'border': '#009688'},
            {'y': 0.13, 'title': '6. OUTPUT & KESIMPULAN',
             'text': '5 Tabel Excel (per-video)  +  7 Grafik Visualisasi (agregat)  +  Laporan Penelitian',
             'color': '#FFF9C4', 'border': '#FBC02D'},
        ]

        for stage in stages:
            rect = mpatches.FancyBboxPatch(
                (0.1, stage['y'] - 0.055), 0.8, 0.09,
                boxstyle="round,pad=0.01",
                facecolor=stage['color'], edgecolor=stage['border'], linewidth=2
            )
            ax.add_patch(rect)
            ax.text(0.5, stage['y'] + 0.018, stage['title'],
                    ha='center', va='center', fontsize=11, fontweight='bold')
            ax.text(0.5, stage['y'] - 0.018, stage['text'],
                    ha='center', va='center', fontsize=11, style='italic')
            if stage['y'] > 0.18:
                ax.annotate('', xy=(0.5, stage['y'] - 0.065), xytext=(0.5, stage['y'] - 0.045),
                            arrowprops=dict(arrowstyle='->', lw=2, color='#424242'))

        ax.text(0.1, 0.04, 'KEY INSIGHT:', fontsize=10, fontweight='bold')
        ax.text(0.1, 0.02,
                'Cascading approach: Rule-Based menyelesaikan kasus yang jelas -> '
                'LLM hanya memproses komentar ambigu -> efisien biaya & waktu.',
                fontsize=8.5, style='italic')

        plt.savefig(self.output_dir / '7_research_flow_diagram.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("Created: 7_research_flow_diagram.png")

    # ================================================================ #
    # MAIN                                                              #
    # ================================================================ #
    def create_all_visualizations(self):
        print("\nCreating visualizations...\n")
        self.plot_confidence_distribution()
        self.plot_latency_distribution()
        self.plot_agreement_analysis()
        self.plot_classification_comparison()
        self.plot_cost_analysis()
        self.plot_complete_dashboard()
        self.plot_research_flow()
        print(f"\nAll visualizations created! -> {self.output_dir}/")


def main():
    print("=" * 70)
    print("RESEARCH RESULTS VISUALIZER")
    print("  Membaca dari: results/aggregate_results.json")
    print("=" * 70)
    viz = ResearchVisualizer(results_dir="results", output_dir="visualizations")
    if not viz.videos:
        print("\n[ERROR] Tidak ada data. Jalankan detector.py terlebih dahulu.")
        return
    viz.create_all_visualizations()
    print("\n" + "=" * 70)
    print("SELESAI! File tersimpan di: visualizations/")
    print("  1. 1_confidence_distribution.png   -  Avg confidence per kategori")
    print("  2. 2_latency_distribution.png      -  Avg latency per kategori")
    print("  3. 3_agreement_analysis.png        -  Agreement rate total & per kategori")
    print("  4. 4_classification_comparison.png  -  Hasil klasifikasi LLM per kategori")
    print("  5. 5_cost_analysis.png             -  Biaya API total & breakdown")
    print("  6. 6_complete_dashboard.png        -  Dashboard ringkasan lengkap")
    print("  7. 7_research_flow_diagram.png     -  Alur metodologi penelitian")
    print("=" * 70)


if __name__ == "__main__":
    main()
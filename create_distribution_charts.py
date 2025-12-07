"""
Script untuk membuat diagram distribusi dataset
untuk keperluan dokumentasi skripsi
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.size'] = 11

# Load data
df_3class = pd.read_csv('data/gojek_reviews_3class_clean.csv')
df_5class = pd.read_csv('data/gojek_reviews_5class_clean.csv')

# ============================================
# GAMBAR 1: Distribusi 3-Kelas (Pie Chart)
# ============================================
fig1, ax1 = plt.subplots(figsize=(8, 6))

# Data 3-kelas
labels_3class = ['Negatif', 'Netral', 'Positif']
sizes_3class = df_3class['sentiment'].value_counts()[['negative', 'neutral', 'positive']].values
colors_3class = ['#FF6B6B', '#FFE66D', '#4ECDC4']
explode_3class = (0.02, 0.02, 0.02)

wedges, texts, autotexts = ax1.pie(
    sizes_3class, 
    labels=None,
    colors=colors_3class,
    autopct='%1.1f%%',
    startangle=90,
    explode=explode_3class,
    pctdistance=0.75
)

# Legend dengan jumlah
legend_labels = [f'{label} ({size:,})' for label, size in zip(labels_3class, sizes_3class)]
ax1.legend(wedges, legend_labels, title="Kelas Sentimen", loc="center left", bbox_to_anchor=(1, 0, 0.5, 1))

ax1.set_title('Distribusi Label Sentimen Skema 3-Kelas\n(Total: {:,} ulasan)'.format(len(df_3class)), 
              fontsize=14, fontweight='bold', pad=20)

plt.tight_layout()
plt.savefig('distribusi_3kelas.png', dpi=300, bbox_inches='tight', facecolor='white')
print("✅ Saved: distribusi_3kelas.png")

# ============================================
# GAMBAR 2: Distribusi 5-Kelas (Pie Chart)
# ============================================
fig2, ax2 = plt.subplots(figsize=(8, 6))

# Data 5-kelas
labels_5class = ['Sangat Negatif\n(Rating 1)', 'Negatif\n(Rating 2)', 'Netral\n(Rating 3)', 
                 'Positif\n(Rating 4)', 'Sangat Positif\n(Rating 5)']
sizes_5class = df_5class['rating'].value_counts().sort_index().values
colors_5class = ['#D62828', '#F77F00', '#FFE66D', '#4ECDC4', '#2A9D8F']
explode_5class = (0.02, 0.02, 0.02, 0.02, 0.02)

wedges2, texts2, autotexts2 = ax2.pie(
    sizes_5class, 
    labels=None,
    colors=colors_5class,
    autopct='%1.1f%%',
    startangle=90,
    explode=explode_5class,
    pctdistance=0.75
)

# Legend dengan jumlah
labels_short = ['Sangat Negatif', 'Negatif', 'Netral', 'Positif', 'Sangat Positif']
legend_labels_5 = [f'{label} ({size:,})' for label, size in zip(labels_short, sizes_5class)]
ax2.legend(wedges2, legend_labels_5, title="Kelas Sentimen", loc="center left", bbox_to_anchor=(1, 0, 0.5, 1))

ax2.set_title('Distribusi Label Sentimen Skema 5-Kelas\n(Total: {:,} ulasan)'.format(len(df_5class)), 
              fontsize=14, fontweight='bold', pad=20)

plt.tight_layout()
plt.savefig('distribusi_5kelas.png', dpi=300, bbox_inches='tight', facecolor='white')
print("✅ Saved: distribusi_5kelas.png")

# ============================================
# GAMBAR 3: Perbandingan Bar Chart (Combined)
# ============================================
fig3, axes = plt.subplots(1, 2, figsize=(14, 5))

# 3-Kelas Bar Chart
ax3a = axes[0]
x_3class = np.arange(len(labels_3class))
bars_3class = ax3a.bar(x_3class, sizes_3class, color=colors_3class, edgecolor='white', linewidth=1.5)
ax3a.set_xlabel('Kelas Sentimen', fontsize=12)
ax3a.set_ylabel('Jumlah Ulasan', fontsize=12)
ax3a.set_title('Skema 3-Kelas', fontsize=13, fontweight='bold')
ax3a.set_xticks(x_3class)
ax3a.set_xticklabels(labels_3class)
ax3a.set_ylim(0, max(sizes_3class) * 1.15)

# Tambah label di atas bar
for bar, size in zip(bars_3class, sizes_3class):
    ax3a.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 50, 
              f'{size:,}', ha='center', va='bottom', fontsize=11, fontweight='bold')

# 5-Kelas Bar Chart
ax3b = axes[1]
labels_5_short = ['Sangat\nNegatif', 'Negatif', 'Netral', 'Positif', 'Sangat\nPositif']
x_5class = np.arange(len(labels_5_short))
bars_5class = ax3b.bar(x_5class, sizes_5class, color=colors_5class, edgecolor='white', linewidth=1.5)
ax3b.set_xlabel('Kelas Sentimen', fontsize=12)
ax3b.set_ylabel('Jumlah Ulasan', fontsize=12)
ax3b.set_title('Skema 5-Kelas', fontsize=13, fontweight='bold')
ax3b.set_xticks(x_5class)
ax3b.set_xticklabels(labels_5_short)
ax3b.set_ylim(0, max(sizes_5class) * 1.15)

# Tambah label di atas bar
for bar, size in zip(bars_5class, sizes_5class):
    ax3b.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 50, 
              f'{size:,}', ha='center', va='bottom', fontsize=11, fontweight='bold')

fig3.suptitle('Perbandingan Distribusi Dataset Sentimen', fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('distribusi_perbandingan.png', dpi=300, bbox_inches='tight', facecolor='white')
print("✅ Saved: distribusi_perbandingan.png")

# ============================================
# Print Summary untuk Skripsi
# ============================================
print("\n" + "="*60)
print("RINGKASAN DATA UNTUK SKRIPSI")
print("="*60)
print(f"\n📊 SKEMA 3-KELAS (Total: {len(df_3class):,} ulasan)")
print("-"*40)
for label, count in zip(['Negatif', 'Netral', 'Positif'], sizes_3class):
    pct = count / len(df_3class) * 100
    print(f"   {label:15} : {count:,} ({pct:.1f}%)")

print(f"\n📊 SKEMA 5-KELAS (Total: {len(df_5class):,} ulasan)")
print("-"*40)
for label, count in zip(['Sangat Negatif (1)', 'Negatif (2)', 'Netral (3)', 'Positif (4)', 'Sangat Positif (5)'], sizes_5class):
    pct = count / len(df_5class) * 100
    print(f"   {label:20} : {count:,} ({pct:.1f}%)")

print("\n✅ Semua gambar berhasil disimpan!")

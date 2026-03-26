# ============================================================
#  14_hospital_doctor_report.py
#  Heart Disease Prediction — Hospital & Doctor Analytics
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

plt.rcParams.update({'font.family': 'DejaVu Sans', 'figure.dpi': 120})

print("=" * 65)
print("       HEART DISEASE — HOSPITAL & DOCTOR ANALYTICS")
print("=" * 65)

df = pd.read_csv('data/heart_dataset.csv')

# Hospital & doctor metadata
hosp_meta = {
    'H01': ('Apollo Heart Institute',    'Delhi'),
    'H02': ('Fortis Cardiac Care',       'Mumbai'),
    'H03': ('AIIMS Cardiology',          'Chennai'),
    'H04': ('Max Super Specialty',       'Bangalore'),
    'H05': ('Medanta Heart Institute',   'Gurugram'),
    'H06': ('Narayana Health',           'Kolkata'),
    'H07': ('Wockhardt Hospital',        'Hyderabad'),
    'H08': ('Ruby Hall Clinic',          'Pune'),
}
df['Hospital_Name'] = df['Hospital_ID'].map(lambda h: hosp_meta.get(h, (h,''))[0])
df['City']          = df['Hospital_ID'].map(lambda h: hosp_meta.get(h, ('',h))[1])

# ════════════════════════════════════════════════════════════════════════════
#  Hospital-level Statistics
# ════════════════════════════════════════════════════════════════════════════
hosp_stats = df.groupby(['Hospital_ID','Hospital_Name','City']).agg(
    Patients       = ('Patient_ID', 'count'),
    Disease_Rate   = ('Target', 'mean'),
    Avg_Age        = ('Age', 'mean'),
    Avg_Chol       = ('Cholesterol', 'mean'),
    Avg_BP         = ('Trestbps', 'mean'),
    High_Risk_Pct  = ('Risk_Level', lambda x: (x == 'High Risk').mean() * 100),
).reset_index()
hosp_stats['Disease_Rate'] *= 100
hosp_stats = hosp_stats.sort_values('Disease_Rate', ascending=False)

print("\n  HOSPITAL PERFORMANCE TABLE:")
print(f"  {'Hospital':<28} {'City':<12} {'Patients':>9} {'Dis%':>6} {'HighRisk%':>10} {'AvgAge':>7}")
print("  " + "-" * 77)
for _, r in hosp_stats.iterrows():
    print(f"  {r['Hospital_Name']:<28} {r['City']:<12} {r['Patients']:>9,} "
          f"{r['Disease_Rate']:>6.1f}%  {r['High_Risk_Pct']:>8.1f}%  {r['Avg_Age']:>7.1f}")

# ════════════════════════════════════════════════════════════════════════════
#  Doctor-level Statistics
# ════════════════════════════════════════════════════════════════════════════
doc_stats = df.groupby(['Doctor_ID','Hospital_ID']).agg(
    Patients     = ('Patient_ID', 'count'),
    Disease_Rate = ('Target', 'mean'),
    Avg_Age      = ('Age', 'mean'),
    Avg_Risk_Score = ('Risk_Level', lambda x: (x.isin(['High Risk','Moderate Risk'])).mean()*100),
).reset_index()
doc_stats['Disease_Rate'] *= 100
doc_stats = doc_stats.sort_values('Patients', ascending=False)

print("\n  DOCTOR PERFORMANCE TABLE:")
print(f"  {'Doctor':<10} {'Hospital':<6} {'Patients':>9} {'Dis%':>6} {'At-Risk%':>10} {'AvgAge':>7}")
print("  " + "-" * 50)
for _, r in doc_stats.iterrows():
    print(f"  {r['Doctor_ID']:<10} {r['Hospital_ID']:<6} {r['Patients']:>9,} "
          f"{r['Disease_Rate']:>6.1f}%  {r['Avg_Risk_Score']:>8.1f}%  {r['Avg_Age']:>7.1f}")

# ════════════════════════════════════════════════════════════════════════════
#  Year-over-Year Trend per Hospital
# ════════════════════════════════════════════════════════════════════════════
df['Visit_Date'] = pd.to_datetime(df['Visit_Date'])
df['Year']       = df['Visit_Date'].dt.year
df['Month']      = df['Visit_Date'].dt.month

yoy = df.groupby(['Year','Hospital_ID'])['Target'].agg(['mean','count']).reset_index()
yoy.columns = ['Year','Hospital_ID','Disease_Rate','Patients']
yoy['Disease_Rate'] *= 100

# ════════════════════════════════════════════════════════════════════════════
#  PLOTS
# ════════════════════════════════════════════════════════════════════════════
fig = plt.figure(figsize=(20, 16))
gs  = gridspec.GridSpec(3, 3, figure=fig, hspace=0.45, wspace=0.38)
fig.suptitle('Hospital & Doctor Analytics Dashboard', fontsize=15, fontweight='bold')

# 1. Patient load per hospital
ax1 = fig.add_subplot(gs[0, 0])
sorted_h = hosp_stats.sort_values('Patients')
colors_h = plt.cm.Blues(np.linspace(0.4, 0.9, len(sorted_h)))
ax1.barh(sorted_h['Hospital_Name'], sorted_h['Patients'], color=colors_h)
for i, (_, r) in enumerate(sorted_h.iterrows()):
    ax1.text(r['Patients']+10, i, f'{r["Patients"]:,}', va='center', fontsize=7)
ax1.set_title('Patients per Hospital', fontweight='bold')
ax1.set_xlabel('Patients')

# 2. Disease rate per hospital
ax2 = fig.add_subplot(gs[0, 1])
sorted_d = hosp_stats.sort_values('Disease_Rate')
colors_d = ['#F44336' if v > 55 else '#FF9800' if v > 50 else '#4CAF50' for v in sorted_d['Disease_Rate']]
ax2.barh(sorted_d['Hospital_Name'], sorted_d['Disease_Rate'], color=colors_d)
for i, (_, r) in enumerate(sorted_d.iterrows()):
    ax2.text(r['Disease_Rate']+0.2, i, f'{r["Disease_Rate"]:.1f}%', va='center', fontsize=7)
ax2.set_title('Disease Rate per Hospital', fontweight='bold')
ax2.set_xlabel('Disease Rate (%)')
ax2.axvline(df['Target'].mean()*100, color='black', lw=1.5, linestyle='--', label='Overall avg')
ax2.legend(fontsize=7)

# 3. High risk % per hospital
ax3 = fig.add_subplot(gs[0, 2])
sorted_r = hosp_stats.sort_values('High_Risk_Pct')
colors_r = plt.cm.Reds(np.linspace(0.3, 0.85, len(sorted_r)))
ax3.barh(sorted_r['Hospital_Name'], sorted_r['High_Risk_Pct'], color=colors_r)
ax3.set_title('High Risk Patients %', fontweight='bold')
ax3.set_xlabel('High Risk (%)')

# 4. Year-over-Year disease rate trend per hospital
ax4 = fig.add_subplot(gs[1, :])
pivot_yoy = yoy.pivot(index='Year', columns='Hospital_ID', values='Disease_Rate')
line_colors = plt.cm.tab10(np.linspace(0, 1, len(pivot_yoy.columns)))
for col, color in zip(pivot_yoy.columns, line_colors):
    label = hosp_meta.get(col, (col,''))[0]
    ax4.plot(pivot_yoy.index, pivot_yoy[col], 'o-', lw=2.5, color=color,
             label=label[:20], markersize=6)
ax4.set_title('Year-over-Year Disease Rate Trend by Hospital', fontweight='bold')
ax4.set_xlabel('Year')
ax4.set_ylabel('Disease Rate (%)')
ax4.legend(loc='upper left', fontsize=7, ncol=2)
ax4.set_xticks(pivot_yoy.index)
ax4.grid(True, alpha=0.3)

# 5. Doctor workload bar
ax5 = fig.add_subplot(gs[2, 0:2])
doc_sorted = doc_stats.sort_values('Patients', ascending=True)
doc_colors = plt.cm.Oranges(np.linspace(0.4, 0.9, len(doc_sorted)))
bars = ax5.barh(doc_sorted['Doctor_ID'], doc_sorted['Patients'], color=doc_colors)
for bar in bars:
    ax5.text(bar.get_width()+5, bar.get_y()+bar.get_height()/2,
             f'{bar.get_width():,}', va='center', fontsize=8)
ax5.set_title('Patient Load per Doctor', fontweight='bold')
ax5.set_xlabel('Patients')

# 6. Doctor disease rate scatter
ax6 = fig.add_subplot(gs[2, 2])
sc = ax6.scatter(doc_stats['Patients'], doc_stats['Disease_Rate'],
                 c=doc_stats['Avg_Risk_Score'],
                 cmap='RdYlGn_r', s=120, edgecolors='white', linewidth=1.5)
for _, r in doc_stats.iterrows():
    ax6.annotate(r['Doctor_ID'], (r['Patients'], r['Disease_Rate']),
                 fontsize=7, ha='center', va='bottom')
plt.colorbar(sc, ax=ax6, label='At-Risk %')
ax6.set_title('Patients vs Disease Rate\n(colour = At-Risk %)', fontweight='bold')
ax6.set_xlabel('Patient Count')
ax6.set_ylabel('Disease Rate (%)')

plt.savefig('plots/22_hospital_doctor_analytics.png', bbox_inches='tight', dpi=150)
plt.close()
print("\n✅ Plot: Hospital & doctor analytics saved")

# ════════════════════════════════════════════════════════════════════════════
#  Save
# ════════════════════════════════════════════════════════════════════════════
hosp_stats.to_csv('outputs/hospital_stats.csv', index=False)
doc_stats.to_csv('outputs/doctor_stats.csv', index=False)
print(f"💾 Saved → outputs/hospital_stats.csv & doctor_stats.csv")
print("\n✅ Hospital/Doctor analytics complete.")

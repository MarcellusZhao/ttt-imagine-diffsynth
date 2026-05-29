import matplotlib.pyplot as plt

DATASET_COUNTS = {
    "bdd100k": 500,
    "physicalai-av": 2000,
    "ego4d": 500,
    "panda": 2000,
    "agibotworld2026": 2000,
    "mixkit": 2000,
    "pixabay": 2000,
}

DOMAIN_MAP = {
    "bdd100k": "Driving",
    "physicalai-av": "Driving",
    "ego4d": "Egocentric",
    "panda": "YouTube",
    "agibotworld2026": "Robotics",
    "mixkit": "Stock",
    "pixabay": "Stock",
}

DOMAIN_COLORS = {
    "Driving": "#4E79A7",
    "Egocentric": "#F28E2B",
    "YouTube": "#59A14F",
    "Robotics": "#E15759",
    "Stock": "#76B7B2",
}

domain_counts: dict[str, int] = {}
for dataset, count in DATASET_COUNTS.items():
    domain = DOMAIN_MAP[dataset]
    domain_counts[domain] = domain_counts.get(domain, 0) + count

labels = list(domain_counts.keys())
sizes = list(domain_counts.values())
colors = [DOMAIN_COLORS[d] for d in labels]
total = sum(sizes)

fig, ax = plt.subplots(figsize=(7, 7))

wedges, texts, autotexts = ax.pie(
    sizes,
    # labels=[f"{d}\n{domain_counts[d]:,}" for d in labels],
    labels=[f"{d}" for d in labels],
    colors=colors,
    autopct="%1.1f%%",
    startangle=140,
    pctdistance=0.72,
    labeldistance=1.15,
    wedgeprops=dict(linewidth=1.2, edgecolor="white"),
)

for t in texts:
    t.set_fontsize(12)
    t.set_fontweight("bold")

for t in autotexts:
    t.set_fontsize(12)
    t.set_fontweight("bold")

plt.tight_layout()
plt.savefig("video_dist_pie_chart.png", dpi=150, bbox_inches="tight")
plt.show()
print("Saved to video_dist_pie_chart.png")

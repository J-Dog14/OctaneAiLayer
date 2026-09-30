"""
Research module — deep exploration of the ai_layer / analytics warehouse.

Base substrate: ai_layer.athlete_profiles gives us a per-athlete Z-scored
metric vector across every assessment modality. That's the "wide-format
athlete × metric matrix" a data scientist would spend a week building.

Submodules:
  profile_matrix    — extract the wide DataFrame with metadata
  correlations      — cross-metric Spearman/Pearson + FDR correction
  clustering        — KMeans / PCA for deficit-pattern archetype discovery
  longitudinal      — for athletes with ≥2 profiles: pre/post metric deltas
  program_response  — cross-reference programs with subsequent metric change
  reports           — Plotly interactive HTML report generation
"""

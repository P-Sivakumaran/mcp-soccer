# Stat Dictionary (Selected)

- performance_gls: Goals scored.
- performance_ast: Assists.
- expected_xg: Expected goals.
- expected_xag: Expected assisted goals (xAG).
- standard_sh: Shots.
- standard_sot: Shots on target.
- standard_sot_pct: Shot on target percentage.
- total_cmp_pct: Pass completion percentage.
- progressive_passes: Completed passes that move the ball significantly forward.
- carries_prgc: Progressive carries.
- kp (key_passes): Passes that directly lead to a shot.
- tackles_tkl: Tackles won.
- interceptions: Interceptions made.
- aerial_duels_won_pct: Aerial duel success percentage.
- touches_att_pen: Touches in the attacking penalty area.

Per-90 derived fields (server computes if missing):
- tackles_per_90, interceptions_per_90, shots_per_90,
  key_passes_per_90, progressive_passes_per_90, dribbles_per_90.

Percentile fields:
- For any metric `m`, `m_pctile` is the percentile (0–100) within league+position groups where applicable.

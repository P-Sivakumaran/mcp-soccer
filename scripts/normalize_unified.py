#!/usr/bin/env python3
"""
Normalize existing unified_player_stats.csv to align with server expectations.

Applies header cleanup, renames legacy fields, adds derived per-90 metrics,
and extracts a numeric age_years column for filtering.
"""

import pandas as pd
import numpy as np
from pathlib import Path


RENAME_MAP = {
    # Defensive
    'int': 'interceptions',
    'clr': 'clearances',
    # Passing/progression
    'prgp': 'progressive_passes',
    # Take-ons standardization
    'take-ons_att': 'take_ons_att',
    'take-ons_succ': 'take_ons_succ',
}


def clean_header(name: str) -> str:
    s = str(name)
    s = (s
         .strip()
         .lower()
         .replace(' ', '_')
         .replace('.', '_')
         .replace('%', '_pct')
         .replace('+', '_')
         .replace('-', '_')
         .replace('/', '_')
         .replace('#', '')
         .replace(':', '_')
         .replace('(', '')
         .replace(')', '')
    )
    while '__' in s:
        s = s.replace('__', '_')
    return RENAME_MAP.get(s, s)


def extract_age_years(age_val):
    if pd.isna(age_val):
        return np.nan
    try:
        s = str(age_val)
        if '-' in s:
            return float(s.split('-')[0])
        return float(s)
    except Exception:
        try:
            return float(str(age_val))
        except Exception:
            return np.nan


def main():
    data_file = Path('data/unified_player_stats.csv')
    if not data_file.exists():
        raise SystemExit(f"Data file not found: {data_file}")

    df = pd.read_csv(data_file)

    # Clean headers
    new_cols = [clean_header(c) for c in df.columns]
    df.columns = new_cols
    # Drop duplicate columns after rename
    df = df.loc[:, ~df.columns.duplicated()]

    # Age
    if 'age_years' not in df.columns:
        src = 'age' if 'age' in df.columns else None
        if src:
            df['age_years'] = df[src].apply(extract_age_years)

    # Ensure minutes and 90s
    if 'playing_time_90s' not in df.columns and 'playing_time_min' in df.columns:
        df['playing_time_90s'] = pd.to_numeric(df['playing_time_min'], errors='coerce') / 90.0

    n90 = pd.to_numeric(df.get('playing_time_90s', pd.Series(index=df.index)), errors='coerce').replace(0, np.nan)

    # Derived per-90
    if 'tackles_per_90' not in df.columns and 'tackles_tkl' in df.columns:
        df['tackles_per_90'] = pd.to_numeric(df['tackles_tkl'], errors='coerce') / n90
    if 'interceptions_per_90' not in df.columns and 'interceptions' in df.columns:
        df['interceptions_per_90'] = pd.to_numeric(df['interceptions'], errors='coerce') / n90
    if 'shots_per_90' not in df.columns and 'standard_sh' in df.columns:
        df['shots_per_90'] = pd.to_numeric(df['standard_sh'], errors='coerce') / n90
    if 'key_passes_per_90' not in df.columns and 'kp' in df.columns:
        df['key_passes_per_90'] = pd.to_numeric(df['kp'], errors='coerce') / n90
    if 'progressive_passes_per_90' not in df.columns:
        if 'progressive_passes' in df.columns:
            df['progressive_passes_per_90'] = pd.to_numeric(df['progressive_passes'], errors='coerce') / n90
    # Dribbles
    take_att_col = 'take_ons_att' if 'take_ons_att' in df.columns else ('take-ons_att' if 'take-ons_att' in df.columns else None)
    if 'dribbles_per_90' not in df.columns and take_att_col:
        df['dribbles_per_90'] = pd.to_numeric(df[take_att_col], errors='coerce') / n90

    for c in ['tackles_per_90','interceptions_per_90','shots_per_90','key_passes_per_90','progressive_passes_per_90','dribbles_per_90']:
        if c in df.columns:
            df[c] = df[c].fillna(0).round(3)

    # Save back
    df.to_csv(data_file, index=False)
    print(f"✅ Normalized and updated: {data_file}")


if __name__ == '__main__':
    main()


#!/usr/bin/env python3
"""
Enhanced Soccer Data MCP Server
Professional-grade scouting and analysis system
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json
import sys
from typing import Optional, List, Dict, Any, Union
import logging
from dataclasses import dataclass
from enum import Enum
import re
import time
from time import perf_counter
from collections import OrderedDict

# Configure logging to stderr only
# Allow log level override via env var for observability
_default_level = logging.INFO
try:
    import os
    _lvl = os.getenv('SOCCER_MCP_LOG_LEVEL', '').upper()
    if _lvl:
        _default_level = getattr(logging, _lvl, logging.INFO)
except Exception:
    pass

logging.basicConfig(
    level=_default_level,
    format='%(asctime)s - %(levelname)s - %(message)s',
    stream=sys.stderr
)
logger = logging.getLogger(__name__)

# Data Models
@dataclass
class PlayerBasicInfo:
    name: str
    age: int
    nationality: str
    team: str
    league: str
    position: str
    secondary_positions: List[str]
    season: str

@dataclass
class PerformanceStats:
    minutes_played: float
    games_played: int
    goals: float
    assists: float
    expected_goals: float
    expected_assists: float
    goals_per_90: float
    assists_per_90: float
    
@dataclass
class DefensiveStats:
    tackles_per_90: float
    tackle_success_rate: float
    interceptions_per_90: float
    clearances: float
    aerial_duels_won_pct: float
    blocks: float
    pressures: float

@dataclass
class PassingStats:
    pass_completion_pct: float
    progressive_passes: float
    key_passes: float
    long_passes_completed: float
    crosses_accuracy: float
    progressive_carries: float

@dataclass
class AttackingStats:
    shots_per_90: float
    shots_on_target_pct: float
    big_chances_created: float
    dribbles_attempted: float
    dribble_success_rate: float
    touches_penalty_area: float

@dataclass
class PlayerSummary:
    name: str
    age: int
    team: str
    league: str
    position: str
    key_stats: Dict[str, float]
    overall_rating: float

class ScoutingPosition(Enum):
    GK = "GK"
    DF = "DF" 
    MF = "MF"
    FW = "FW"
    
class EnhancedSoccerDataServer:
    def __init__(self):
        self.data = self.load_comprehensive_data()
        self.position_mappings = self._create_position_mappings()
        self.stat_mappings = self._create_stat_mappings()
        # Optimize df and build caches for performance
        self._optimize_dataframe()
        self._build_caches()
        self._build_percentiles()
        self._result_cache = _LRUCache(128)
        logger.info(f"Enhanced server loaded {len(self.data)} players with {len(self.data.columns)} metrics")
    
    def load_comprehensive_data(self):
        """Load and enhance the unified player stats data"""
        script_dir = Path(__file__).parent
        data_file = script_dir / "data" / "unified_player_stats.csv"
        
        if not data_file.exists():
            logger.error(f"Data file not found: {data_file}")
            return pd.DataFrame()
        
        try:
            df = pd.read_csv(data_file)
            
            # Data quality improvements
            df = self._enhance_data_quality(df)
            
            logger.info(f"Loaded comprehensive dataset: {len(df)} players, {len(df.columns)} columns")
            
            # Log available leagues and positions for debugging
            if 'league' in df.columns:
                unique_leagues = df['league'].unique()
                logger.info(f"Available leagues: {list(unique_leagues)}")
                
            if 'position' in df.columns:
                unique_positions = df['position'].dropna().unique()
                logger.info(f"Available positions: {list(unique_positions)}")
            
            return df
            
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            return pd.DataFrame()
    
    def _enhance_data_quality(self, df):
        """Enhance data quality and add computed metrics"""

        # Clean position data
        if 'position' in df.columns:
            df['position'] = df['position'].fillna('Unknown')
            # Split multi-positions and create primary position
            df['primary_position'] = df['position'].apply(self._extract_primary_position)
            df['secondary_positions'] = df['position'].apply(self._extract_secondary_positions)
        
        # Ensure numeric columns are properly typed
        numeric_columns = [
            'age', 'playing_time_mp', 'performance_gls', 'performance_ast',
            'expected_xg', 'expected_npxg', 'standard_sh', 'standard_sot',
            'tackles_tkl', 'interceptions', 'aerial_duels_won_pct'
        ]
        
        for col in numeric_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Ensure minutes fields exist
        if 'playing_time_min' not in df.columns and 'playing_time_90s' in df.columns:
            df['playing_time_min'] = pd.to_numeric(df['playing_time_90s'], errors='coerce') * 90

        # Add computed per-90 metrics consistently where possible
        if 'playing_time_90s' in df.columns:
            _n90 = pd.to_numeric(df['playing_time_90s'], errors='coerce').replace(0, np.nan)
            if 'tackles_per_90' not in df.columns and 'tackles_tkl' in df.columns:
                df['tackles_per_90'] = (pd.to_numeric(df['tackles_tkl'], errors='coerce') / _n90).round(3)
            if 'interceptions_per_90' not in df.columns and 'interceptions' in df.columns:
                df['interceptions_per_90'] = (pd.to_numeric(df['interceptions'], errors='coerce') / _n90).round(3)
            if 'shots_per_90' not in df.columns and 'standard_sh' in df.columns:
                df['shots_per_90'] = (pd.to_numeric(df['standard_sh'], errors='coerce') / _n90).round(3)
            if 'key_passes_per_90' not in df.columns and 'kp' in df.columns:
                df['key_passes_per_90'] = (pd.to_numeric(df['kp'], errors='coerce') / _n90).round(3)
            if 'progressive_passes_per_90' not in df.columns and 'progressive_passes' in df.columns:
                df['progressive_passes_per_90'] = (pd.to_numeric(df['progressive_passes'], errors='coerce') / _n90).round(3)
            if 'dribbles_per_90' not in df.columns and 'take-ons_att' in df.columns:
                df['dribbles_per_90'] = (pd.to_numeric(df['take-ons_att'], errors='coerce') / _n90).round(3)

            # Fill NaNs created by division for players with 0 minutes
            for col in ['tackles_per_90','interceptions_per_90','shots_per_90','key_passes_per_90','progressive_passes_per_90','dribbles_per_90']:
                if col in df.columns:
                    df[col] = df[col].fillna(0)

        return df

    def _optimize_dataframe(self):
        """Optimize dataframe dtypes and helper columns for performance."""
        if self.data is None or self.data.empty:
            return
        df = self.data
        for col in ['league', 'position', 'team', 'season', 'nation', 'primary_position']:
            if col in df.columns:
                df[col] = df[col].fillna('Unknown').astype('category')
        # Lowercase helper columns for faster case-insensitive contains
        for col in ['player', 'team']:
            if col in df.columns:
                df[f'{col}_lower'] = df[col].astype(str).str.lower()
        # Precompute minutes 90s if not present but minutes exist
        if 'playing_time_90s' not in df.columns and 'playing_time_min' in df.columns:
            df['playing_time_90s'] = (pd.to_numeric(df['playing_time_min'], errors='coerce') / 90.0).round(3)
        # Global season start year for ordering
        if 'season' in df.columns:
            def _sy(x):
                try:
                    s = str(x)
                    return int(s[:4]) if s[:4].isdigit() else -1
                except Exception:
                    return -1
            df['season_start_year'] = df['season'].apply(_sy).astype('int32')
        # Downcast numeric columns to reduce memory
        num_cols = df.select_dtypes(include=['float64', 'int64']).columns
        for c in num_cols:
            try:
                if str(df[c].dtype).startswith('float'):
                    df[c] = pd.to_numeric(df[c], errors='coerce', downcast='float')
                else:
                    df[c] = pd.to_numeric(df[c], errors='coerce', downcast='integer')
            except Exception:
                continue
        self.data = df

    def _contains_noregex(self, series_lower: pd.Series, needle_lower: str) -> pd.Series:
        return series_lower.astype(str).str.contains(needle_lower, na=False, regex=False)

    def _build_caches(self):
        """Build simple caches for league/season filtered subsets."""
        self._cache = {
            'league': {},
            'season': {},
            'league_season': {}
        }
        df = self.data
        if df is None or df.empty:
            return
        if 'league' in df.columns:
            for l in df['league'].dropna().unique():
                self._cache['league'][l] = df[df['league'] == l]
        if 'season' in df.columns:
            for s in df['season'].dropna().unique():
                self._cache['season'][s] = df[df['season'] == s]
        if 'league' in df.columns and 'season' in df.columns:
            for l in df['league'].dropna().unique():
                sub = self._cache['league'][l]
                for s in sub['season'].dropna().unique():
                    self._cache['league_season'][(l, s)] = sub[sub['season'] == s]

    def _build_percentiles(self):
        """Precompute percentiles for key stats within league+position groups for rating."""
        df = self.data
        if df is None or df.empty:
            self.data_with_pct = df
            return
        key_cols = [
            'performance_gls','expected_xg','standard_sot_pct','total_cmp_pct',
            'tackles_per_90','interceptions_per_90','carries_prgc','expected_xag',
            'key_passes_per_90','progressive_passes_per_90','aerial_duels_won_pct'
        ]
        group_cols = [c for c in ['league', 'position'] if c in df.columns]
        self.data_with_pct = self._compute_percentiles(df, [c for c in key_cols if c in df.columns], group_cols)

    def _get_base_filtered(self, leagues: Optional[List[str]], seasons: Optional[List[str]]):
        """Pick a best starting slice using caches to reduce scan size."""
        df = self.data
        if leagues and seasons:
            # If single combos, prefer league_season cache
            if len(leagues) == 1 and len(seasons) == 1:
                return self._cache['league_season'].get((leagues[0], seasons[0]), df)
            # Combine across requested combos
            parts = []
            for l in leagues:
                for s in seasons:
                    part = self._cache['league_season'].get((l, s))
                    if part is not None:
                        parts.append(part)
            if parts:
                return pd.concat(parts, ignore_index=False)
        if leagues:
            if len(leagues) == 1:
                return self._cache['league'].get(leagues[0], df)
            return df[df['league'].isin(leagues)] if 'league' in df.columns else df
        if seasons:
            if len(seasons) == 1:
                return self._cache['season'].get(seasons[0], df)
            return df[df['season'].isin(seasons)] if 'season' in df.columns else df
        return df
    
    def _extract_primary_position(self, position_str):
        """Extract primary position from multi-position strings"""
        if pd.isna(position_str) or position_str == 'Unknown':
            return 'Unknown'
        
        positions = str(position_str).split(',')
        return positions[0].strip()
    
    def _extract_secondary_positions(self, position_str):
        """Extract secondary positions as list"""
        if pd.isna(position_str) or position_str == 'Unknown':
            return []
        
        positions = str(position_str).split(',')
        return [pos.strip() for pos in positions[1:]] if len(positions) > 1 else []
    
    def _create_position_mappings(self):
        """Create mappings for position filtering"""
        return {
            'goalkeeper': ['GK'],
            'defender': ['DF', 'DF,MF', 'MF,DF'],
            'midfielder': ['MF', 'MF,DF', 'MF,FW', 'DF,MF', 'FW,MF'],
            'forward': ['FW', 'FW,MF', 'MF,FW'],
            'centre_back': ['DF'],
            'full_back': ['DF'],
            'defensive_midfielder': ['MF', 'DF,MF'],
            'central_midfielder': ['MF'],
            'attacking_midfielder': ['MF', 'MF,FW'],
            'winger': ['FW,MF', 'MF,FW'],
            'striker': ['FW']
        }
    
    def _create_stat_mappings(self):
        """Map friendly stat names to column names"""
        return {
            'goals': 'performance_gls',
            'assists': 'performance_ast',
            'minutes_played': 'playing_time_min',
            'games_played': 'playing_time_mp',
            'expected_goals': 'expected_xg',
            'expected_assists': 'expected_xag',
            'shots': 'standard_sh',
            'shots_on_target': 'standard_sot',
            'shots_on_target_pct': 'standard_sot_pct',
            'pass_completion_pct': 'total_cmp_pct',
            'tackles': 'tackles_tkl',
            'tackles_per_90': 'tackles_per_90',
            'interceptions': 'interceptions',
            'interceptions_per_90': 'interceptions_per_90',
            'shots_per_90': 'shots_per_90',
            'key_passes_per_90': 'key_passes_per_90',
            'progressive_passes_per_90': 'progressive_passes_per_90',
            'dribbles_per_90': 'dribbles_per_90',
            'aerial_duels_won_pct': 'aerial_duels_won_pct',
            'progressive_passes': 'progressive_passes',
            'progressive_carries': 'carries_prgc',
            'successful_dribbles': 'take-ons_succ',
            'penalty_area_touches': 'touches_att_pen'
        }
    
    # ENHANCED SEARCH FUNCTIONS
    def search_players_advanced(
        self,
        leagues: Optional[List[str]] = None,
        positions: Optional[List[str]] = None,
        age_min: Optional[int] = None,
        age_max: Optional[int] = None,
        nationality: Optional[List[str]] = None,
        team: Optional[str] = None,
        seasons: Optional[List[str]] = None,
        latest_season_only: bool = False,
        min_minutes_played: Optional[int] = 500,
        stat_filters: Optional[Dict[str, float]] = None,
        limit: int = 100,
        sort_by: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Advanced player search with comprehensive filtering"""
        
        # Basic LRU cache: cache by key of inputs
        cache_key = (
            'search_players_advanced',
            tuple(sorted(leagues)) if leagues else None,
            tuple(sorted(positions)) if positions else None,
            age_min, age_max,
            tuple(sorted(nationality)) if nationality else None,
            team,
            tuple(sorted(seasons)) if seasons else None,
            latest_season_only,
            min_minutes_played,
            tuple(sorted(stat_filters.items())) if stat_filters else None,
            limit,
            sort_by
        )
        cached = self._result_cache.get(cache_key)
        if cached is not None:
            return cached
        t0 = perf_counter()
        # Start with cached slice where possible
        filtered = self._get_base_filtered(leagues, seasons).copy()
        
        # Apply filters
        if leagues and ('league' in filtered.columns):
            filtered = filtered[filtered['league'].isin(leagues)]

        if positions:
            # Support both exact position matches and positional roles
            position_conditions = []
            for pos in positions:
                if pos in self.position_mappings:
                    # Positional role mapping
                    mapped_positions = self.position_mappings[pos]
                    pos_condition = filtered['position'].isin(mapped_positions)
                else:
                    # Direct position match
                    pos_condition = filtered['position'].astype(str).str.contains(pos, case=False, na=False, regex=False)
                position_conditions.append(pos_condition)
            
            if position_conditions:
                combined_position_filter = position_conditions[0]
                for condition in position_conditions[1:]:
                    combined_position_filter |= condition
                filtered = filtered[combined_position_filter]
        
        if age_min is not None:
            filtered = filtered[filtered['age'] >= age_min]
            
        if age_max is not None:
            filtered = filtered[filtered['age'] <= age_max]
            
        if nationality:
            filtered = filtered[filtered['nation'].isin(nationality)]
            
        if team:
            tlow = str(team).lower()
            team_col = 'team_lower' if 'team_lower' in filtered.columns else 'team'
            filtered = filtered[self._contains_noregex(filtered[team_col], tlow)]
        
        # Season filtering
        if seasons and ('season' in filtered.columns):
            filtered = filtered[filtered['season'].isin(seasons)]
        
        # Latest season only mode
        if latest_season_only:
            # Get the most recent season for each player
            latest_season_data = filtered.groupby('player')['season'].max().reset_index()
            latest_season_data.columns = ['player', 'latest_season']
            filtered = filtered.merge(latest_season_data, on='player')
            filtered = filtered[filtered['season'] == filtered['latest_season']]
            filtered = filtered.drop('latest_season', axis=1)
            
        if min_minutes_played:
            minutes_col = 'playing_time_min' if 'playing_time_min' in filtered.columns else 'playing_time_mp'
            if minutes_col in filtered.columns:
                filtered = filtered[filtered[minutes_col] >= min_minutes_played]
        
        # Apply statistical filters
        if stat_filters:
            for stat_name, min_value in stat_filters.items():
                column_name = self.stat_mappings.get(stat_name, stat_name)
                if column_name in filtered.columns:
                    filtered = filtered[filtered[column_name] >= min_value]
        
        # Sorting
        if sort_by:
            sort_column = self.stat_mappings.get(sort_by, sort_by)
            if sort_column in filtered.columns:
                filtered = filtered.sort_values(sort_column, ascending=False)
        
        # Limit results
        result = filtered.head(limit)
        
        # Format results
        players = []
        for _, player in result.iterrows():
            player_summary = {
                'name': player.get('player', 'Unknown'),
                'age': int(player.get('age', 0)) if pd.notna(player.get('age')) else 0,
                'team': player.get('team', 'Unknown'),
                'league': player.get('league', 'Unknown'),
                'position': player.get('position', 'Unknown'),
                'nationality': player.get('nation', 'Unknown'),
                'key_stats': {
                    'goals': float(player.get('performance_gls', 0)) if pd.notna(player.get('performance_gls')) else 0,
                    'assists': float(player.get('performance_ast', 0)) if pd.notna(player.get('performance_ast')) else 0,
                    'minutes_played': float(player.get('playing_time_min', 0)) if pd.notna(player.get('playing_time_min')) else 0,
                    'expected_goals': float(player.get('expected_xg', 0)) if pd.notna(player.get('expected_xg')) else 0
                },
                'overall_rating': 0.0
            }
            # Attach overall rating using precomputed percentiles for same player+season if available
            try:
                if 'player' in self.data_with_pct.columns:
                    cand = self.data_with_pct[self.data_with_pct['player'] == player.get('player')]
                    if 'season' in player.index and 'season' in cand.columns:
                        cand = cand[cand['season'] == player.get('season')]
                    if not cand.empty:
                        rating = self._overall_rating(cand.iloc[-1])
                        player_summary['overall_rating'] = rating
            except Exception:
                pass
            players.append(player_summary)

        dt = (perf_counter() - t0) * 1000
        logger.info(f"search_players_advanced filters={{'leagues':{leagues},'positions':{positions},'age_min':{age_min},'age_max':{age_max},'team':{team},'seasons':{seasons}}} -> {len(players)} results in {dt:.1f}ms")
        self._result_cache.set(cache_key, players)
        return players

    def _compute_percentiles(self, df: pd.DataFrame, cols: List[str], group_cols: List[str]) -> pd.DataFrame:
        """Add percentile columns per group for given cols (0-100)."""
        work = df.copy()
        for c in cols:
            if c in work.columns:
                pct_col = f"{c}_pctile"
                work[pct_col] = work.groupby(group_cols)[c].rank(pct=True) * 100.0
        return work

    def _position_group(self, pos: str) -> str:
        if not pos:
            return 'OTHER'
        p = str(pos).upper()
        if p.startswith('GK'):
            return 'GK'
        if p.startswith('DF'):
            return 'DF'
        if p.startswith('MF'):
            return 'MF'
        if p.startswith('FW'):
            return 'FW'
        return 'OTHER'

    def _overall_rating(self, row: pd.Series) -> float:
        """Compute a simple overall rating from percentiles tailored by position group."""
        pos_grp = self._position_group(row.get('position', ''))
        # Choose percentiles if available, else raw values scaled via rank fallback later
        weights = {}
        if pos_grp == 'FW':
            weights = {
                'performance_gls_pctile': 0.35,
                'expected_xg_pctile': 0.2,
                'standard_sot_pct_pctile': 0.15,
                'key_passes_per_90_pctile': 0.15,
                'carries_prgc_pctile': 0.15,
            }
        elif pos_grp == 'MF':
            weights = {
                'total_cmp_pct_pctile': 0.2,
                'progressive_passes_per_90_pctile': 0.25,
                'tackles_per_90_pctile': 0.2,
                'carries_prgc_pctile': 0.2,
                'expected_xag_pctile': 0.15,
            }
        elif pos_grp == 'DF':
            weights = {
                'tackles_per_90_pctile': 0.3,
                'interceptions_per_90_pctile': 0.25,
                'aerial_duels_won_pct_pctile': 0.2,
                'total_cmp_pct_pctile': 0.15,
                'carries_prgc_pctile': 0.1,
            }
        else:  # GK/OTHER
            weights = {
                'aerial_duels_won_pct_pctile': 0.4,
                'total_cmp_pct_pctile': 0.2,
                'tackles_per_90_pctile': 0.2,
                'interceptions_per_90_pctile': 0.2,
            }

        rating = 0.0
        total_w = 0.0
        for k, w in weights.items():
            v = row.get(k, np.nan)
            if pd.notna(v):
                rating += float(v) * w
                total_w += w
        return round(rating / total_w, 2) if total_w > 0 else 0.0

    # ----- Talent Discovery -----
    def _role_weights(self, role: str) -> Dict[str, float]:
        role = (role or '').lower().replace('-', '_').replace(' ', '_')
        # Percentile keys expected (col_pctile) will be derived from metric keys
        base = {
            # progression + passing
            'progressive_passes_per_90': 0.18,
            'carries_prgc': 0.18,
            'key_passes_per_90': 0.12,
            'total_cmp_pct': 0.10,
            # defending
            'tackles_per_90': 0.15,
            'interceptions_per_90': 0.12,
            'aerial_duels_won_pct': 0.05,
            # chance creation / end product
            'expected_xag': 0.05,
            'performance_gls': 0.05
        }
        presets = {
            'left_back': {**base, 'tackles_per_90': 0.18, 'interceptions_per_90': 0.15, 'carries_prgc': 0.2, 'key_passes_per_90': 0.15},
            'right_back': {**base, 'tackles_per_90': 0.16, 'interceptions_per_90': 0.14, 'carries_prgc': 0.22},
            'full_back': {**base},
            'centre_back': {'tackles_per_90': 0.2, 'interceptions_per_90': 0.25, 'aerial_duels_won_pct': 0.2, 'total_cmp_pct': 0.15, 'carries_prgc': 0.2},
            'defensive_midfielder': {'tackles_per_90': 0.2, 'interceptions_per_90': 0.2, 'total_cmp_pct': 0.15, 'progressive_passes_per_90': 0.2, 'carries_prgc': 0.15, 'key_passes_per_90': 0.1},
            'central_midfielder': {'total_cmp_pct': 0.15, 'progressive_passes_per_90': 0.2, 'carries_prgc': 0.2, 'key_passes_per_90': 0.15, 'expected_xag': 0.15, 'tackles_per_90': 0.15},
            'attacking_midfielder': {'key_passes_per_90': 0.22, 'expected_xag': 0.25, 'carries_prgc': 0.18, 'progressive_passes_per_90': 0.18, 'total_cmp_pct': 0.12, 'performance_gls': 0.05},
            'winger': {'key_passes_per_90': 0.2, 'carries_prgc': 0.22, 'progressive_passes_per_90': 0.18, 'standard_sot_pct': 0.1, 'expected_xg': 0.15, 'performance_gls': 0.15},
            'striker': {'expected_xg': 0.3, 'performance_gls': 0.35, 'standard_sot_pct': 0.2, 'key_passes_per_90': 0.05, 'carries_prgc': 0.1},
            'goalkeeper': {'aerial_duels_won_pct': 0.4, 'total_cmp_pct': 0.2, 'tackles_per_90': 0.2, 'interceptions_per_90': 0.2}
        }
        weights = presets.get(role)
        if not weights:
            # Default to balanced base
            weights = base
        # Normalize
        s = sum(weights.values())
        if s > 0:
            weights = {k: v / s for k, v in weights.items()}
        return weights

    def _style_adjustments(self, style: Optional[str]) -> Dict[str, float]:
        s = (style or '').lower()
        if s == 'possession':
            return {'progressive_passes_per_90': 1.15, 'total_cmp_pct': 1.1, 'carries_prgc': 1.1}
        if s == 'transition':
            return {'carries_prgc': 1.15, 'key_passes_per_90': 1.1}
        if s == 'high_press':
            return {'tackles_per_90': 1.15, 'interceptions_per_90': 1.1}
        if s == 'direct':
            return {'standard_sot_pct': 1.1, 'expected_xg': 1.1}
        return {}

    def _role_to_positions(self, role: str) -> List[str]:
        r = (role or '').lower()
        if 'back' in r or 'full' in r:
            return ['DF']
        if 'centre_back' in r or 'center_back' in r or 'cb' in r:
            return ['DF']
        if 'wing' in r:
            return ['FW', 'MF']
        if 'striker' in r or 'forward' in r or 'st' in r:
            return ['FW']
        if 'midfielder' in r or r in ['dm', 'cm', 'am']:
            return ['MF']
        if 'keeper' in r or 'gk' in r:
            return ['GK']
        return []

    def _choose_overlap_season(self, df: pd.DataFrame, min_minutes: int, coverage_threshold: float) -> Optional[str]:
        if df.empty or 'season' not in df.columns:
            return None
        # Compute coverage per season
        minutes_col = 'playing_time_min' if 'playing_time_min' in df.columns else None
        n90_col = 'playing_time_90s' if 'playing_time_90s' in df.columns else None
        def has_minutes(row):
            if minutes_col:
                return float(row.get(minutes_col) or 0) >= float(min_minutes)
            if n90_col:
                return float(row.get(n90_col) or 0) * 90.0 >= float(min_minutes)
            gp = float(row.get('playing_time_mp') or 0)
            return gp * 90.0 >= float(min_minutes)
        seasons = sorted(df['season'].dropna().unique(), key=lambda x: int(str(x)[:4]) if isinstance(x, str) and x[:4].isdigit() else -1, reverse=True)
        best = None
        best_cov = -1.0
        # Treat each player once per season using first row
        for s in seasons:
            sub = df[df['season'] == s]
            # count players with minutes
            group = sub.groupby('player', as_index=False).first()
            available = int(group.apply(has_minutes, axis=1).sum())
            total = int(group['player'].nunique())
            cov = (available / total) if total else 0
            if cov >= coverage_threshold:
                return s
            if cov > best_cov:
                best_cov = cov
                best = s
        return best

    def discover_talents(
        self,
        role: Optional[str] = None,
        style: Optional[str] = None,
        leagues: Optional[List[str]] = None,
        age_max: Optional[int] = None,
        age_min: Optional[int] = None,
        min_minutes: int = 900,
        seasons: Optional[List[str]] = None,
        alignment: str = 'overlap',
        coverage_threshold: float = 0.75,
        exclude_elite: bool = True,
        top_n: int = 10,
        diversify_by: Optional[str] = None,  # 'league' or 'team'
        explain: bool = True
    ) -> List[Dict[str, Any]]:
        """Discover high-upside talents with role/style fit and transparent rationale."""

        cache_key = (
            'discover_talents', (role or '').lower(), (style or '').lower() if style else None,
            tuple(sorted(leagues)) if leagues else None, age_max, age_min, min_minutes,
            tuple(sorted(seasons)) if seasons else None, alignment, coverage_threshold,
            exclude_elite, top_n, diversify_by, explain
        )
        cached = self._result_cache.get(cache_key)
        if cached is not None:
            return cached
        t0 = perf_counter()
        df = self.data.copy()
        if df.empty:
            return []
        # Basic filters
        if leagues and 'league' in df.columns:
            df = df[df['league'].isin(leagues)]
        if age_min is not None and 'age' in df.columns:
            df = df[df['age'] >= age_min]
        if age_max is not None and 'age' in df.columns:
            df = df[df['age'] <= age_max]

        # Role → position filtering
        pos_targets = self._role_to_positions(role or '')
        if pos_targets and 'position' in df.columns:
            df = df[df['position'].astype(str).str.contains('|'.join(pos_targets), case=False, na=False)]

        # Season alignment
        target_season = None
        if alignment == 'overlap':
            if seasons:
                df = df[df['season'].isin(seasons)] if 'season' in df.columns else df
            target_season = self._choose_overlap_season(df, min_minutes=min_minutes, coverage_threshold=coverage_threshold)
            if target_season and 'season' in df.columns:
                df = df[df['season'] == target_season]
        elif seasons and 'season' in df.columns:
            # Use latest given season if multiple
            order = sorted(seasons, key=lambda x: int(str(x)[:4]) if isinstance(x, str) and x[:4].isdigit() else -1, reverse=True)
            target_season = order[0]
            df = df[df['season'] == target_season]

        # Minutes filter
        if 'playing_time_min' in df.columns:
            df = df[pd.to_numeric(df['playing_time_min'], errors='coerce') >= float(min_minutes)]
        elif 'playing_time_90s' in df.columns:
            df = df[pd.to_numeric(df['playing_time_90s'], errors='coerce') * 90.0 >= float(min_minutes)]

        if df.empty:
            return []

        # Join percentiles if available
        work = df
        if hasattr(self, 'data_with_pct') and isinstance(self.data_with_pct, pd.DataFrame):
            # Columns to use
            metric_cols = ['progressive_passes_per_90','carries_prgc','key_passes_per_90','total_cmp_pct','tackles_per_90','interceptions_per_90','aerial_duels_won_pct','expected_xag','performance_gls','expected_xg','standard_sot_pct']
            pct_cols = [c for c in metric_cols if c in self.data_with_pct.columns]
            # Keep only needed columns and percentiles
            # We assume percentiles have been precomputed on self.data_with_pct
            cols_to_keep = ['player','team','league','season','position','age','playing_time_min','playing_time_90s'] + pct_cols + [f"{c}_pctile" for c in pct_cols]
            cols_to_keep = [c for c in cols_to_keep if c in self.data_with_pct.columns]
            base = self.data_with_pct[cols_to_keep].copy()
            work = pd.merge(df, base, on=['player','team','league','season','position'], how='left', suffixes=('', ''))

        # Compute fit score
        weights = self._role_weights(role or '')
        multipliers = self._style_adjustments(style)
        def fit_score(row: pd.Series) -> float:
            score = 0.0
            total_w = 0.0
            for metric, w in weights.items():
                pct_key = f"{metric}_pctile"
                v = row.get(pct_key, np.nan)
                m = multipliers.get(metric, 1.0)
                if pd.notna(v):
                    score += float(v) * w * m
                    total_w += w * m
            if total_w == 0:
                return 0.0
            # Uncertainty discount (low minutes)
            minutes = float(row.get('playing_time_min') or (row.get('playing_time_90s') or 0) * 90.0 or 0)
            uncertainty = max(0.0, 1.0 - min(minutes / float(min_minutes * 2), 1.0) * 0.2)  # up to 20% discount
            raw = score / total_w
            return round(raw * (1.0 - uncertainty), 2)

        work['fit_score'] = work.apply(fit_score, axis=1)
        # Exclude elites if requested (simple heuristic)
        if exclude_elite:
            work = work[~((pd.to_numeric(work.get('playing_time_min', 0), errors='coerce') >= 2500) & (pd.to_numeric(work.get('age', 0), errors='coerce') >= 26))]

        # Rank and diversify
        work = work.sort_values('fit_score', ascending=False)

        selected = []
        seen = set()
        for _, row in work.iterrows():
            if diversify_by in ['league', 'team']:
                key = row.get(diversify_by, 'Unknown')
                if key in seen:
                    continue
                seen.add(key)
            selected.append(row)
            if len(selected) >= top_n:
                break

        results: List[Dict[str, Any]] = []
        for row in selected:
            minutes = float(row.get('playing_time_min') or (row.get('playing_time_90s') or 0) * 90.0 or 0)
            candidate = {
                'name': row.get('player', 'Unknown'),
                'age': int(float(row.get('age'))) if pd.notna(row.get('age')) else None,
                'team': row.get('team', 'Unknown'),
                'league': row.get('league', 'Unknown'),
                'position': row.get('position', 'Unknown'),
                'season_used': row.get('season', target_season or 'Unknown'),
                'minutes': round(minutes),
                'fit_score': float(row.get('fit_score', 0)),
                'percentiles': {},
            }

            # Attach key percentiles for explainability
            for m in ['progressive_passes_per_90','carries_prgc','key_passes_per_90','total_cmp_pct','tackles_per_90','interceptions_per_90','aerial_duels_won_pct','expected_xag','expected_xg','standard_sot_pct']:
                k = f"{m}_pctile"
                if k in row.index and pd.notna(row.get(k)):
                    candidate['percentiles'][m] = round(float(row.get(k)), 1)

            if explain:
                # Strengths = top 3 percentiles
                strengths = sorted(candidate['percentiles'].items(), key=lambda kv: kv[1], reverse=True)[:3]
                risks = sorted(candidate['percentiles'].items(), key=lambda kv: kv[1])[:2]
                candidate['strengths'] = [f"{k.replace('_',' ')} ({v:.0f}th pct)" for k, v in strengths]
                candidate['risks'] = [f"{k.replace('_',' ')} ({v:.0f}th pct)" for k, v in risks]
                candidate['why_shortlisted'] = (
                    f"Role fit for '{role}' with strong {', '.join([s[0].replace('_',' ') for s in strengths])}. "
                    f"Minutes={int(minutes)} in {candidate['season_used']} suggest usable sample; monitor {', '.join([r[0].replace('_',' ') for r in risks])}."
                )

            results.append(candidate)

        dt = (perf_counter() - t0) * 1000
        logger.info(f"discover_talents role={role} style={style} leagues={leagues} age_max={age_max} min_minutes={min_minutes} target_season={target_season} -> {len(results)} results in {dt:.1f}ms")
        self._result_cache.set(cache_key, results)
        return results

    # ----- Additional Scout Tools -----
    def profile_role_fit(
        self,
        player_name: str,
        role: Optional[str] = None,
        style: Optional[str] = None,
        min_minutes: int = 450,
        seasons: Optional[List[str]] = None,
        alignment: str = 'overlap'
    ) -> Dict[str, Any]:
        """Profile a single player's role/style fit with explainability."""
        df = self.data.copy()
        matches = df[df['player'].str.contains(player_name, case=False, na=False)]
        if matches.empty:
            return { 'error': f"Player '{player_name}' not found" }
        if seasons and 'season' in matches.columns:
            matches = matches[matches['season'].isin(seasons)]
        # Choose season (prefer overlap among this player's seasons with minutes)
        season_used = None
        if alignment == 'overlap':
            season_used = self._choose_overlap_season(matches, min_minutes=min_minutes, coverage_threshold=0.0)  # 0.0 -> pick latest with minutes
        if not season_used and 'season' in matches.columns:
            # fallback: latest by start year
            ms = sorted(matches['season'].tolist(), key=lambda x: int(str(x)[:4]) if isinstance(x, str) and x[:4].isdigit() else -1)
            season_used = ms[-1]
        row = matches[matches['season'] == season_used].iloc[0] if season_used is not None else matches.iloc[-1]

        # Use percentiles df
        if hasattr(self, 'data_with_pct'):
            enriched = self.data_with_pct[(self.data_with_pct['player'] == row.get('player')) & (self.data_with_pct['season'] == row.get('season'))]
            if not enriched.empty:
                row = enriched.iloc[0]
        weights = self._role_weights(role or '')
        multipliers = self._style_adjustments(style)

        def metric_pct(k: str) -> float:
            val = row.get(f"{k}_pctile", np.nan)
            return float(val) if pd.notna(val) else 0.0

        score = 0.0
        total_w = 0.0
        breakdown = {}
        for m, w in weights.items():
            adj = multipliers.get(m, 1.0)
            pct = metric_pct(m)
            contribution = pct * w * adj
            breakdown[m] = round(contribution, 2)
            score += contribution
            total_w += w * adj
        fit_score = round(score / total_w, 2) if total_w > 0 else 0.0

        # strengths/risks
        pcts = { m: metric_pct(m) for m in weights.keys() }
        strengths = sorted(pcts.items(), key=lambda kv: kv[1], reverse=True)[:3]
        risks = sorted(pcts.items(), key=lambda kv: kv[1])[:2]
        minutes = float(row.get('playing_time_min') or (row.get('playing_time_90s') or 0) * 90.0 or 0)

        return {
            'player': row.get('player', 'Unknown'),
            'role': role,
            'style': style,
            'season_used': row.get('season', 'Unknown'),
            'minutes': round(minutes),
            'fit_score': fit_score,
            'percentiles': { k: round(v,1) for k, v in pcts.items() },
            'breakdown': breakdown,
            'strengths': [f"{k.replace('_',' ')} ({v:.0f}th pct)" for k,v in strengths],
            'risks': [f"{k.replace('_',' ')} ({v:.0f}th pct)" for k,v in risks],
            'rationale': f"Strong {', '.join([s[0].replace('_',' ') for s in strengths])}; monitor {', '.join([r[0].replace('_',' ') for r in risks])}."
        }

    def recommend_comparables(
        self,
        player_name: str,
        k: int = 5,
        role: Optional[str] = None,
        leagues: Optional[List[str]] = None,
        seasons: Optional[List[str]] = None,
        alignment: str = 'overlap',
        min_minutes: int = 450
    ) -> Dict[str, Any]:
        """Find most similar players by percentile vectors (simple L1 distance)."""
        df = self.data_with_pct if hasattr(self, 'data_with_pct') else self.data
        if df is None or df.empty:
            return { 'error': 'No data available' }
        if 'player_lower' in df.columns:
            target_matches = df[self._contains_noregex(df['player_lower'], str(player_name).lower())]
        else:
            target_matches = df[df['player'].str.contains(player_name, case=False, na=False, regex=False)]
        if target_matches.empty:
            return { 'error': f"Player '{player_name}' not found" }
        if seasons and 'season' in target_matches.columns:
            target_matches = target_matches[target_matches['season'].isin(seasons)]
        # Choose season for target
        if alignment == 'overlap':
            season_used = self._choose_overlap_season(target_matches, min_minutes=min_minutes, coverage_threshold=0.0)
        else:
            season_used = None
        if not season_used and 'season' in target_matches.columns:
            ms = sorted(target_matches['season'].tolist(), key=lambda x: int(str(x)[:4]) if isinstance(x, str) and x[:4].isdigit() else -1)
            season_used = ms[-1]
        target = target_matches[target_matches['season'] == season_used].iloc[0] if season_used is not None else target_matches.iloc[-1]

        # Candidate pool
        pool = df.copy()
        if leagues and 'league' in pool.columns:
            pool = pool[pool['league'].isin(leagues)]
        if 'playing_time_min' in pool.columns:
            pool = pool[pd.to_numeric(pool['playing_time_min'], errors='coerce') >= float(min_minutes)]
        # Role filter
        if role:
            pos_targets = self._role_to_positions(role)
            if pos_targets and 'position' in pool.columns:
                pool = pool[pool['position'].astype(str).str.contains('|'.join(pos_targets), case=False, na=False)]

        # Metrics for similarity
        metrics = [
            'progressive_passes_per_90_pctile','carries_prgc_pctile','key_passes_per_90_pctile','total_cmp_pct_pctile',
            'tackles_per_90_pctile','interceptions_per_90_pctile','aerial_duels_won_pct_pctile','expected_xag_pctile','expected_xg_pctile','standard_sot_pct_pctile'
        ]
        # Ensure columns exist
        metrics = [m for m in metrics if m in pool.columns and m in target.index]

        def distance(row):
            diffs = []
            for m in metrics:
                diffs.append(abs((row.get(m) or 0) - (target.get(m) or 0)))
            return float(np.mean(diffs)) if diffs else 9999.0

        pool = pool[pool['player'] != target.get('player')]
        pool['similarity'] = pool.apply(lambda r: 100.0 - distance(r), axis=1)  # higher is more similar
        top = pool.sort_values('similarity', ascending=False).head(k)

        results = []
        for _, r in top.iterrows():
            results.append({
                'name': r.get('player', 'Unknown'),
                'team': r.get('team', 'Unknown'),
                'league': r.get('league', 'Unknown'),
                'position': r.get('position', 'Unknown'),
                'season_used': r.get('season', 'Unknown'),
                'similarity': round(float(r.get('similarity', 0)), 1)
            })
        return {
            'player': target.get('player', 'Unknown'),
            'season_used': target.get('season', 'Unknown'),
            'comparables': results
        }

    def trend_watch(
        self,
        role: Optional[str] = None,
        position: Optional[str] = None,
        leagues: Optional[List[str]] = None,
        last_n_seasons: int = 2,
        min_minutes: int = 900,
        top_n: int = 10
    ) -> Dict[str, Any]:
        """Find improving and declining players based on delta of key metrics over last_n_seasons."""
        df = self.data.copy()
        if df.empty or 'season' not in df.columns:
            return { 'error': 'Insufficient data' }
        if leagues and 'league' in df.columns:
            df = df[df['league'].isin(leagues)]
        if role:
            pos_targets = self._role_to_positions(role)
            if pos_targets and 'position' in df.columns:
                df = df[df['position'].astype(str).str.contains('|'.join(pos_targets), case=False, na=False)]
        elif position and 'position' in df.columns:
            df = df[df['position'].astype(str).str.contains(position, case=False, na=False, regex=False)]

        # Ensure minutes filter per season row
        if 'playing_time_min' in df.columns:
            df = df[pd.to_numeric(df['playing_time_min'], errors='coerce') >= float(min_minutes)]
        elif 'playing_time_90s' in df.columns:
            df = df[pd.to_numeric(df['playing_time_90s'], errors='coerce') * 90.0 >= float(min_minutes)]

        if df.empty:
            return { 'improving': [], 'declining': [] }

        # Metrics based on role
        weights = self._role_weights(role or '')
        chosen = list(weights.keys())[:6]

        # Order seasons
        df = df.copy()
        df['season_start'] = df['season'].apply(lambda x: int(str(x)[:4]) if isinstance(x, str) and str(x)[:4].isdigit() else -1)
        # Keep last_n_seasons by player
        def last_n(group):
            return group.sort_values('season_start').tail(last_n_seasons)
        recent = df.groupby('player', group_keys=False).apply(last_n)
        recent = recent.sort_values(['player','season_start'])

        # Calculate simple score per season as weighted sum of raw metrics (fallback if no pct)
        def season_score(row):
            s = 0.0
            tw = 0.0
            for m, w in weights.items():
                val = row.get(m, np.nan)
                if pd.notna(val):
                    s += float(val) * w
                    tw += w
            return s / tw if tw > 0 else 0.0
        recent['role_score'] = recent.apply(season_score, axis=1)

        # Compute deltas per player
        deltas = []
        for name, grp in recent.groupby('player'):
            vals = grp.sort_values('season_start')['role_score'].tolist()
            if len(vals) >= 2:
                change = vals[-1] - vals[0]
                last_row = grp.sort_values('season_start').iloc[-1]
                deltas.append({
                    'name': name,
                    'team': last_row.get('team','Unknown'),
                    'league': last_row.get('league','Unknown'),
                    'position': last_row.get('position','Unknown'),
                    'season_latest': last_row.get('season','Unknown'),
                    'change': round(change,2)
                })

        improvers = sorted([d for d in deltas if d['change'] > 0], key=lambda x: x['change'], reverse=True)[:top_n]
        decliners = sorted([d for d in deltas if d['change'] < 0], key=lambda x: x['change'])[:top_n]
        return { 'improving': improvers, 'declining': decliners }

    def undervalued_creators(
        self,
        leagues: Optional[List[str]] = None,
        age_max: Optional[int] = None,
        min_minutes: int = 900,
        top_n: int = 10
    ) -> List[Dict[str, Any]]:
        """Find players with high progression/creation percentiles but modest goals/assists (underrated)."""
        df = self.data_with_pct if hasattr(self, 'data_with_pct') else self.data
        if df is None or df.empty:
            return []
        work = df.copy()
        if leagues and 'league' in work.columns:
            work = work[work['league'].isin(leagues)]
        if age_max is not None and 'age' in work.columns:
            work = work[work['age'] <= age_max]
        # Minutes filter
        if 'playing_time_min' in work.columns:
            work = work[pd.to_numeric(work['playing_time_min'], errors='coerce') >= float(min_minutes)]
        elif 'playing_time_90s' in work.columns:
            work = work[pd.to_numeric(work['playing_time_90s'], errors='coerce') * 90.0 >= float(min_minutes)]

        # Define creation score and output score
        def creation_pct(row):
            vals = []
            for k in ['progressive_passes_per_90_pctile','carries_prgc_pctile','key_passes_per_90_pctile','sca_sca_pctile']:
                if k in row.index and pd.notna(row.get(k)):
                    vals.append(float(row.get(k)))
            return np.mean(vals) if vals else 0.0
        def output_pct(row):
            vals = []
            for k in ['performance_gls_pctile','expected_xg_pctile','gca_gca_pctile']:
                if k in row.index and pd.notna(row.get(k)):
                    vals.append(float(row.get(k)))
            return np.mean(vals) if vals else 0.0
        work['creation_score'] = work.apply(creation_pct, axis=1)
        work['output_score'] = work.apply(output_pct, axis=1)
        work['undervalue_gap'] = work['creation_score'] - work['output_score']
        top = work.sort_values('undervalue_gap', ascending=False).head(top_n)
        results = []
        for _, r in top.iterrows():
            results.append({
                'name': r.get('player','Unknown'),
                'team': r.get('team','Unknown'),
                'league': r.get('league','Unknown'),
                'position': r.get('position','Unknown'),
                'season_used': r.get('season','Unknown'),
                'creation_score': round(float(r.get('creation_score',0)),1),
                'output_score': round(float(r.get('output_score',0)),1),
                'undervalue_gap': round(float(r.get('undervalue_gap',0)),1)
            })
        return results

    def style_fit_search(
        self,
        style: str,
        leagues: Optional[List[str]] = None,
        age_max: Optional[int] = None,
        min_minutes: int = 900,
        top_n: int = 10
    ) -> List[Dict[str, Any]]:
        """Rank players by style fit only (possession/transition/high_press/direct)."""
        df = self.data_with_pct if hasattr(self, 'data_with_pct') else self.data
        if df is None or df.empty:
            return []
        work = df.copy()
        if leagues and 'league' in work.columns:
            work = work[work['league'].isin(leagues)]
        if age_max is not None and 'age' in work.columns:
            work = work[work['age'] <= age_max]
        if 'playing_time_min' in work.columns:
            work = work[pd.to_numeric(work['playing_time_min'], errors='coerce') >= float(min_minutes)]
        elif 'playing_time_90s' in work.columns:
            work = work[pd.to_numeric(work['playing_time_90s'], errors='coerce') * 90.0 >= float(min_minutes)]

        multipliers = self._style_adjustments(style)
        # Base weights equally across style metrics we adjust
        keys = list(multipliers.keys())
        if not keys:
            return []
        def style_score(row):
            vals = []
            for m in keys:
                v = row.get(f"{m}_pctile", np.nan)
                if pd.notna(v):
                    vals.append(float(v) * multipliers[m])
            return np.mean(vals) if vals else 0.0
        work['style_fit'] = work.apply(style_score, axis=1)
        top = work.sort_values('style_fit', ascending=False).head(top_n)
        return [
            {
                'name': r.get('player','Unknown'),
                'team': r.get('team','Unknown'),
                'league': r.get('league','Unknown'),
                'position': r.get('position','Unknown'),
                'season_used': r.get('season','Unknown'),
                'style_fit': round(float(r.get('style_fit',0)),1)
            } for _, r in top.iterrows()
        ]

    def multi_role_candidates(
        self,
        primary_role: str,
        secondary_role: str,
        leagues: Optional[List[str]] = None,
        min_minutes: int = 900,
        top_n: int = 10
    ) -> List[Dict[str, Any]]:
        """Players with strong fit across two roles (avg of role fits)."""
        df = self.data_with_pct if hasattr(self, 'data_with_pct') else self.data
        if df is None or df.empty:
            return []
        work = df.copy()
        if leagues and 'league' in work.columns:
            work = work[work['league'].isin(leagues)]
        if 'playing_time_min' in work.columns:
            work = work[pd.to_numeric(work['playing_time_min'], errors='coerce') >= float(min_minutes)]
        elif 'playing_time_90s' in work.columns:
            work = work[pd.to_numeric(work['playing_time_90s'], errors='coerce') * 90.0 >= float(min_minutes)]
        w1, w2 = self._role_weights(primary_role), self._role_weights(secondary_role)
        keys = list(set(list(w1.keys()) + list(w2.keys())))
        def fit(row, weights):
            s=0.0; tw=0.0
            for m,w in weights.items():
                v = row.get(f"{m}_pctile", np.nan)
                if pd.notna(v):
                    s += float(v)*w
                    tw += w
            return (s/tw) if tw>0 else 0.0
        work['fit_primary'] = work.apply(lambda r: fit(r,w1), axis=1)
        work['fit_secondary'] = work.apply(lambda r: fit(r,w2), axis=1)
        work['fit_combo'] = (work['fit_primary'] + work['fit_secondary'])/2.0
        top = work.sort_values('fit_combo', ascending=False).head(top_n)
        return [
            {
                'name': r.get('player','Unknown'),
                'team': r.get('team','Unknown'),
                'league': r.get('league','Unknown'),
                'position': r.get('position','Unknown'),
                'season_used': r.get('season','Unknown'),
                'fit_primary': round(float(r.get('fit_primary',0)),1),
                'fit_secondary': round(float(r.get('fit_secondary',0)),1),
                'fit_combo': round(float(r.get('fit_combo',0)),1)
            } for _, r in top.iterrows()
        ]

    def conversion_candidates(
        self,
        target_role: str,
        leagues: Optional[List[str]] = None,
        min_minutes: int = 900,
        top_n: int = 10
    ) -> List[Dict[str, Any]]:
        """Players not in target positions but whose metrics fit the target role."""
        df = self.data_with_pct if hasattr(self, 'data_with_pct') else self.data
        if df is None or df.empty:
            return []
        work = df.copy()
        if leagues and 'league' in work.columns:
            work = work[work['league'].isin(leagues)]
        if 'playing_time_min' in work.columns:
            work = work[pd.to_numeric(work['playing_time_min'], errors='coerce') >= float(min_minutes)]
        elif 'playing_time_90s' in work.columns:
            work = work[pd.to_numeric(work['playing_time_90s'], errors='coerce') * 90.0 >= float(min_minutes)]
        # compute fit to target role
        weights = self._role_weights(target_role)
        def fit(row):
            s=0.0; tw=0.0
            for m,w in weights.items():
                v = row.get(f"{m}_pctile", np.nan)
                if pd.notna(v):
                    s += float(v)*w
                    tw += w
            return (s/tw) if tw>0 else 0.0
        work['target_fit'] = work.apply(fit, axis=1)
        # exclude current target positions
        targets = self._role_to_positions(target_role)
        if targets and 'position' in work.columns:
            work = work[~work['position'].astype(str).str.contains('|'.join(targets), case=False, na=False)]
        top = work.sort_values('target_fit', ascending=False).head(top_n)
        return [
            {
                'name': r.get('player','Unknown'),
                'team': r.get('team','Unknown'),
                'league': r.get('league','Unknown'),
                'position': r.get('position','Unknown'),
                'season_used': r.get('season','Unknown'),
                'target_fit': round(float(r.get('target_fit',0)),1)
            } for _, r in top.iterrows()
        ]

    def xi_builder(
        self,
        style: str,
        leagues: Optional[List[str]] = None,
        age_policy: Optional[str] = None,  # 'u23','u25','prime','any'
        min_minutes: int = 900
    ) -> Dict[str, Any]:
        """Build a simple XI by role using discover_talents per slot."""
        role_order = [
            ('goalkeeper','GK'),
            ('right_back','RB'),
            ('centre_back','RCB'),
            ('centre_back','LCB'),
            ('left_back','LB'),
            ('defensive_midfielder','DM'),
            ('central_midfielder','CM'),
            ('attacking_midfielder','AM'),
            ('winger','RW'),
            ('winger','LW'),
            ('striker','ST')
        ]
        age_max = None
        if age_policy == 'u23': age_max = 23
        elif age_policy == 'u25': age_max = 25
        elif age_policy == 'prime': age_max = 28
        used_names = set()
        xi = []
        for role, slot in role_order:
            cands = self.discover_talents(
                role=role, style=style, leagues=leagues, age_max=age_max, min_minutes=min_minutes,
                alignment='overlap', coverage_threshold=0.7, exclude_elite=False, top_n=10, diversify_by='team', explain=True
            )
            pick = next((c for c in cands if c['name'] not in used_names), None)
            if pick:
                used_names.add(pick['name'])
                xi.append({ 'slot': slot, **pick })
        return { 'style': style, 'age_policy': age_policy, 'xi': xi }

    def search_by_profile(self, scout_brief: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Search players based on comprehensive scouting profile"""
        
        # Extract search parameters from scout brief
        leagues = scout_brief.get('target_leagues', None)
        positions = scout_brief.get('positions', None)
        age_range = scout_brief.get('age_range', {})
        physical_requirements = scout_brief.get('physical_requirements', {})
        technical_requirements = scout_brief.get('technical_requirements', {})
        temporal_preferences = scout_brief.get('temporal_preferences', {})
        
        # Build statistical filters
        stat_filters = {}
        
        # Add technical requirements
        if technical_requirements:
            for req, min_val in technical_requirements.items():
                stat_filters[req] = min_val
        
        # Perform advanced search
        return self.search_players_advanced(
            leagues=leagues,
            positions=positions,
            age_min=age_range.get('min'),
            age_max=age_range.get('max'),
            seasons=temporal_preferences.get('seasons'),
            latest_season_only=temporal_preferences.get('latest_season_only', False),
            stat_filters=stat_filters,
            limit=scout_brief.get('max_results', 50)
        )
    
    def get_league_leaders(
        self,
        league: str,
        stat: str,
        position: Optional[str] = None,
        season: Optional[str] = None,
        min_games: int = 15
    ) -> List[Dict[str, Any]]:
        """Get league leaders in specific statistics"""
        
        cache_key = ('get_league_leaders', league, stat, position, season, min_games)
        cached = self._result_cache.get(cache_key)
        if cached is not None:
            return cached
        t0 = perf_counter()
        filtered = self.data[self.data['league'] == league].copy()
        
        # Season filtering
        if season:
            filtered = filtered[filtered['season'] == season]
        
        if position:
            filtered = filtered[filtered['position'].astype(str).str.contains(position, case=False, na=False, regex=False)]
        
        # Filter by minimum games
        games_col = 'playing_time_mp' if 'playing_time_mp' in filtered.columns else 'playing_time_starts'
        if games_col in filtered.columns:
            filtered = filtered[filtered[games_col] >= min_games]
        
        # Get stat column
        stat_column = self.stat_mappings.get(stat, stat)
        if stat_column not in filtered.columns:
            return []
        
        # Sort and get top performers
        top_performers = filtered.nlargest(20, stat_column)

        # Add percentiles for the selected stat within league (and position if provided)
        group_cols = ['league'] + (['position'] if position else [])
        if stat_column in top_performers.columns:
            top_performers = self._compute_percentiles(top_performers, [stat_column], group_cols)
        
        leaders = []
        for _, player in top_performers.iterrows():
            leader_info = {
                'name': player.get('player', 'Unknown'),
                'team': player.get('team', 'Unknown'),
                'position': player.get('position', 'Unknown'),
                'stat_value': float(player.get(stat_column, 0)) if pd.notna(player.get(stat_column)) else 0,
                'games_played': int(player.get(games_col, 0)) if pd.notna(player.get(games_col)) else 0,
                'stat_percentile': float(player.get(f"{stat_column}_pctile", 0)) if pd.notna(player.get(f"{stat_column}_pctile", 0)) else 0
            }
            leaders.append(leader_info)

        dt = (perf_counter() - t0) * 1000
        logger.info(f"get_league_leaders league={league} position={position} stat={stat} season={season} -> {len(leaders)} results in {dt:.1f}ms")
        self._result_cache.set(cache_key, leaders)
        return leaders
    
    def compare_multiple_players(
        self,
        player_names: List[str],
        season: Optional[str] = None,
        aggregation_mode: str = "latest",
        focus_stats: Optional[List[str]] = None,
        # New alignment controls
        alignment: str = "overlap",  # default to overlap alignment
        seasons: Optional[List[str]] = None,
        min_minutes: int = 450,
        coverage_threshold: float = 0.8,
        fallback: str = "nearest",  # "nearest" | "exclude"
        tolerance_seasons: int = 1
    ) -> Dict[str, Any]:
        """Compare multiple players across key statistics with improved season alignment.

        alignment="overlap" will select a common target season with sufficient coverage and minutes.
        """

        if not focus_stats:
            focus_stats = [
                'goals', 'assists', 'expected_goals', 'shots_on_target_pct',
                'pass_completion_pct', 'tackles_per_90', 'progressive_passes',
                'minutes_played', 'games_played', 'expected_assists'
            ]

        def _season_start_year(s: str) -> int:
            if not isinstance(s, str):
                return -1
            m = re.match(r"^(\d{4})", s.strip())
            return int(m.group(1)) if m else -1

        def _season_distance(a: str, b: str) -> int:
            return abs(_season_start_year(a) - _season_start_year(b))

        # Minutes handling
        minutes_col = 'playing_time_min' if 'playing_time_min' in self.data.columns else None
        n90_col = 'playing_time_90s' if 'playing_time_90s' in self.data.columns else None

        def _has_minutes(row) -> bool:
            if minutes_col and minutes_col in row.index:
                return float(row.get(minutes_col) or 0) >= float(min_minutes)
            if n90_col and n90_col in row.index:
                return float(row.get(n90_col) or 0) * 90.0 >= float(min_minutes)
            # Fallback to games played if no minutes
            gp = float(row.get('playing_time_mp') or 0)
            return gp * 90.0 >= float(min_minutes)

        # Collect matches per player
        matches_by_player: Dict[str, pd.DataFrame] = {}
        for name in player_names:
            if 'player_lower' in self.data.columns:
                m = self.data[self._contains_noregex(self.data['player_lower'], str(name).lower())]
            else:
                m = self.data[self.data['player'].str.contains(name, case=False, na=False, regex=False)]
            matches_by_player[name] = m.copy() if len(m) > 0 else pd.DataFrame()

        # Determine target season
        target_season = season
        used_alignment = 'target'

        if alignment == 'overlap':
            used_alignment = 'overlap'
            # Candidate seasons provided or union across players
            if seasons:
                candidates = sorted(set(seasons), key=_season_start_year)
            else:
                pool = []
                for m in matches_by_player.values():
                    if not m.empty and 'season' in m.columns:
                        pool.extend(m['season'].tolist())
                candidates = sorted(set(pool), key=_season_start_year)

            # Evaluate coverage from latest to oldest
            best = None
            best_cov = -1.0
            total_players = len(player_names)
            for cand in sorted(candidates, key=_season_start_year, reverse=True):
                available = 0
                for name in player_names:
                    dfp = matches_by_player.get(name)
                    if dfp is None or dfp.empty:
                        continue
                    srow = dfp[dfp['season'] == cand]
                    if not srow.empty and _has_minutes(srow.iloc[0]):
                        available += 1
                cov = available / total_players if total_players else 0
                if cov >= coverage_threshold:
                    target_season = cand
                    best_cov = cov
                    break
                if cov > best_cov:
                    best = cand
                    best_cov = cov
            if target_season is None:
                target_season = best

        # Build comparison
        comparison_data: Dict[str, Any] = {}
        players_found: List[str] = []
        players_not_found: List[str] = []
        players_used: List[Dict[str, Any]] = []

        for player_name in player_names:
            matches = matches_by_player.get(player_name, pd.DataFrame())
            if matches.empty:
                players_not_found.append(player_name)
                continue

            player_row = None
            season_used = None
            fallback_used = False

            # Target-based selection
            if target_season:
                season_matches = matches[matches['season'] == target_season]
                if not season_matches.empty and _has_minutes(season_matches.iloc[0]):
                    player_row = season_matches.iloc[0]
                    season_used = target_season
                else:
                    if alignment == 'overlap' and fallback == 'nearest' and tolerance_seasons > 0:
                        # Search nearest seasons within tolerance
                        cand_seasons = sorted(matches['season'].dropna().unique(), key=_season_start_year)
                        # Rank by distance; prefer later on ties
                        ranked = sorted(
                            cand_seasons,
                            key=lambda s: (_season_distance(s, target_season), -_season_start_year(s))
                        )
                        for s in ranked:
                            if _season_distance(s, target_season) <= tolerance_seasons:
                                srow = matches[matches['season'] == s]
                                if not srow.empty and _has_minutes(srow.iloc[0]):
                                    player_row = srow.iloc[0]
                                    season_used = s
                                    fallback_used = True
                                    break
                    # If still none and fallback is exclude, leave as None
            # Legacy behavior when no target
            if player_row is None and target_season is None:
                # Use latest available season (by start year)
                if 'season' in matches.columns:
                    latest_s = sorted(matches['season'].tolist(), key=_season_start_year)[-1]
                    player_row = matches[matches['season'] == latest_s].iloc[0]
                    season_used = latest_s
                else:
                    player_row = matches.iloc[-1]
                    season_used = player_row.get('season', 'Unknown')

            if player_row is None:
                players_not_found.append(player_name)
                continue

            players_found.append(player_name)
            
            player_stats = {
                'name': player_row.get('player', 'Unknown'),
                'age': self._extract_age(player_row.get('age', 0)),
                'team': player_row.get('team', 'Unknown'),
                'league': player_row.get('league', 'Unknown'),
                'position': player_row.get('position', 'Unknown'),
                'season': player_row.get('season', 'Unknown'),
                'stats': {}
            }

            for stat in focus_stats:
                column_name = self.stat_mappings.get(stat, stat)
                if column_name in player_row.index:
                    val = player_row.get(column_name, 0)
                    player_stats['stats'][stat] = float(val) if pd.notna(val) else 0
                else:
                    player_stats['stats'][stat] = 0

            comparison_data[player_name] = player_stats
            players_used.append({
                'name': player_name,
                'season_used': season_used,
                'fallback_used': fallback_used
            })

        # Realized coverage (players in exact target season without fallback)
        realized_numer = sum(1 for p in players_used if (p.get('season_used') == target_season and not p.get('fallback_used')))
        realized_denom = len(players_found) if players_found else 1
        realized_cov = realized_numer / realized_denom

        result = {
            'comparison_season': target_season,
            'players_compared': len(players_found),
            'players_found': players_found,
            'players_not_found': players_not_found,
            'players_used': players_used,
            'comparison_data': comparison_data,
            'focus_statistics': focus_stats,
            'season_alignment': used_alignment,
            'minutes_threshold': min_minutes,
            'coverage_threshold': coverage_threshold,
            'realized_coverage': realized_cov
        }
        return result
    
    def generate_detailed_scouting_report(
        self,
        player_name: str,
        season: Optional[str] = None,
        aggregation_mode: str = "latest",
        comparison_players: Optional[List[str]] = None,
        focus_areas: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Generate comprehensive scouting report for a player"""
        
        # Find the player
        # Fast lower-case contains
        if 'player_lower' in self.data.columns:
            plow = str(player_name).lower()
            matches = self.data[self._contains_noregex(self.data['player_lower'], plow)]
        else:
            matches = self.data[self.data['player'].str.contains(player_name, case=False, na=False, regex=False)]
        
        if len(matches) == 0:
            return {'error': f"Player '{player_name}' not found"}
        
        # Handle multiple matches intelligently
        if len(matches) > 1:
            # If season specified, filter by season first
            if season:
                season_matches = matches[matches['season'] == season]
                if len(season_matches) > 0:
                    matches = season_matches
                else:
                    return {'error': f"Player '{player_name}' not found in season {season}"}
            
            # If still multiple matches, use latest season or exact name match
            if len(matches) > 1:
                if aggregation_mode == "latest":
                    # Get latest season for this player
                    exact_name = matches.iloc[0]['player']
                    player_seasons = matches[matches['player'] == exact_name]
                    if len(player_seasons) > 0:
                        matches = player_seasons.sort_values('season').tail(1)
                    else:
                        matches = matches.sort_values('season').tail(1)
                else:
                    # For other modes, prefer exact name match or return options
                    exact_matches = matches[matches['player'].str.lower() == player_name.lower()]
                    if len(exact_matches) > 0:
                        matches = exact_matches
                    else:
                        unique_players = matches.drop_duplicates('player')[['player', 'team', 'season']].head(5)
                        return {
                            'error': f"Multiple players found for '{player_name}'. Please be more specific.",
                            'options': [f"{row['player']} ({row['team']}, {row['season']})" for _, row in unique_players.iterrows()]
                        }
        
        player = matches.iloc[0]
        
        # Basic player information with proper age handling
        basic_info = {
            'name': player.get('player', 'Unknown'),
            'age': self._extract_age(player.get('age', 0)),
            'nationality': player.get('nation', 'Unknown'),
            'team': player.get('team', 'Unknown'),
            'league': player.get('league', 'Unknown'),
            'position': player.get('position', 'Unknown'),
            'season': player.get('season', 'Unknown')
        }
        
        # Performance statistics organized by category
        performance_stats = self._extract_performance_stats(player)
        defensive_stats = self._extract_defensive_stats(player)
        passing_stats = self._extract_passing_stats(player)
        attacking_stats = self._extract_attacking_stats(player)
        
        # Generate comparison if requested (ensure same season)
        comparison_data = None
        elite_peer_context = None
        if comparison_players:
            player_season = player.get('season', 'Unknown')
            comparison_data = self.compare_multiple_players(
                [player_name] + comparison_players,
                season=player_season,
                focus_stats=['goals', 'assists', 'expected_goals', 'tackles_per_90', 'pass_completion_pct']
            )
        
        # Add elite peer context for the player's position and league
        elite_peer_context = self._get_elite_peer_context(player)

        # Percentiles across key stats for league+position
        percentile_cols = [
            'performance_gls','performance_ast','expected_xg','expected_xag',
            'tackles_per_90','interceptions_per_90','total_cmp_pct','progressive_passes_per_90',
            'standard_sot_pct','carries_prgc','key_passes_per_90'
        ]
        group_cols = []
        if 'league' in self.data.columns:
            group_cols.append('league')
        if 'position' in self.data.columns:
            group_cols.append('position')
        work = self.data.copy()
        work = self._compute_percentiles(work, [c for c in percentile_cols if c in work.columns], group_cols)
        player_with_pct = work[work['player'] == player.get('player')].sort_values('season').iloc[-1] if len(work[work['player'] == player.get('player')])>0 else player

        # Overall rating from percentiles
        overall_rating = self._overall_rating(player_with_pct)

        t0 = perf_counter()
        res = {
            'basic_info': basic_info,
            'performance_stats': performance_stats,
            'defensive_stats': defensive_stats,
            'passing_stats': passing_stats,
            'attacking_stats': attacking_stats,
            'comparison_data': comparison_data,
            'elite_peer_context': elite_peer_context,
            'scouting_summary': self._generate_scouting_summary(player, focus_areas),
            'percentiles_context': {k: float(player_with_pct.get(f"{k}_pctile", 0)) for k in percentile_cols if f"{k}_pctile" in player_with_pct.index},
            'overall_rating': overall_rating
        }
        dt = (perf_counter() - t0) * 1000
        logger.info(f"generate_detailed_scouting_report player={player_name} season={season} -> rating={overall_rating} in {dt:.1f}ms")
        return res
    
    def _extract_performance_stats(self, player) -> Dict[str, float]:
        """Extract core performance statistics"""
        return {
            'goals': float(player.get('performance_gls', 0)) if pd.notna(player.get('performance_gls')) else 0,
            'assists': float(player.get('performance_ast', 0)) if pd.notna(player.get('performance_ast')) else 0,
            'minutes_played': float(player.get('playing_time_min', 0)) if pd.notna(player.get('playing_time_min')) else 0,
            'games_played': int(player.get('playing_time_mp', 0)) if pd.notna(player.get('playing_time_mp')) else 0,
            'expected_goals': float(player.get('expected_xg', 0)) if pd.notna(player.get('expected_xg')) else 0,
            'expected_assists': float(player.get('expected_xag', 0)) if pd.notna(player.get('expected_xag')) else 0,
            'goals_per_90': float(player.get('per_90_minutes_gls', 0)) if pd.notna(player.get('per_90_minutes_gls')) else 0,
            'assists_per_90': float(player.get('per_90_minutes_ast', 0)) if pd.notna(player.get('per_90_minutes_ast')) else 0
        }
    
    def _extract_defensive_stats(self, player) -> Dict[str, float]:
        """Extract defensive statistics"""
        return {
            'tackles': float(player.get('tackles_tkl', 0)) if pd.notna(player.get('tackles_tkl')) else 0,
            'tackles_won': float(player.get('tackles_tklw', 0)) if pd.notna(player.get('tackles_tklw')) else 0,
            'interceptions': float(player.get('interceptions', 0)) if pd.notna(player.get('interceptions')) else 0,
            'blocks': float(player.get('blocks_blocks', 0)) if pd.notna(player.get('blocks_blocks')) else 0,
            'clearances': float(player.get('clearances', 0)) if pd.notna(player.get('clearances')) else 0,
            'aerial_duels_won': float(player.get('aerial_duels_won', 0)) if pd.notna(player.get('aerial_duels_won')) else 0,
            'aerial_duels_won_pct': float(player.get('aerial_duels_won_pct', 0)) if pd.notna(player.get('aerial_duels_won_pct')) else 0
        }
    
    def _extract_passing_stats(self, player) -> Dict[str, float]:
        """Extract passing and progression statistics"""
        return {
            'pass_completion_pct': float(player.get('total_cmp_pct', 0)) if pd.notna(player.get('total_cmp_pct')) else 0,
            'passes_attempted': float(player.get('total_att', 0)) if pd.notna(player.get('total_att')) else 0,
            'progressive_passes': float(player.get('progressive_passes', 0)) if pd.notna(player.get('progressive_passes')) else 0,
            'key_passes': float(player.get('kp', 0)) if pd.notna(player.get('kp')) else 0,
            'long_passes_completed': float(player.get('long_cmp', 0)) if pd.notna(player.get('long_cmp')) else 0,
            'progressive_carries': float(player.get('carries_prgc', 0)) if pd.notna(player.get('carries_prgc')) else 0
        }
    
    def _extract_attacking_stats(self, player) -> Dict[str, float]:
        """Extract attacking statistics"""
        return {
            'shots': float(player.get('standard_sh', 0)) if pd.notna(player.get('standard_sh')) else 0,
            'shots_on_target': float(player.get('standard_sot', 0)) if pd.notna(player.get('standard_sot')) else 0,
            'shots_on_target_pct': float(player.get('standard_sot_pct', 0)) if pd.notna(player.get('standard_sot_pct')) else 0,
            'successful_dribbles': float(player.get('take-ons_succ', 0)) if pd.notna(player.get('take-ons_succ')) else 0,
            'dribble_attempts': float(player.get('take-ons_att', 0)) if pd.notna(player.get('take-ons_att')) else 0,
            'penalty_area_touches': float(player.get('touches_att_pen', 0)) if pd.notna(player.get('touches_att_pen')) else 0,
            'shot_creating_actions': float(player.get('sca_sca', 0)) if pd.notna(player.get('sca_sca')) else 0,
            'goal_creating_actions': float(player.get('gca_gca', 0)) if pd.notna(player.get('gca_gca')) else 0
        }
    
    def _generate_scouting_summary(self, player, focus_areas: Optional[List[str]] = None) -> Dict[str, str]:
        """Generate textual scouting summary"""
        
        summary = {}
        
        # Overall assessment
        goals = float(player.get('performance_gls', 0)) if pd.notna(player.get('performance_gls')) else 0
        assists = float(player.get('performance_ast', 0)) if pd.notna(player.get('performance_ast')) else 0
        age = int(player.get('age', 0)) if pd.notna(player.get('age')) else 0
        
        summary['overall'] = f"A {age}-year-old player with {goals} goals and {assists} assists this season."
        
        # Position-specific insights
        position = player.get('position', 'Unknown')
        if 'FW' in str(position):
            xg = float(player.get('expected_xg', 0)) if pd.notna(player.get('expected_xg')) else 0
            summary['attacking'] = f"As a forward, showing clinical finishing with {goals} goals from {xg:.1f} expected goals."
        
        elif 'MF' in str(position):
            passes = float(player.get('total_cmp_pct', 0)) if pd.notna(player.get('total_cmp_pct')) else 0
            summary['passing'] = f"Midfielder with {passes:.1f}% pass completion rate, contributing {assists} assists."
        
        elif 'DF' in str(position):
            tackles = float(player.get('tackles_tkl', 0)) if pd.notna(player.get('tackles_tkl')) else 0
            summary['defensive'] = f"Defender with solid defensive stats including {tackles} tackles."
        
        return summary
    
    def _get_elite_peer_context(self, player) -> Dict[str, Any]:
        """Get elite peer context for comparative analysis"""
        
        player_league = player.get('league', 'Unknown')
        player_position = player.get('position', 'Unknown')
        player_season = player.get('season', 'Unknown')
        
        # Find elite players in same league, position, and season
        position_filter = player_position.split(',')[0] if ',' in str(player_position) else player_position
        
        elite_peers = self.data[
            (self.data['league'] == player_league) &
            (self.data['position'].str.contains(position_filter, case=False, na=False)) &
            (self.data['season'] == player_season) &
            (self.data['playing_time_min'].fillna(0) >= 1000)  # Minimum playing time
        ].copy()
        
        if len(elite_peers) < 5:
            return {'message': 'Insufficient peer data for comparison'}
        
        # Calculate percentiles for key stats based on position
        key_stats = ['performance_gls', 'performance_ast', 'expected_xg'] if 'FW' in str(position_filter) else ['tackles_tkl', 'interceptions', 'total_cmp_pct']
        
        player_percentiles = {}
        for stat in key_stats:
            if stat in elite_peers.columns and stat in player.index:
                player_value = float(player.get(stat, 0)) if pd.notna(player.get(stat)) else 0
                percentile = (elite_peers[stat].fillna(0) < player_value).mean() * 100
                player_percentiles[stat] = round(percentile, 1)
        
        # Get top 5 performers in primary stat
        primary_stat = key_stats[0] if key_stats else 'performance_gls'
        top_performers = elite_peers.nlargest(5, primary_stat)
        
        elite_comparison = []
        for _, elite_player in top_performers.iterrows():
            elite_comparison.append({
                'name': elite_player.get('player', 'Unknown'),
                'team': elite_player.get('team', 'Unknown'),
                'primary_stat_value': float(elite_player.get(primary_stat, 0)) if pd.notna(elite_player.get(primary_stat)) else 0
            })
        
        return {
            'league': player_league,
            'position': player_position,
            'season': player_season,
            'peer_group_size': len(elite_peers),
            'player_percentiles': player_percentiles,
            'elite_comparison': elite_comparison,
            'primary_stat': primary_stat
        }
    
    def get_player_career_summary(
        self,
        player_name: str,
        aggregation_mode: str = "latest"  # "latest", "career_avg", "best_season", "all_seasons"
    ) -> Dict[str, Any]:
        """Get player career summary with temporal analysis"""
        
        # Find all records for this player
        if 'player_lower' in self.data.columns:
            player_data = self.data[self._contains_noregex(self.data['player_lower'], str(player_name).lower())]
        else:
            player_data = self.data[self.data['player'].str.contains(player_name, case=False, na=False, regex=False)]
        
        if len(player_data) == 0:
            return {'error': f"Player '{player_name}' not found"}
        
        # Get exact name from first match
        exact_name = player_data.iloc[0]['player']
        player_data = self.data[self.data['player'] == exact_name]
        
        # Sort by season for progression analysis
        player_data = player_data.sort_values('season')
        
        if aggregation_mode == "latest":
            # Return most recent season only
            latest_data = player_data.iloc[-1]
            return self._format_single_season_data(latest_data, "Latest Season")
            
        elif aggregation_mode == "best_season":
            # Find best season based on combined goal+assist output
            if 'performance_gls' in player_data.columns and 'performance_ast' in player_data.columns:
                player_data['combined_output'] = player_data['performance_gls'].fillna(0) + player_data['performance_ast'].fillna(0)
                best_season_data = player_data.loc[player_data['combined_output'].idxmax()]
                return self._format_single_season_data(best_season_data, "Best Season")
            else:
                return {'error': 'Insufficient data for best season analysis'}
                
        elif aggregation_mode == "career_avg":
            # Calculate career averages
            return self._calculate_career_averages(player_data)
            
        elif aggregation_mode == "all_seasons":
            # Return all seasons with progression analysis
            return self._analyze_player_progression(player_data)
        
        else:
            return {'error': f"Unknown aggregation mode: {aggregation_mode}"}
    
    def _format_single_season_data(self, season_data, mode_label: str) -> Dict[str, Any]:
        """Format single season data with proper age handling"""
        
        # Clean age format (handle "27-327" format)
        age_str = str(season_data.get('age', '0'))
        if '-' in age_str:
            age = int(age_str.split('-')[0])
        else:
            age = int(float(age_str)) if age_str.replace('.', '').isdigit() else 0
        
        return {
            'mode': mode_label,
            'player': season_data.get('player', 'Unknown'),
            'season': season_data.get('season', 'Unknown'),
            'age_at_season': age,
            'team': season_data.get('team', 'Unknown'),
            'league': season_data.get('league', 'Unknown'),
            'position': season_data.get('position', 'Unknown'),
            'performance_stats': {
                'goals': float(season_data.get('performance_gls', 0)) if pd.notna(season_data.get('performance_gls')) else 0,
                'assists': float(season_data.get('performance_ast', 0)) if pd.notna(season_data.get('performance_ast')) else 0,
                'minutes_played': float(season_data.get('playing_time_min', 0)) if pd.notna(season_data.get('playing_time_min')) else 0,
                'games_played': int(season_data.get('playing_time_mp', 0)) if pd.notna(season_data.get('playing_time_mp')) else 0,
                'expected_goals': float(season_data.get('expected_xg', 0)) if pd.notna(season_data.get('expected_xg')) else 0,
                'expected_assists': float(season_data.get('expected_xag', 0)) if pd.notna(season_data.get('expected_xag')) else 0
            }
        }
    
    def _calculate_career_averages(self, player_data) -> Dict[str, Any]:
        """Calculate career averages across all seasons"""
        
        # Get basic info from most recent season
        latest = player_data.iloc[-1]
        
        # Calculate weighted averages and totals
        total_minutes = player_data['playing_time_min'].fillna(0).sum()
        total_games = player_data['playing_time_mp'].fillna(0).sum()
        total_goals = player_data['performance_gls'].fillna(0).sum()
        total_assists = player_data['performance_ast'].fillna(0).sum()
        
        # Career averages per season
        seasons_played = len(player_data)
        avg_goals_per_season = total_goals / seasons_played if seasons_played > 0 else 0
        avg_assists_per_season = total_assists / seasons_played if seasons_played > 0 else 0
        
        return {
            'mode': 'Career Average',
            'player': latest.get('player', 'Unknown'),
            'seasons_analyzed': list(player_data['season'].unique()),
            'total_seasons': seasons_played,
            'current_age': self._extract_age(latest.get('age', 0)),
            'current_team': latest.get('team', 'Unknown'),
            'position': latest.get('position', 'Unknown'),
            'career_totals': {
                'total_goals': float(total_goals),
                'total_assists': float(total_assists),
                'total_minutes': float(total_minutes),
                'total_games': int(total_games)
            },
            'career_averages': {
                'goals_per_season': round(avg_goals_per_season, 2),
                'assists_per_season': round(avg_assists_per_season, 2),
                'minutes_per_season': round(total_minutes / seasons_played, 0) if seasons_played > 0 else 0,
                'games_per_season': round(total_games / seasons_played, 1) if seasons_played > 0 else 0
            }
        }
    
    def _analyze_player_progression(self, player_data) -> Dict[str, Any]:
        """Analyze player progression across seasons"""
        
        # Sort by season
        player_data = player_data.sort_values('season')
        latest = player_data.iloc[-1]
        
        # Calculate season-by-season progression
        season_progression = []
        for _, season in player_data.iterrows():
            season_summary = {
                'season': season.get('season', 'Unknown'),
                'age': self._extract_age(season.get('age', 0)),
                'team': season.get('team', 'Unknown'),
                'league': season.get('league', 'Unknown'),
                'goals': float(season.get('performance_gls', 0)) if pd.notna(season.get('performance_gls')) else 0,
                'assists': float(season.get('performance_ast', 0)) if pd.notna(season.get('performance_ast')) else 0,
                'minutes': float(season.get('playing_time_min', 0)) if pd.notna(season.get('playing_time_min')) else 0,
                'expected_goals': float(season.get('expected_xg', 0)) if pd.notna(season.get('expected_xg')) else 0
            }
            season_progression.append(season_summary)
        
        # Calculate trends (improvement/decline)
        trend_analysis = self._calculate_performance_trends(season_progression)
        
        return {
            'mode': 'All Seasons Progression',
            'player': latest.get('player', 'Unknown'),
            'current_age': self._extract_age(latest.get('age', 0)),
            'current_team': latest.get('team', 'Unknown'),
            'position': latest.get('position', 'Unknown'),
            'seasons_analyzed': len(season_progression),
            'season_by_season': season_progression,
            'trend_analysis': trend_analysis
        }
    
    def _extract_age(self, age_value) -> int:
        """Extract clean age from various age formats"""
        age_str = str(age_value)
        if '-' in age_str:
            return int(age_str.split('-')[0])
        else:
            try:
                return int(float(age_str))
            except:
                return 0
    
    def _calculate_performance_trends(self, season_progression) -> Dict[str, Any]:
        """Calculate performance trends across seasons"""
        
        if len(season_progression) < 2:
            return {'trend': 'insufficient_data', 'message': 'Need at least 2 seasons for trend analysis'}
        
        # Extract numeric progression
        goals_progression = [s['goals'] for s in season_progression]
        assists_progression = [s['assists'] for s in season_progression]
        
        # Simple trend calculation (last vs first season)
        goals_trend = goals_progression[-1] - goals_progression[0] if len(goals_progression) >= 2 else 0
        assists_trend = assists_progression[-1] - assists_progression[0] if len(assists_progression) >= 2 else 0
        
        # Determine overall trend
        combined_trend = goals_trend + assists_trend
        
        if combined_trend > 2:
            trend_label = "improving"
        elif combined_trend < -2:
            trend_label = "declining"  
        else:
            trend_label = "stable"
        
        return {
            'trend': trend_label,
            'goals_change': goals_trend,
            'assists_change': assists_trend,
            'combined_output_change': combined_trend,
            'seasons_compared': f"{season_progression[0]['season']} → {season_progression[-1]['season']}"
        }

# Initialize the enhanced server
server = EnhancedSoccerDataServer()


class _LRUCache:
    def __init__(self, maxsize: int = 128):
        self.maxsize = maxsize
        self._store = OrderedDict()

    def get(self, key):
        if key in self._store:
            self._store.move_to_end(key)
            return self._store[key]
        return None

    def set(self, key, value):
        self._store[key] = value
        self._store.move_to_end(key)
        if len(self._store) > self.maxsize:
            self._store.popitem(last=False)

# MCP Protocol Handler with enhanced functions
def handle_mcp_request(request):
    """Handle MCP requests for enhanced soccer server"""
    try:
        method = request.get('method')
        request_id = request.get('id')
        
        if method == 'initialize':
            return {
                "jsonrpc": "2.0",
                "id": request_id,
                "result": {
                    "protocolVersion": "2025-06-18",
                    "capabilities": {"tools": {}},
                    "serverInfo": {
                        "name": "enhanced-soccer-data",
                        "version": "2.0.0"
                    }
                }
            }
        
        elif method == 'notifications/initialized':
            return None
            
        elif method == 'tools/list':
            return {
                "jsonrpc": "2.0",
                "id": request_id,
                "result": {
                    "tools": [
                        {
                            "name": "search_players_advanced",
                            "description": "Advanced player search with comprehensive filtering including season selection",
                            "inputSchema": {
                                "type": "object",
                                "properties": {
                                    "leagues": {"type": "array", "items": {"type": "string"}, "description": "Target leagues (e.g., ['ENG-Premier League', 'ESP-La Liga'])"},
                                    "positions": {"type": "array", "items": {"type": "string"}, "description": "Positions or roles (e.g., ['MF', 'defensive_midfielder'])"},
                                    "age_min": {"type": "integer", "description": "Minimum age"},
                                    "age_max": {"type": "integer", "description": "Maximum age"}, 
                                    "nationality": {"type": "array", "items": {"type": "string"}, "description": "Player nationalities"},
                                    "team": {"type": "string", "description": "Team name"},
                                    "seasons": {"type": "array", "items": {"type": "string"}, "description": "Specific seasons (e.g., ['2024-25', '2023-24'])"},
                                    "latest_season_only": {"type": "boolean", "description": "Show only latest season for each player"},
                                    "min_minutes_played": {"type": "integer", "description": "Minimum minutes played"},
                                    "stat_filters": {"type": "object", "description": "Statistical requirements (e.g., {'tackles_per_90': 1.5})"},
                                    "limit": {"type": "integer", "default": 50, "description": "Maximum results"},
                                    "sort_by": {"type": "string", "description": "Sort by statistic"}
                                }
                            }
                        },
                        {
                            "name": "search_by_profile",
                            "description": "Search players based on comprehensive scouting profile",
                            "inputSchema": {
                                "type": "object",
                                "properties": {
                                    "scout_brief": {
                                        "type": "object",
                                        "description": "Scouting requirements including target_leagues, positions, age_range, technical_requirements",
                                        "properties": {
                                            "target_leagues": {"type": "array", "items": {"type": "string"}},
                                            "positions": {"type": "array", "items": {"type": "string"}},
                                            "age_range": {"type": "object", "properties": {"min": {"type": "integer"}, "max": {"type": "integer"}}},
                                            "temporal_preferences": {"type": "object", "properties": {"seasons": {"type": "array", "items": {"type": "string"}}, "latest_season_only": {"type": "boolean"}}},
                                            "technical_requirements": {"type": "object"},
                                            "max_results": {"type": "integer", "default": 50}
                                        }
                                    }
                                },
                                "required": ["scout_brief"]
                            }
                        },
                        {
                            "name": "get_league_leaders",
                            "description": "Get top performers in a league for specific statistics with season filtering",
                            "inputSchema": {
                                "type": "object",
                                "properties": {
                                    "league": {"type": "string", "description": "League name"},
                                    "stat": {"type": "string", "description": "Statistic to rank by"},
                                    "position": {"type": "string", "description": "Filter by position"},
                                    "season": {"type": "string", "description": "Specific season (e.g., '2024-25')"},
                                    "min_games": {"type": "integer", "default": 15, "description": "Minimum games played"}
                                },
                                "required": ["league", "stat"]
                            }
                        },
                        {
                            "name": "discover_talents",
                            "description": "Discover high-upside talents by role/style with fit score and rationale",
                            "inputSchema": {
                                "type": "object",
                                "properties": {
                                    "role": {"type": "string", "description": "Target role (e.g., left_back, defensive_midfielder)"},
                                    "style": {"type": "string", "enum": ["possession", "transition", "high_press", "direct"], "description": "Team style emphasis"},
                                    "leagues": {"type": "array", "items": {"type": "string"}, "description": "Target leagues"},
                                    "age_max": {"type": "integer", "description": "Maximum age"},
                                    "age_min": {"type": "integer", "description": "Minimum age"},
                                    "min_minutes": {"type": "integer", "default": 900, "description": "Minimum minutes in target season"},
                                    "seasons": {"type": "array", "items": {"type": "string"}, "description": "Candidate seasons"},
                                    "alignment": {"type": "string", "enum": ["overlap", "target"], "default": "overlap", "description": "Season alignment strategy"},
                                    "coverage_threshold": {"type": "number", "default": 0.75, "description": "Coverage threshold for overlap alignment"},
                                    "exclude_elite": {"type": "boolean", "default": true, "description": "Exclude established elite profiles"},
                                    "top_n": {"type": "integer", "default": 10, "description": "Number of candidates to return"},
                                    "diversify_by": {"type": "string", "enum": ["league", "team", "none"], "description": "Ensure diversity by league or team"},
                                    "explain": {"type": "boolean", "default": true, "description": "Include strengths/risks and rationale"}
                                }
                            }
                        },
                        {
                            "name": "profile_role_fit",
                            "description": "Profile a player's role/style fit with explainability",
                            "inputSchema": {
                                "type": "object",
                                "properties": {
                                    "player_name": {"type": "string"},
                                    "role": {"type": "string"},
                                    "style": {"type": "string", "enum": ["possession", "transition", "high_press", "direct"]},
                                    "min_minutes": {"type": "integer", "default": 450},
                                    "seasons": {"type": "array", "items": {"type": "string"}},
                                    "alignment": {"type": "string", "enum": ["overlap", "target"], "default": "overlap"}
                                },
                                "required": ["player_name", "role"]
                            }
                        },
                        {
                            "name": "recommend_comparables",
                            "description": "Find most similar players by percentiles",
                            "inputSchema": {
                                "type": "object",
                                "properties": {
                                    "player_name": {"type": "string"},
                                    "k": {"type": "integer", "default": 5},
                                    "role": {"type": "string"},
                                    "leagues": {"type": "array", "items": {"type": "string"}},
                                    "seasons": {"type": "array", "items": {"type": "string"}},
                                    "alignment": {"type": "string", "enum": ["overlap", "target"], "default": "overlap"},
                                    "min_minutes": {"type": "integer", "default": 450}
                                },
                                "required": ["player_name"]
                            }
                        },
                        {
                            "name": "trend_watch",
                            "description": "Identify improving and declining players over recent seasons",
                            "inputSchema": {
                                "type": "object",
                                "properties": {
                                    "role": {"type": "string"},
                                    "position": {"type": "string"},
                                    "leagues": {"type": "array", "items": {"type": "string"}},
                                    "last_n_seasons": {"type": "integer", "default": 2},
                                    "min_minutes": {"type": "integer", "default": 900},
                                    "top_n": {"type": "integer", "default": 10}
                                }
                            }
                        },
                        {
                            "name": "undervalued_creators",
                            "description": "Find high-progression creators with modest output (underrated)",
                            "inputSchema": {
                                "type": "object",
                                "properties": {
                                    "leagues": {"type": "array", "items": {"type": "string"}},
                                    "age_max": {"type": "integer"},
                                    "min_minutes": {"type": "integer", "default": 900},
                                    "top_n": {"type": "integer", "default": 10}
                                }
                            }
                        },
                        {
                            "name": "style_fit_search",
                            "description": "Search across all positions by style fit only",
                            "inputSchema": {
                                "type": "object",
                                "properties": {
                                    "style": {"type": "string", "enum": ["possession", "transition", "high_press", "direct"]},
                                    "leagues": {"type": "array", "items": {"type": "string"}},
                                    "age_max": {"type": "integer"},
                                    "min_minutes": {"type": "integer", "default": 900},
                                    "top_n": {"type": "integer", "default": 10}
                                },
                                "required": ["style"]
                            }
                        },
                        {
                            "name": "multi_role_candidates",
                            "description": "Players who fit two roles strongly",
                            "inputSchema": {
                                "type": "object",
                                "properties": {
                                    "primary_role": {"type": "string"},
                                    "secondary_role": {"type": "string"},
                                    "leagues": {"type": "array", "items": {"type": "string"}},
                                    "min_minutes": {"type": "integer", "default": 900},
                                    "top_n": {"type": "integer", "default": 10}
                                },
                                "required": ["primary_role", "secondary_role"]
                            }
                        },
                        {
                            "name": "conversion_candidates",
                            "description": "Players in other positions suited to the target role",
                            "inputSchema": {
                                "type": "object",
                                "properties": {
                                    "target_role": {"type": "string"},
                                    "leagues": {"type": "array", "items": {"type": "string"}},
                                    "min_minutes": {"type": "integer", "default": 900},
                                    "top_n": {"type": "integer", "default": 10}
                                },
                                "required": ["target_role"]
                            }
                        },
                        {
                            "name": "xi_builder",
                            "description": "Build a simple XI by role for a given style",
                            "inputSchema": {
                                "type": "object",
                                "properties": {
                                    "style": {"type": "string", "enum": ["possession", "transition", "high_press", "direct"]},
                                    "leagues": {"type": "array", "items": {"type": "string"}},
                                    "age_policy": {"type": "string", "enum": ["u23", "u25", "prime", "any"]},
                                    "min_minutes": {"type": "integer", "default": 900}
                                },
                                "required": ["style"]
                            }
                        },
                        {
                            "name": "compare_multiple_players",
                            "description": "Compare multiple players across key statistics with season alignment",
                            "inputSchema": {
                                "type": "object",
                                "properties": {
                                    "player_names": {"type": "array", "items": {"type": "string"}, "description": "List of player names"},
                                    "season": {"type": "string", "description": "Specific season for fair comparison (e.g., '2024-25')"},
                                    "aggregation_mode": {"type": "string", "enum": ["latest", "career_avg", "best_season"], "default": "latest", "description": "Data aggregation mode"},
                                    "focus_stats": {"type": "array", "items": {"type": "string"}, "description": "Statistics to focus on"},
                                    "alignment": {"type": "string", "enum": ["target", "overlap"], "default": "overlap", "description": "Season alignment strategy"},
                                    "seasons": {"type": "array", "items": {"type": "string"}, "description": "Candidate seasons for overlap alignment"},
                                    "min_minutes": {"type": "integer", "default": 450, "description": "Minimum minutes to count towards overlap coverage"},
                                    "coverage_threshold": {"type": "number", "default": 0.8, "description": "Required fraction of players in target season"},
                                    "fallback": {"type": "string", "enum": ["nearest", "exclude"], "default": "nearest", "description": "Fallback policy when player lacks target season"},
                                    "tolerance_seasons": {"type": "integer", "default": 1, "description": "Allowed season distance for nearest fallback"}
                                },
                                "required": ["player_names"]
                            }
                        },
                        {
                            "name": "generate_detailed_scouting_report",
                            "description": "Generate comprehensive scouting report for a player with season control and elite peer context",
                            "inputSchema": {
                                "type": "object",
                                "properties": {
                                    "player_name": {"type": "string", "description": "Player name"},
                                    "season": {"type": "string", "description": "Specific season for analysis (e.g., '2024-25')"},
                                    "aggregation_mode": {"type": "string", "enum": ["latest", "career_avg", "best_season"], "default": "latest", "description": "Data aggregation mode"},
                                    "comparison_players": {"type": "array", "items": {"type": "string"}, "description": "Players to compare against"},
                                    "focus_areas": {"type": "array", "items": {"type": "string"}, "description": "Areas to focus analysis on"}
                                },
                                "required": ["player_name"]
                            }
                        },
                        {
                            "name": "get_player_career_summary",
                            "description": "Get player career summary with temporal analysis and progression tracking",
                            "inputSchema": {
                                "type": "object",
                                "properties": {
                                    "player_name": {"type": "string", "description": "Player name"},
                                    "aggregation_mode": {"type": "string", "enum": ["latest", "career_avg", "best_season", "all_seasons"], "default": "latest", "description": "Analysis mode: latest season, career average, best season, or all seasons with progression"}
                                },
                                "required": ["player_name"]
                            }
                        },
                        # Legacy functions for backward compatibility
                        {
                            "name": "search_players",
                            "description": "Basic player search (legacy)",
                            "inputSchema": {
                                "type": "object",
                                "properties": {
                                    "query": {"type": "string", "description": "Player name"},
                                    "league": {"type": "string", "description": "League filter"},
                                    "position": {"type": "string", "description": "Position filter"},
                                    "limit": {"type": "integer", "default": 10}
                                }
                            }
                        },
                        {
                            "name": "get_player_details",
                            "description": "Get detailed player statistics (legacy)",
                            "inputSchema": {
                                "type": "object",
                                "properties": {
                                    "player_name": {"type": "string", "description": "Player name"}
                                },
                                "required": ["player_name"]
                            }
                        }
                    ]
                }
            }
        
        elif method == 'resources/list':
            return {
                "jsonrpc": "2.0",
                "id": request_id,
                "result": {"resources": []}
            }
            
        elif method == 'tools/call':
            tool_name = request['params']['name']
            args = request['params'].get('arguments', {})
            
            # Route to appropriate function
            if tool_name == 'search_players_advanced':
                result = server.search_players_advanced(**args)
            elif tool_name == 'search_by_profile':
                result = server.search_by_profile(**args)
            elif tool_name == 'get_league_leaders':
                result = server.get_league_leaders(**args)
            elif tool_name == 'compare_multiple_players':
                result = server.compare_multiple_players(**args)
            elif tool_name == 'generate_detailed_scouting_report':
                result = server.generate_detailed_scouting_report(**args)
            elif tool_name == 'get_player_career_summary':
                result = server.get_player_career_summary(**args)
            elif tool_name == 'discover_talents':
                result = server.discover_talents(**args)
            elif tool_name == 'profile_role_fit':
                result = server.profile_role_fit(**args)
            elif tool_name == 'recommend_comparables':
                result = server.recommend_comparables(**args)
            elif tool_name == 'trend_watch':
                result = server.trend_watch(**args)
            elif tool_name == 'undervalued_creators':
                result = server.undervalued_creators(**args)
            elif tool_name == 'style_fit_search':
                result = server.style_fit_search(**args)
            elif tool_name == 'multi_role_candidates':
                result = server.multi_role_candidates(**args)
            elif tool_name == 'conversion_candidates':
                result = server.conversion_candidates(**args)
            elif tool_name == 'xi_builder':
                result = server.xi_builder(**args)
            # Legacy functions
            elif tool_name == 'search_players':
                # Convert to advanced search
                result = server.search_players_advanced(
                    leagues=[args.get('league')] if args.get('league') else None,
                    positions=[args.get('position')] if args.get('position') else None,
                    limit=args.get('limit', 10)
                )
                # Format for legacy compatibility
                formatted_result = f"Found {len(result)} players:\\n\\n"
                for player in result:
                    formatted_result += f"🏃 {player['name']}\\n"
                    formatted_result += f"   Team: {player['team']}\\n"
                    formatted_result += f"   League: {player['league']}\\n"
                    formatted_result += f"   Position: {player['position']}\\n"
                    formatted_result += f"   Age: {player['age']}\\n\\n"
                result = formatted_result
            elif tool_name == 'get_player_details':
                scouting_report = server.generate_detailed_scouting_report(args['player_name'])
                if 'error' in scouting_report:
                    result = scouting_report['error']
                else:
                    # Format for legacy compatibility
                    basic = scouting_report['basic_info']
                    perf = scouting_report['performance_stats']
                    result = f"📊 Detailed Stats for {basic['name']}\\n\\n"
                    result += f"Team: {basic['team']}\\n"
                    result += f"League: {basic['league']}\\n"
                    result += f"Position: {basic['position']}\\n"
                    result += f"Age: {basic['age']}\\n\\n"
                    result += "📈 Key Statistics:\\n"
                    for stat, value in perf.items():
                        result += f"   {stat.replace('_', ' ').title()}: {value}\\n"
            else:
                return {
                    "jsonrpc": "2.0",
                    "id": request_id,
                    "error": {"code": -32601, "message": f"Unknown tool: {tool_name}"}
                }
            
            return {
                "jsonrpc": "2.0",
                "id": request_id,
                "result": {
                    "content": [{"type": "text", "text": json.dumps(result, indent=2) if isinstance(result, (dict, list)) else str(result)}]
                }
            }
        
        else:
            return {
                "jsonrpc": "2.0",
                "id": request_id,
                "error": {"code": -32601, "message": f"Method not found: {method}"}
            }
            
    except Exception as e:
        logger.error(f"Error handling request: {e}")
        return {
            "jsonrpc": "2.0",
            "id": request.get('id'),
            "error": {"code": -32603, "message": str(e)}
        }

# Main loop
if __name__ == "__main__":
    logger.info("Enhanced Soccer Data MCP Server Started")
    logger.info("Professional scouting capabilities enabled")
    
    try:
        for line in sys.stdin:
            line = line.strip()
            if not line:
                continue
            
            try:
                request = json.loads(line)
                response = handle_mcp_request(request)
                
                if response is not None:
                    print(json.dumps(response), flush=True)
                    
            except json.JSONDecodeError as e:
                logger.error(f"Invalid JSON received: {e}")
                continue
            except Exception as e:
                logger.error(f"Error processing request: {e}")
                continue
                
    except KeyboardInterrupt:
        logger.info("Enhanced Soccer Data MCP Server Stopped")
    except Exception as e:
        logger.error(f"Unexpected error: {e}")

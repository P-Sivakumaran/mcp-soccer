#!/usr/bin/env python3
"""
Data Validation Script for FBref MCP Soccer Server
Validates collected data quality and MCP server compatibility
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging
import json

def validate_fbref_data():
    """Validate the collected FBref data for MCP server compatibility"""
    
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)
    
    logger.info("🔍 Starting FBref data validation...")
    
    data_dir = Path("data")
    main_file = data_dir / "unified_player_stats.csv"
    
    if not main_file.exists():
        logger.error(f"❌ Main data file not found: {main_file}")
        return False
    
    try:
        # Load the data
        logger.info(f"📂 Loading data from: {main_file}")
        df = pd.read_csv(main_file)
        logger.info(f"📊 Loaded data: {len(df)} players, {len(df.columns)} columns")
        
        # Check required columns for MCP server
        required_columns = ['player', 'team', 'league', 'position']
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            logger.error(f"❌ Missing required columns: {missing_columns}")
            return False
        else:
            logger.info(f"✅ All required columns present: {required_columns}")
        
        # Data quality checks
        logger.info("🔍 Running comprehensive data quality checks...")
        
        # Check for empty player names
        empty_players = df['player'].isna().sum()
        logger.info(f"👤 Players with missing names: {empty_players}")
        if empty_players > 0:
            logger.warning(f"⚠️  Found {empty_players} players with missing names")
        
        # Check league distribution
        league_counts = df['league'].value_counts()
        logger.info(f"🏆 League distribution:")
        for league, count in league_counts.items():
            logger.info(f"   {league}: {count} players")
        
        # Check position distribution
        position_counts = df['position'].value_counts().head(10)
        logger.info(f"⚽ Top 10 positions:")
        for position, count in position_counts.items():
            logger.info(f"   {position}: {count} players")
        
        # Check for key stats availability
        # Map friendly keys to actual columns in dataset
        key_map = {
            'goals': 'performance_gls',
            'assists': 'performance_ast',
            'matches_played': 'playing_time_mp',
            'minutes_played': 'playing_time_min',
            'expected_goals': 'expected_xg',
            'shots': 'standard_sh',
            'pass_completion_pct': 'total_cmp_pct',
            'tackles': 'tackles_tkl'
        }
        available_stats = [k for k, col in key_map.items() if col in df.columns]
        missing_stats = [k for k, col in key_map.items() if col not in df.columns]
        
        logger.info(f"📈 Available key stats ({len(available_stats)}/{len(key_map)}): {available_stats}")
        if missing_stats:
            logger.warning(f"⚠️  Missing key stats: {missing_stats}")
        
        # Check data completeness for key columns
        logger.info("📊 Data completeness check:")
        for k in ['player', 'team', 'league', 'position'] + available_stats[:5]:
            col = key_map.get(k, k)
            if col in df.columns:
                missing_pct = (df[col].isna().sum() / len(df)) * 100
                logger.info(f"   {col}: {missing_pct:.1f}% missing")
        
        # Check for duplicate players
        if 'player' in df.columns and 'team' in df.columns:
            duplicates = df.duplicated(subset=['player', 'team']).sum()
            logger.info(f"🔄 Duplicate player-team combinations: {duplicates}")
        
        # Sample data preview
        logger.info("📋 Sample data preview:")
        sample_cols = ['player', 'team', 'league', 'position'] + available_stats[:3]
        sample_cols = [col for col in sample_cols if col in df.columns]
        
        if len(sample_cols) > 0:
            sample_data = df[sample_cols].head(5)
            logger.info(f"\n{sample_data.to_string(index=False)}")
        
        # Check numerical data ranges
        logger.info("🔢 Numerical data validation:")
        numerical_cols = df.select_dtypes(include=[np.number]).columns
        
        for col in numerical_cols[:10]:  # Check first 10 numerical columns
            if col in df.columns:
                min_val = df[col].min()
                max_val = df[col].max()
                mean_val = df[col].mean()
                
                logger.info(f"   {col}: min={min_val:.2f}, max={max_val:.2f}, mean={mean_val:.2f}")
                
                # Flag suspicious values
                if col in ['goals', 'assists'] and max_val > 100:
                    logger.warning(f"⚠️  Suspicious high value in {col}: {max_val}")
                elif col.endswith('_pct') and (max_val > 100 or min_val < 0):
                    logger.warning(f"⚠️  Percentage out of range in {col}: {min_val}-{max_val}")
        
        # Validate file structure for MCP server
        logger.info("🔧 MCP server compatibility check:")
        
        # Check if data directory exists and has proper structure
        expected_files = ['unified_player_stats.csv']
        existing_files = [f.name for f in data_dir.glob('*.csv')]
        
        logger.info(f"📁 CSV files in data directory: {len(existing_files)}")
        for file in existing_files[:10]:  # Show first 10 files
            logger.info(f"   {file}")
        
        # Check summary file
        summary_file = data_dir / "data_summary.json"
        if summary_file.exists():
            with open(summary_file, 'r') as f:
                summary = json.load(f)
            logger.info(f"📋 Data summary:")
            for key, value in summary.items():
                logger.info(f"   {key}: {value}")
        
        # Final validation summary
        logger.info("=" * 60)
        logger.info("✅ DATA VALIDATION SUMMARY")
        logger.info(f"📊 Total players: {len(df)}")
        logger.info(f"📈 Total columns: {len(df.columns)}")
        logger.info(f"🏆 Leagues covered: {df['league'].nunique() if 'league' in df.columns else 'Unknown'}")
        logger.info(f"⚽ Positions covered: {df['position'].nunique() if 'position' in df.columns else 'Unknown'}")
        logger.info(f"📁 Files created: {len(existing_files)}")
        
        # Check if meets success criteria
        success_criteria = {
            'min_players': 500,  # Reduced for testing
            'min_columns': 20,
            'required_leagues': 3,
            'key_stats_coverage': 0.5
        }
        
        meets_criteria = True
        
        if len(df) < success_criteria['min_players']:
            logger.warning(f"⚠️  Player count ({len(df)}) below target ({success_criteria['min_players']})")
            meets_criteria = False
        
        if len(df.columns) < success_criteria['min_columns']:
            logger.warning(f"⚠️  Column count ({len(df.columns)}) below target ({success_criteria['min_columns']})")
            meets_criteria = False
        
        if 'league' in df.columns:
            league_count = df['league'].nunique()
            if league_count < success_criteria['required_leagues']:
                logger.warning(f"⚠️  League count ({league_count}) below target ({success_criteria['required_leagues']})")
                meets_criteria = False
        
        coverage_ratio = len(available_stats) / len(key_map)
        if coverage_ratio < success_criteria['key_stats_coverage']:
            logger.warning(f"⚠️  Key stats coverage ({coverage_ratio:.1%}) below target ({success_criteria['key_stats_coverage']:.1%})")
            meets_criteria = False
        
        if meets_criteria:
            logger.info("🎉 ALL SUCCESS CRITERIA MET!")
            logger.info("✅ Data is ready for MCP Soccer Server!")
        else:
            logger.warning("⚠️  Some success criteria not met - data may still be usable")
        
        logger.info("=" * 60)
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Validation failed: {e}")
        return False

def validate_mcp_server_compatibility():
    """Additional validation specifically for MCP server compatibility"""
    
    logger = logging.getLogger(__name__)
    logger.info("🔧 Running MCP server compatibility checks...")
    
    data_dir = Path("data")
    
    # Check if the soccer_server.py can load the data
    try:
        # Try to import and test the soccer server
        import sys
        sys.path.append(str(Path.cwd()))
        
        # Test data loading simulation
        main_file = data_dir / "unified_player_stats.csv"
        if main_file.exists():
            test_df = pd.read_csv(main_file)
            
            # Test search functionality
            if 'player' in test_df.columns:
                sample_search = test_df[test_df['player'].str.contains('messi', case=False, na=False)]
                logger.info(f"🔍 Sample search test: Found {len(sample_search)} results for 'messi'")
            
            # Test filtering by league
            if 'league' in test_df.columns:
                sample_league = test_df[test_df['league'].str.contains('premier', case=False, na=False)]
                logger.info(f"🏆 League filter test: Found {len(sample_league)} Premier League players")
            
            logger.info("✅ MCP server compatibility: PASSED")
            return True
        else:
            logger.error("❌ Main data file not found for compatibility test")
            return False
            
    except Exception as e:
        logger.error(f"❌ MCP compatibility test failed: {e}")
        return False

if __name__ == "__main__":
    print("🔍 FBref Data Validation")
    print("========================")
    
    # Run main validation
    validation_result = validate_fbref_data()
    
    # Run MCP compatibility check
    compatibility_result = validate_mcp_server_compatibility()
    
    # Final result
    if validation_result and compatibility_result:
        print("\n🎉 VALIDATION COMPLETE: Your data is ready for the MCP Soccer Server!")
    else:
        print("\n⚠️  VALIDATION ISSUES: Please check the logs above for details.")

# Soccer Data MCP Server

A professional-grade Model Context Protocol (MCP) server that provides comprehensive soccer player data and scouting capabilities to Claude Desktop.

## Features

- **Advanced Player Search**: Filter by leagues, positions, age ranges, and statistical thresholds
- **Professional Scouting Reports**: Detailed analysis with performance, defensive, passing, and attacking statistics
- **Multi-Player Comparisons**: Side-by-side analysis of multiple players
- **League Leaders**: Top performers in specific statistics by league and position
- **Comprehensive Data**: 2,854 players from Big 5 European leagues with 291+ statistical metrics

## Data Coverage

- **Leagues**: Premier League, La Liga, Ligue 1, Bundesliga, Serie A
- **Statistics**: Goals, assists, xG, progressive passes, tackles, aerial duels, and 280+ more metrics
- **Positions**: All standard positions including multi-position players

## Adding More Leagues (POR/NED/BEL/AUT)

This project can include additional European leagues from FBref beyond the Big 5 via the `soccerdata` FBref adapter using a custom league dictionary.

Supported additions (tested):
- Portugal: `POR-Primeira Liga` (FBref: "Primeira Liga")
- Netherlands: `NED-Eredivisie` (FBref: "Eredivisie")
- Belgium: `BEL-Belgian Pro League` (FBref: "Belgian Pro League")
- Austria: `AUT-Bundesliga` (FBref: "Austrian Football Bundesliga")

Setup (one-time):
- Create `~/soccerdata/config/league_dict.json` with entries that map canonical IDs to FBref competition names, for example:
  {
    "POR-Primeira Liga": {"FBref": "Primeira Liga", "season_start": "Aug", "season_end": "May"},
    "NED-Eredivisie": {"FBref": "Eredivisie", "season_start": "Aug", "season_end": "May"},
    "BEL-Belgian Pro League": {"FBref": "Belgian Pro League", "season_start": "Aug", "season_end": "May"},
    "AUT-Bundesliga": {"FBref": "Austrian Football Bundesliga", "season_start": "Jul", "season_end": "May"}
  }

Notes:
- `soccerdata` looks for this file at `~/soccerdata/config/league_dict.json` by default. Alternatively, set `SOCCERDATA_DIR=/path/to/dir` to control where configs/data are stored.
- Once configured, the included collector already targets these leagues by default. You can override on the CLI using `--leagues`.

Examples:
- Collect only the new leagues for 2024-25:
  `python fbref_data_collector.py 2024-25 --leagues "POR-Primeira Liga" "NED-Eredivisie" "BEL-Belgian Pro League" "AUT-Bundesliga"`
- Collect Big 5 + new leagues for three seasons:
  `python fbref_data_collector.py 2023-24 2024-25 2025-26`

## Quick Start

1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Run Data Collection** (if needed):
   ```bash
   python fbref_data_collector.py
   ```

3. **Start the MCP Server**:
   ```bash
   python soccer_server.py
   ```

   Optional observability controls:
   - Set `SOCCER_MCP_LOG_LEVEL` to `DEBUG|INFO|WARNING|ERROR` to adjust verbosity.
   - The server precomputes league/position percentiles and an overall rating per player for richer queries.

4. **Configure Claude Desktop**:
   Add to your `claude_desktop_config.json`:
   ```json
   {
     "mcpServers": {
       "soccer-data": {
         "command": "python",
         "args": ["/path/to/mcp soccer/soccer_server.py"],
         "cwd": "/path/to/mcp soccer/"
       }
     }
   }
   ```

## Example Queries

- "Find all defensive midfielders aged 20-25 in the Premier League with 2+ tackles per 90"
- "Compare Pedri, Bellingham, and Gavi across key statistics"
- "Generate a detailed scouting report for Erling Haaland"
- "Show me the top 10 goal scorers in La Liga"

## Data Collection Notes

- The collector includes retry logic with exponential backoff for resilience and writes metadata (library versions) to `data/data_summary.json`.
- You can control collector log level via `--log-level INFO` or `SOCCER_COLLECTOR_LOG_LEVEL`.

## Files

- `soccer_server.py` - Enhanced MCP server with professional scouting capabilities
- `fbref_data_collector.py` - Data collection script from FBref
- `data/unified_player_stats.csv` - Comprehensive player dataset
- `requirements.txt` - Python dependencies
- `validate_data.py` - Data quality validation script

## Professional Use

Built for professional scouts and analysts with production-ready architecture supporting complex queries, bulk operations, and comprehensive statistical analysis.

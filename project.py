import os
import mysql.connector
import pandas as pd
import numpy as np
from pybaseball import statcast
from xgboost import XGBClassifier, XGBRegressor
from datetime import timedelta
import warnings
import pybaseball
import requests
from io import StringIO
from tqdm import tqdm

print("Starting Script....")
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.filterwarnings('ignore', message='.*_OVERSIZE_WARNING.*')
warnings.filterwarnings('ignore', category=UserWarning, module="pandas")
warnings.filterwarnings('ignore', category=UserWarning)
pybaseball.cache.enable()

# --- 1. INITIAL DATA DOWNLOAD (The "Master" Download) ---
print("📥 Downloading 2025 Statcast Data (Full Season)...")
raw_df = statcast(start_dt='2025-01-01', end_dt='2025-11-30')
print(f"✅ Download Complete: {len(raw_df)} rows.")

def get_db_conn():
    return mysql.connector.connect(
        host=os.environ.get("DB_HOST"),
        user=os.environ.get("DB_USER"),
        password=os.environ.get("DB_PASSWORD"),
        port=int(os.environ.get("DB_PORT", 3306)),
        database=os.environ.get("DB_NAME"),
        ssl_disabled=False
    )

# ==========================================
# MODULE 1: xBA (Expected Batting Average)
# ==========================================
print("\n--- Starting xBA Module (Park-Adjusted) ---")
conn = get_db_conn()
cursor = conn.cursor()

# Fetch training data
train_df_xba = pd.read_sql("""
    SELECT s.launch_speed, s.launch_angle, v.park_hit_rate, s.hit
    FROM statcast_data s
    LEFT JOIN venue_xba_factor v
        ON s.venue_id = v.venue_id
    WHERE s.launch_speed IS NOT NULL
      AND s.launch_angle IS NOT NULL
""", conn)

league_avg = train_df_xba['park_hit_rate'].mean()
train_df_xba['park_hit_rate'] = train_df_xba['park_hit_rate'].fillna(league_avg)

# Train xBA model
model_xba = XGBClassifier(
    n_estimators=300, max_depth=6, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.8,
    eval_metric='logloss', use_label_encoder=False
)
model_xba.fit(
    train_df_xba[['launch_speed', 'launch_angle', 'park_hit_rate']],
    train_df_xba['hit']
)

# Prepare prediction dataset
xba_df = raw_df[[ 
    'events', 'hit_distance_sc', 'launch_speed', 'launch_angle',
    'game_date', 'home_team', 'away_team', 'inning_topbot', 'game_pk',
    'batter', 'pitcher', 'at_bat_number', 'pitch_number'
]].copy()
xba_df = xba_df[xba_df['events'].notna()]

xba_df['team_batting'] = np.where(xba_df['inning_topbot'] == 'Top', xba_df['away_team'], xba_df['home_team'])
xba_df['team_pitching'] = np.where(xba_df['inning_topbot'] == 'Top', xba_df['home_team'], xba_df['away_team'])

# Unique IDs
xba_df['unique_game_id'] = xba_df['game_pk'].astype(str) + '_' + xba_df['game_date'].astype(str)
xba_df['gameid'] = xba_df['game_date'].astype(str) + "_" + xba_df['home_team'] + "_" + xba_df['away_team']
xba_df['AB_id'] = xba_df['game_pk'].astype(str) + "_" + xba_df['at_bat_number'].astype(str) + "_" + xba_df['pitch_number'].astype(str)

# Attach Park Factor
venue_lookup = pd.read_sql("SELECT DISTINCT game_pk, venue_id FROM statcast_data", conn)
park_lookup = pd.read_sql("SELECT venue_id, park_hit_rate FROM venue_xba_factor", conn)
xba_df = xba_df.merge(venue_lookup, on='game_pk', how='left')
xba_df = xba_df.merge(park_lookup, on='venue_id', how='left')
xba_df['park_hit_rate'] = xba_df['park_hit_rate'].fillna(league_avg)

# Predict xBA
predict_mask = xba_df[['launch_speed', 'launch_angle', 'park_hit_rate']].notnull().all(axis=1)
xba_df['xBA'] = np.nan
xba_df.loc[predict_mask, 'xBA'] = model_xba.predict_proba(
    xba_df.loc[predict_mask, ['launch_speed', 'launch_angle', 'park_hit_rate']]
)[:, 1]
xba_df.loc[xba_df['events'].isin(['strikeout', 'strikeout_double_play']), 'xBA'] = 0.0
xba_df = xba_df[xba_df['xBA'].notnull()]

# Create xBA table if not exists
cursor.execute("""
CREATE TABLE IF NOT EXISTS statcast_2025 (
    unique_game_id VARCHAR(50),
    events VARCHAR(255),
    hit_distance_sc FLOAT,
    launch_speed FLOAT,
    launch_angle FLOAT,
    game_date DATE,
    team_batting VARCHAR(255),
    team_pitching VARCHAR(255),
    gameid VARCHAR(255),
    gamepk BIGINT,
    player_id BIGINT,
    pitcher_id BIGINT,
    park_hit_rate FLOAT,
    xBA FLOAT,
    AB_id VARCHAR(255) PRIMARY KEY UNIQUE
)
""")

# Upload xBA with incremental upsert
data_xba = [
    tuple(None if pd.isna(x) else int(x) if isinstance(x, (np.integer, np.int64)) else x
          for x in row)
    for row in xba_df[['unique_game_id', 'events', 'hit_distance_sc', 'launch_speed', 'launch_angle',
                       'game_date', 'team_batting', 'team_pitching', 'gameid', 'game_pk',
                       'batter', 'pitcher', 'park_hit_rate', 'xBA', 'AB_id']].itertuples(index=False, name=None)
]

insert_query = """
INSERT INTO statcast_2025 (
    unique_game_id, events, hit_distance_sc, launch_speed, launch_angle,
    game_date, team_batting, team_pitching, gameid, gamepk,
    player_id, pitcher_id, park_hit_rate, xBA, AB_id
)
VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
ON DUPLICATE KEY UPDATE
    events = VALUES(events),
    hit_distance_sc = VALUES(hit_distance_sc),
    launch_speed = VALUES(launch_speed),
    launch_angle = VALUES(launch_angle),
    team_batting = VALUES(team_batting),
    team_pitching = VALUES(team_pitching),
    gameid = VALUES(gameid),
    gamepk = VALUES(gamepk),
    player_id = VALUES(player_id),
    pitcher_id = VALUES(pitcher_id),
    park_hit_rate = VALUES(park_hit_rate),
    xBA = VALUES(xBA)
"""

for i in tqdm(range(0, len(data_xba), 5000), desc="Uploading xBA"):
    cursor.executemany(insert_query, data_xba[i:i+5000])
    conn.commit()

# ==========================================
# MODULE 2: xRuns
# ==========================================
# (Same approach as above: use AB_id as UNIQUE key for incremental updates)
# Create table if not exists
cursor.execute("""
CREATE TABLE IF NOT EXISTS expected_runs_2025 (
    unique_game_id VARCHAR(50),
    game_pk INT,
    game_id VARCHAR(50),
    game_date DATE,
    home_team VARCHAR(10),
    away_team VARCHAR(10),
    inning_topbot VARCHAR(5),
    team_batting VARCHAR(10),
    team_pitching VARCHAR(10),
    batter BIGINT,
    pitcher BIGINT,
    hit_distance_sc FLOAT,
    launch_angle FLOAT,
    launch_speed FLOAT,
    outs_when_up INT,
    runners_on_base INT,
    runs_scored INT,
    expected_runs FLOAT,
    events VARCHAR(50),
    description VARCHAR(100),
    AB_id VARCHAR(255) PRIMARY KEY UNIQUE,
    inning INT,
    park_run_factor FLOAT
)
""")

# ... rest of xRuns preprocessing and prediction ...
# then upload with:
xruns_insert_query = """
INSERT INTO expected_runs_2025 (
    unique_game_id, game_pk, game_id, game_date,
    home_team, away_team, inning_topbot,
    team_batting, team_pitching, batter, pitcher,
    hit_distance_sc, launch_angle, launch_speed,
    outs_when_up, runners_on_base, runs_scored,
    expected_runs, events, description,
    AB_id, inning, park_run_factor
)
VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
ON DUPLICATE KEY UPDATE
    game_pk = VALUES(game_pk),
    game_id = VALUES(game_id),
    game_date = VALUES(game_date),
    home_team = VALUES(home_team),
    away_team = VALUES(away_team),
    inning_topbot = VALUES(inning_topbot),
    team_batting = VALUES(team_batting),
    team_pitching = VALUES(team_pitching),
    batter = VALUES(batter),
    pitcher = VALUES(pitcher),
    hit_distance_sc = VALUES(hit_distance_sc),
    launch_angle = VALUES(launch_angle),
    launch_speed = VALUES(launch_speed),
    outs_when_up = VALUES(outs_when_up),
    runners_on_base = VALUES(runners_on_base),
    runs_scored = VALUES(runs_scored),
    expected_runs = VALUES(expected_runs),
    events = VALUES(events),
    description = VALUES(description),
    inning = VALUES(inning),
    park_run_factor = VALUES(park_run_factor)
"""

# ==========================================
# MODULE 3: Game Score
# ==========================================
TEAM_ABBR_MAP = {
    "Los Angeles Angels": "LAA", "San Diego Padres": "SD", "San Francisco Giants": "SF",
    "Athletics": "ATH", "Seattle Mariners": "SEA", "New York Yankees": "NYY",
    "Philadelphia Phillies": "PHI", "Chicago Cubs": "CHC", "Washington Nationals": "WSH",
    "Milwaukee Brewers": "MIL", "Toronto Blue Jays": "TOR", "Cleveland Guardians": "CLE",
    "Miami Marlins": "MIA", "Atlanta Braves": "ATL", "Boston Red Sox": "BOS",
    "Texas Rangers": "TEX", "Arizona Diamondbacks": "AZ", "Cincinnati Reds": "CIN",
    "Baltimore Orioles": "BAL", "Minnesota Twins": "MIN", "Los Angeles Dodgers": "LAD",
    "Houston Astros": "HOU", "Kansas City Royals": "KC", "St. Louis Cardinals": "STL",
    "Colorado Rockies": "COL", "Pittsburgh Pirates": "PIT", "Chicago White Sox": "CWS",
    "Tampa Bay Rays": "TB", "New York Mets": "NYM", "Detroit Tigers": "DET"
}

def get_mlb_schedule(start_date, end_date):
    url = (f"https://statsapi.mlb.com/api/v1/schedule?sportId=1"
           f"&startDate={start_date}&endDate={end_date}&hydrate=linescore,probablePitcher")
    response = requests.get(url, timeout=30)
    response.raise_for_status()
    data = response.json()
    games = []
    for date_data in data.get("dates", []):
        for game in date_data.get("games", []):
            if game.get("gameType") in ["E"]:
                continue
            game_utc_str = game["gameDate"]
            status = game["status"]["detailedState"]
            ls = game.get("linescore", {})
            if status == "Final":
                inn_disp = ls.get("scheduledInnings", 9)
                state_disp = "Final"
            elif status == "In Progress":
                inn_disp = ls.get("currentInning", 0)
                state_disp = ls.get("inningState", "In-Progress")
            else:
                inn_disp = 0
                state_disp = "Scheduled"
            home_name = game["teams"]["home"]["team"]["name"]
            away_name = game["teams"]["away"]["team"]["name"]
            games.append({
                "game_pk": game["gamePk"],
                "game_date": game_utc_str,
                "game_time": game_utc_str,
                "game_status": status,
                "home_team": TEAM_ABBR_MAP.get(home_name, home_name),
                "away_team": TEAM_ABBR_MAP.get(away_name, away_name),
                "home_pitcher": game["teams"]["home"].get("probablePitcher", {}).get("fullName", "TBD"),
                "away_pitcher": game["teams"]["away"].get("probablePitcher", {}).get("fullName", "TBD"),
                "home_score_final": game["teams"]["home"].get("score", 0),
                "away_score_final": game["teams"]["away"].get("score", 0),
                "current_inning": inn_disp,
                "inning_state": state_disp
            })
    df = pd.DataFrame(games)
    if not df.empty:
        df['game_time'] = pd.to_datetime(df['game_time']) - timedelta(hours=4)
        df['game_time'] = df['game_time'].dt.strftime("%I:%M %p") + " ET"
        df['game_date'] = pd.to_datetime(df['game_date']) - timedelta(hours=4)
        df['game_date'] = df['game_date'].dt.strftime("%Y-%m-%d")
        df['unique_game_id'] = df['game_pk'].astype(str) + '_' + df['game_date'].astype(str)
    return df

def run_game_score_update():
    start = "2026-01-01"
    end = "2026-12-31"
    df = get_mlb_schedule(start, end)
    if df.empty:
        print("⚠️ No games found.")
        return
    df = df.replace({np.nan: None})
    data_values = [
        (
            row['unique_game_id'], row["game_pk"], row["game_date"], row["game_time"],
            row["game_status"], row["home_team"], row["away_team"],
            row["home_pitcher"], row["away_pitcher"], row["home_score_final"],
            row["away_score_final"], row["current_inning"], row["inning_state"]
        )
        for _, row in df.iterrows()
    ]
    conn = get_db_conn()
    cursor = conn.cursor()
    insert_query = """
        INSERT INTO actual_game_scores_2026 (
            unique_game_id, game_pk, game_date, game_time, game_status,
            home_team, away_team, home_pitcher, away_pitcher,
            home_score_final, away_score_final, current_inning, inning_state
        )
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        ON DUPLICATE KEY UPDATE
            game_date = VALUES(game_date),
            game_status = VALUES(game_status),
            home_pitcher = VALUES(home_pitcher),
            away_pitcher = VALUES(away_pitcher),
            home_score_final = VALUES(home_score_final),
            away_score_final = VALUES(away_score_final),
            current_inning = VALUES(current_inning),
            inning_state = VALUES(inning_state),
            game_time = VALUES(game_time)
    """
    for i in tqdm(range(0, len(data_values), 1000), desc="Uploading"):
        cursor.executemany(insert_query, data_values[i:i+1000])
        conn.commit()
    cursor.close()
    conn.close()
    print(f"✅ Loaded {len(data_values)} games (Eastern Time).")

if __name__ == "__main__":
    run_game_score_update()

# ==========================================
# MODULE 4: Player Names
# ==========================================
print("\n--- Starting Player Name Module ---")
conn = get_db_conn()
cursor = conn.cursor()
cursor.execute("SELECT DISTINCT batter FROM expected_runs_2025 UNION SELECT DISTINCT pitcher FROM expected_runs_2025")
all_ids = [r[0] for r in cursor.fetchall() if r[0] is not None]

dfs = []
for char in tqdm('0123456789abcdef', desc="Downloading Player Register"):
    url = f"https://raw.githubusercontent.com/chadwickbureau/register/master/data/people-{char}.csv"
    res = requests.get(url)
    if res.status_code == 200:
        dfs.append(pd.read_csv(StringIO(res.text), usecols=['key_mlbam', 'name_first', 'name_last']))

player_register = pd.concat(dfs, ignore_index=True)
player_lookup = player_register[player_register['key_mlbam'].isin(all_ids)].copy()
player_lookup['full_name'] = player_lookup['name_first'] + ' ' + player_lookup['name_last']
data_names = player_lookup[['key_mlbam', 'full_name']].drop_duplicates().values.tolist()

cursor.execute("DROP TABLE IF EXISTS player_names")
cursor.execute("CREATE TABLE player_names (player_id BIGINT PRIMARY KEY, full_name VARCHAR(100))")
cursor.executemany("INSERT INTO player_names (player_id, full_name) VALUES (%s, %s)", data_names)
conn.commit()
cursor.close()
conn.close()

print("\n✅ ALL MODULES COMPLETED SUCCESSFULLY!")

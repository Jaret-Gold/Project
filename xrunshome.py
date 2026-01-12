import os
import mysql.connector
import pandas as pd
import numpy as np
from pybaseball import statcast
from xgboost import XGBClassifier, XGBRegressor
from datetime import datetime
import warnings
from dateutil import parser
import pytz
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
print("📥 Downloading 2026 Statcast Data (Full Season)...")
raw_df = statcast(start_dt='2026-01-01', end_dt='2026-11-30')
print(f"✅ Download Complete: {len(raw_df)} rows.")

def get_db_conn():
    return mysql.connector.connect(
        host=os.getenv("DB_HOST"),
        user=os.getenv("DB_USER"),
        password=os.getenv("DB_PASSWORD"),
        port=int(os.getenv("DB_PORT", 15560)),
        database=os.getenv("DB_NAME", "statcast"),
        ssl_disabled=False
    )

# ==========================================
# MODULE 1: xBA (Expected Batting Average)
# ==========================================
print("\n--- Starting xBA Module ---")
conn = get_db_conn()
cursor = conn.cursor()

# Fetch training data for xBA
print("Fetching xBA training data...")
train_df_xba = pd.read_sql("SELECT events, launch_speed, launch_angle, hit FROM statcast_data", conn).dropna(subset=['launch_speed', 'launch_angle'])
model_xba = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
model_xba.fit(train_df_xba[['launch_speed', 'launch_angle']], train_df_xba['hit'])

# Process xBA
xba_df = raw_df[['events', 'hit_distance_sc', 'launch_speed', 'launch_angle', 'game_date', 'home_team', 'away_team', 'inning_topbot', 'game_pk', 'batter', 'pitcher','at_bat_number','pitch_number']].copy()
xba_df = xba_df[xba_df['events'].notna()]
xba_df['team_batting'] = np.where(xba_df['inning_topbot'] == 'Top', xba_df['away_team'], xba_df['home_team'])
xba_df['team_pitching'] = np.where(xba_df['inning_topbot'] == 'Top', xba_df['home_team'], xba_df['away_team'])
xba_df['gameid'] = xba_df['game_date'].astype(str) + "_" + xba_df['home_team'] + "_" + xba_df['away_team']
xba_df['AB_id'] = xba_df['game_pk'].astype(str) + "_" + xba_df['at_bat_number'].astype(str) + "_" + xba_df['pitch_number'].astype(str)

# Predict xBA
predict_mask = xba_df[['launch_speed', 'launch_angle']].notnull().all(axis=1)
xba_df['xBA'] = np.nan
xba_df.loc[predict_mask, 'xBA'] = model_xba.predict_proba(xba_df.loc[predict_mask, ['launch_speed', 'launch_angle']])[:, 1]
xba_df.loc[xba_df['events'].isin(['strikeout', 'strikeout_double_play']), 'xBA'] = 0.0
xba_df = xba_df[xba_df['xBA'].notnull()]

# Fix types and Insert
cursor.execute("DROP TABLE IF EXISTS statcast_2026")
cursor.execute("""CREATE TABLE statcast_2026 (id INT AUTO_INCREMENT PRIMARY KEY, events VARCHAR(255), hit_distance_sc FLOAT, launch_speed FLOAT, launch_angle FLOAT, game_date DATE, team_batting VARCHAR(255), team_pitching VARCHAR(255), gameid VARCHAR(255), gamepk BIGINT, player_id BIGINT, pitcher_id BIGINT, xBA FLOAT, AB_id VARCHAR(255))""")

data_xba = [tuple(None if pd.isna(x) else int(x) if isinstance(x, (np.integer, np.int64)) else x for x in row) 
            for row in xba_df[['events', 'hit_distance_sc', 'launch_speed', 'launch_angle', 'game_date', 'team_batting', 'team_pitching', 'gameid', 'game_pk', 'batter', 'pitcher', 'xBA', 'AB_id']].itertuples(index=False, name=None)]

for i in tqdm(range(0, len(data_xba), 5000), desc="Uploading xBA"):
    cursor.executemany("INSERT INTO statcast_2026 (events, hit_distance_sc, launch_speed, launch_angle, game_date, team_batting, team_pitching, gameid, gamepk, player_id, pitcher_id, xBA, AB_id) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)", data_xba[i:i+5000])
    conn.commit()

cursor.close()
conn.close()

# ==========================================
# MODULE 2: xRuns (Expected Runs)
# ==========================================
print("\n--- Starting xRuns Module ---")
conn = get_db_conn()
cursor = conn.cursor()

train_df_xruns = pd.read_sql("SELECT hit_distance_sc, launch_angle, launch_speed, outs_when_up, runners_on_base, runs_scored FROM expected_run_background WHERE hit_distance_sc IS NOT NULL AND launch_angle IS NOT NULL AND launch_speed IS NOT NULL", conn)
train_df_xruns['runners_on_base'] = train_df_xruns['runners_on_base'].astype(int)
model_xruns = XGBRegressor()
features_xruns = ['hit_distance_sc', 'launch_angle', 'launch_speed', 'outs_when_up', 'runners_on_base']
model_xruns.fit(train_df_xruns[features_xruns], train_df_xruns['runs_scored'])

xruns_df = raw_df[~raw_df['game_type'].isin(['E', 'S'])].copy()
xruns_df = xruns_df.dropna(subset=['hit_distance_sc', 'launch_angle', 'launch_speed', 'outs_when_up', 'events'])
xruns_df['runners_on_base'] = (pd.notna(xruns_df['on_1b']).astype(int).astype(str) + pd.notna(xruns_df['on_2b']).astype(int).astype(str) + pd.notna(xruns_df['on_3b']).astype(int).astype(str)).astype(int)
xruns_df['runs_scored'] = (xruns_df['post_bat_score'] - xruns_df['bat_score']).fillna(0)
xruns_df['team_batting'] = np.where(xruns_df['inning_topbot'] == 'Top', xruns_df['away_team'], xruns_df['home_team'])
xruns_df['team_pitching'] = np.where(xruns_df['inning_topbot'] == 'Top', xruns_df['home_team'], xruns_df['away_team'])
xruns_df['game_id'] = xruns_df['game_date'].astype(str) + '_' + xruns_df['home_team'] + '_' + xruns_df['away_team']
xruns_df['AB_id'] = xruns_df['game_pk'].astype(str) + "_" + xruns_df['at_bat_number'].astype(str) + "_" + xruns_df['pitch_number'].astype(str)
xruns_df['expected_runs'] = model_xruns.predict(xruns_df[features_xruns]).clip(0, 3.99)

cursor.execute("DROP TABLE IF EXISTS expected_runs_2026")
cursor.execute("""CREATE TABLE expected_runs_2026 (game_pk INT, game_id VARCHAR(50), game_date DATE, home_team VARCHAR(10), away_team VARCHAR(10), inning_topbot VARCHAR(5), team_batting VARCHAR(10), team_pitching VARCHAR(10), batter BIGINT, pitcher BIGINT, hit_distance_sc FLOAT, launch_angle FLOAT, launch_speed FLOAT, outs_when_up INT, runners_on_base INT, runs_scored INT, expected_runs FLOAT, events VARCHAR(50), description VARCHAR(100), AB_id VARCHAR(255) PRIMARY KEY, inning INT)""")

data_xruns = [tuple(None if pd.isna(x) else int(x) if isinstance(x, (np.integer, np.int64)) else x for x in row) 
              for row in xruns_df[['game_pk', 'game_id', 'game_date', 'home_team', 'away_team', 'inning_topbot', 'team_batting', 'team_pitching', 'batter', 'pitcher', 'hit_distance_sc', 'launch_angle', 'launch_speed', 'outs_when_up', 'runners_on_base', 'runs_scored', 'expected_runs', 'events', 'description', 'AB_id', 'inning']].itertuples(index=False, name=None)]

for i in tqdm(range(0, len(data_xruns), 5000), desc="Uploading xRuns"):
    cursor.executemany("INSERT INTO expected_runs_2026 (game_pk, game_id, game_date, home_team, away_team, inning_topbot, team_batting, team_pitching, batter, pitcher, hit_distance_sc, launch_angle, launch_speed, outs_when_up, runners_on_base, runs_scored, expected_runs, events, description, AB_id, inning) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)", data_xruns[i:i+5000])
    conn.commit()

cursor.close()
conn.close()

# ==========================================
# MODULE 3: Game Score
# ==========================================
print("Starting Game Score Module...")
def get_mlb_schedule(start_date, end_date):
    url = (
        f"https://statsapi.mlb.com/api/v1/schedule"
        f"?sportId=1&startDate={start_date}&endDate={end_date}"
        f"&hydrate=linescore,probablePitcher"
    )
    
    response = requests.get(url)
    data = response.json()
    games = []

    for date_data in data.get('dates', []):
        for game in date_data.get('games', []):
            if game.get('gameType') in ['E', 'S']: continue

            # Time Handling
            game_date_utc = parser.isoparse(game['gameDate']).astimezone(pytz.timezone('US/Eastern'))
            
            # Linescore data
            ls = game.get('linescore', {})
            status = game['status']['detailedState']

            # Extract Innings with Fallbacks
            if status == "Final":
                inn_disp = ls.get('scheduledInnings', 9)
                state_disp = "Final"
            elif status == "In Progress":
                inn_disp = ls.get('currentInning', 0)
                state_disp = ls.get('inningState', 'In-Progress')
            else:
                inn_disp = 0
                state_disp = "Scheduled"

            games.append({
                'game_pk': game.get('gamePk'),
                'game_date': game_date_utc.strftime('%Y-%m-%d'),
                'game_time': game_date_utc.strftime('%I:%M %p'),
                'game_status': status,
                'home_team': game['teams']['home']['team']['name'],
                'away_team': game['teams']['away']['team']['name'],
                'home_pitcher': game['teams']['home'].get('probablePitcher', {}).get('fullName', 'TBD'),
                'away_pitcher': game['teams']['away'].get('probablePitcher', {}).get('fullName', 'TBD'),
                'home_score_final': game['teams']['home'].get('score', 0),
                'away_score_final': game['teams']['away'].get('score', 0),
                'current_inning': inn_disp,
                'inning_state': state_disp
            })

    return pd.DataFrame(games)

def run_game_score_update():
    print("🚀 Fetching Game Scores...")
    start_date = "2026-02-27"
    end_date = datetime.now().strftime("%Y-%m-%d")
    
    df = get_mlb_schedule(start_date, end_date)
    
    if df.empty:
        print("⚠️ No data found.")
        return

    # 1. CLEAN DATA TYPES (Crucial for Aiven)
    # Convert all 'NaN' to None and ensure ints are ints
    df = df.replace({np.nan: None})
    
    # 2. Convert to list of tuples
    data_values = []
    for _, row in df.iterrows():
        unique_game_id = f"{row['game_pk']}_{row['game_date']}"
        data_values.append((
            unique_game_id,
            int(row['game_pk']),
            row['game_date'],
            row['game_time'],
            row['game_status'],
            row['home_team'],
            row['away_team'],
            row['home_pitcher'],
            row['away_pitcher'],
            int(row['home_score_final']) if row['home_score_final'] is not None else 0,
            int(row['away_score_final']) if row['away_score_final'] is not None else 0,
            int(row['current_inning']) if row['current_inning'] is not None else 0,
            row['inning_state']
        ))

    conn = get_db_conn()
    try:
        cursor = conn.cursor()
        
        # Use DELETE instead of TRUNCATE for better lock handling on cloud
        print("🧹 Clearing old scores...")
        cursor.execute("DELETE FROM actual_game_scores_2026")
        
        insert_query = """
            INSERT INTO actual_game_scores_2026 (
                unique_game_id, game_pk, game_date, game_time, game_status, home_team, 
                away_team, home_pitcher, away_pitcher, home_score_final, 
                away_score_final, current_inning, inning_state
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        """
        
        cursor.executemany(insert_query, data_values)
        conn.commit()
        print(f"✅ Success! {cursor.rowcount} games uploaded to Aiven.")

    except Exception as e:
        print(f"❌ Database Error: {e}")
        conn.rollback()
    finally:
        cursor.close()
        conn.close()

if __name__ == "__main__":
    run_game_score_update()
# ==========================================
# MODULE 4: Player Names
# ==========================================
print("\n--- Starting Player Name Module ---")
conn = get_db_conn()
cursor = conn.cursor()

cursor.execute("SELECT DISTINCT batter FROM expected_runs_2026 UNION SELECT DISTINCT pitcher FROM expected_runs_2026")
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
# xrunshome_etl.py

import os
#from dotenv import load_dotenv
import mysql.connector
import pandas as pd
import numpy as np
import pybaseball
from pybaseball import statcast
from xgboost import XGBClassifier, XGBRegressor
from datetime import datetime, timedelta, timezone
import warnings
import requests
from io import StringIO

#load_dotenv('/root/.env')



def get_db_conn():
    """Create a MySQL connection using environment variables"""
    return mysql.connector.connect(
        host=os.environ.get("DB_HOST"),
        user=os.environ.get("DB_USER"),
        password=os.environ.get("DB_PASSWORD"),
        port=int(os.environ.get("DB_PORT", 15560)),
        database=os.environ.get("DB_NAME"),
        ssl_disabled=False
    )

def main():
    try:
        print("Starting MLB ETL pipeline...")

        warnings.simplefilter(action='ignore', category=FutureWarning)
        warnings.filterwarnings('ignore', message='.*_OVERSIZE_WARNING.*')
        warnings.filterwarnings('ignore', category=UserWarning, module="pandas")
        warnings.filterwarnings('ignore', category=UserWarning)
        pybaseball.cache.enable()

        # -----------------------
        # 1️⃣ Download recent Statcast data
        # -----------------------
        now = datetime.now(timezone.utc)
        start_dt = '2025-01-01' #(now - timedelta(days=2)).strftime('%Y-%m-%d')
        end_dt = '2025-12-31' #now.strftime('%Y-%m-%d')
        raw_df = statcast(start_dt=start_dt, end_dt=end_dt)

        if raw_df.empty:
            print("No new Statcast data to process. Exiting.")
            return
        print(f"Downloaded {len(raw_df)} Statcast rows")

        # -----------------------
        # 2️⃣ xBA Module
        # -----------------------
        print("Starting xBA Module...")
        conn = get_db_conn()
        cursor = conn.cursor()

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

        model_xba = XGBClassifier(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            eval_metric='logloss',
            use_label_encoder=False
        )
        model_xba.fit(
            train_df_xba[['launch_speed', 'launch_angle', 'park_hit_rate']],
            train_df_xba['hit']
        )

        # Prepare prediction data
        xba_df= raw_df[~raw_df['game_type'].isin(['E'])].copy()
        xba_df = xba_df[['events', 'hit_distance_sc', 'launch_speed', 'launch_angle',
                         'game_date', 'home_team', 'away_team', 'inning_topbot', 'game_pk',
                         'batter', 'pitcher', 'at_bat_number', 'pitch_number','game_type']].copy()
        xba_df = xba_df[xba_df['events'].notna()]
        xba_df['team_batting'] = np.where(xba_df['inning_topbot'] == 'Top', xba_df['away_team'], xba_df['home_team'])
        xba_df['team_pitching'] = np.where(xba_df['inning_topbot'] == 'Top', xba_df['home_team'], xba_df['away_team'])
        xba_df['unique_game_id'] = xba_df['game_pk'].astype(str) + '_' + xba_df['game_date'].astype(str)
        xba_df['gameid'] = xba_df['game_date'].astype(str) + "_" + xba_df['home_team'] + "_" + xba_df['away_team']
        xba_df['AB_id'] = xba_df['game_pk'].astype(str) + "_" + xba_df['at_bat_number'].astype(str) + "_" + xba_df['pitch_number'].astype(str)

        # Attach Park Factors
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

        # Upload xBA to MySQL (upsert)
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS statcast_2025 (
            id INT AUTO_INCREMENT PRIMARY KEY,
            unique_game_id VARCHAR(50),
            events VARCHAR(255),
            hit_distance_sc FLOAT,
            launch_speed FLOAT,
            launch_angle FLOAT,
            game_date DATE,
            game_type VARCHAR(255),           
            team_batting VARCHAR(255),
            team_pitching VARCHAR(255),
            gameid VARCHAR(255),
            gamepk BIGINT,
            player_id BIGINT,
            pitcher_id BIGINT,
            park_hit_rate FLOAT,
            xBA FLOAT,
            AB_id VARCHAR(255) UNIQUE
        )
        """)
        data_xba = [
            tuple(None if pd.isna(x) else int(x) if isinstance(x, (np.integer, np.int64)) else x
                  for x in row)
            for row in xba_df[['unique_game_id', 'events', 'hit_distance_sc', 'launch_speed', 'launch_angle',
                               'game_date','game_type','team_batting', 'team_pitching', 'gameid','game_pk',
                               'batter', 'pitcher', 'park_hit_rate', 'xBA', 'AB_id']].itertuples(index=False, name=None)
        ]
        for i in range(0, len(data_xba), 5000):
            cursor.executemany("""
            INSERT INTO statcast_2025 (
                unique_game_id, events, hit_distance_sc, launch_speed, launch_angle,
                game_date, game_type, team_batting, team_pitching, gameid, gamepk,
                player_id, pitcher_id, park_hit_rate, xBA, AB_id
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            ON DUPLICATE KEY UPDATE
                events = VALUES(events),
                hit_distance_sc = VALUES(hit_distance_sc),
                launch_speed = VALUES(launch_speed),
                launch_angle = VALUES(launch_angle),
                game_date = VALUES(game_date),
                game_type = VALUES(game_type),               
                team_batting = VALUES(team_batting),
                team_pitching = VALUES(team_pitching),
                gameid = VALUES(gameid),
                gamepk = VALUES(gamepk),
                player_id = VALUES(player_id),
                pitcher_id = VALUES(pitcher_id),
                park_hit_rate = VALUES(park_hit_rate),
                xBA = VALUES(xBA)
            """, data_xba[i:i+5000])
            conn.commit()

        print(f"xBA Module completed successfully! {len(data_xba)} rows uploaded")

        # -----------------------
        # 3️⃣ xRuns Module
        # -----------------------
        print("Starting xRuns Module...")
        train_df_xruns = pd.read_sql("""
            SELECT hit_distance_sc, launch_angle, launch_speed,
                   outs_when_up, runners_on_base, runs_scored, venue_id
            FROM expected_run_background
            WHERE hit_distance_sc IS NOT NULL
              AND launch_angle IS NOT NULL
              AND launch_speed IS NOT NULL
        """, conn)

        train_df_xruns['runners_on_base'] = train_df_xruns['runners_on_base'].astype(int)

        park_factors = pd.read_sql("SELECT venue_id, park_run_factor FROM venue_xruns_factor", conn)
        train_df_xruns = train_df_xruns.merge(park_factors, on='venue_id', how='left')
        league_avg = park_factors['park_run_factor'].mean()
        train_df_xruns['park_run_factor'] = train_df_xruns['park_run_factor'].fillna(league_avg)

        features_xruns = [
            'hit_distance_sc',
            'launch_angle',
            'launch_speed',
            'outs_when_up',
            'runners_on_base',
            'park_run_factor'
        ]

        model_xruns = XGBRegressor(
            n_estimators=300,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            objective='reg:squarederror',
            random_state=42
        )
        model_xruns.fit(train_df_xruns[features_xruns], train_df_xruns['runs_scored'])

        xruns_df = raw_df[~raw_df['game_type'].isin(['E'])].copy()
        xruns_df = xruns_df.dropna(subset=['hit_distance_sc', 'launch_angle', 'launch_speed', 'outs_when_up', 'events', 'game_type'])

        xruns_df['runners_on_base'] = (
            pd.notna(xruns_df['on_1b']).astype(int).astype(str) +
            pd.notna(xruns_df['on_2b']).astype(int).astype(str) +
            pd.notna(xruns_df['on_3b']).astype(int).astype(str)
        ).astype(int)

        xruns_df['runs_scored'] = (xruns_df['post_bat_score'] - xruns_df['bat_score']).fillna(0)
        xruns_df['team_batting'] = np.where(xruns_df['inning_topbot'] == 'Top', xruns_df['away_team'], xruns_df['home_team'])
        xruns_df['team_pitching'] = np.where(xruns_df['inning_topbot'] == 'Top', xruns_df['home_team'], xruns_df['away_team'])
        xruns_df['AB_id'] = xruns_df['game_pk'].astype(str) + "_" + xruns_df['at_bat_number'].astype(str) + "_" + xruns_df['pitch_number'].astype(str)
        xruns_df['unique_game_id'] = xruns_df['game_pk'].astype(str) + "_" + xruns_df['game_date'].astype(str)
        xruns_df['game_id'] = xruns_df['game_date'].astype(str) + "_" + xruns_df['home_team'] + "_" + xruns_df['away_team']

        xruns_df = xruns_df.merge(pd.read_sql("SELECT DISTINCT game_pk, venue_id FROM statcast_data", conn), on='game_pk', how='left')
        xruns_df = xruns_df.merge(park_factors, on='venue_id', how='left')
        xruns_df['park_run_factor'] = xruns_df['park_run_factor'].fillna(league_avg)
        xruns_df['expected_runs_raw'] = model_xruns.predict(xruns_df[features_xruns]).clip(0, 3.99)
        scale = xruns_df['runs_scored'].mean() / xruns_df['expected_runs_raw'].mean()
        xruns_df['expected_runs'] = xruns_df['expected_runs_raw'] * scale

        # Create xRuns table
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS expected_runs_2025 (
            unique_game_id VARCHAR(50),
            game_pk INT,
            game_id VARCHAR(50),
            game_date DATE,
            game_type VARCHAR(255),           
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
            AB_id VARCHAR(255) PRIMARY KEY,
            inning INT,
            park_run_factor FLOAT
        )
        """)

        data_xruns = [
            tuple(None if pd.isna(x) else int(x) if isinstance(x, (np.integer, np.int64)) else x
                  for x in row)
            for row in xruns_df[
                ['unique_game_id','game_pk','game_id','game_date','game_type','home_team','away_team','inning_topbot',
                 'team_batting','team_pitching','batter','pitcher','hit_distance_sc','launch_angle','launch_speed',
                 'outs_when_up','runners_on_base','runs_scored','expected_runs','events','description',
                 'AB_id','inning','park_run_factor']
            ].itertuples(index=False, name=None)
        ]

        for i in range(0, len(data_xruns), 5000):
            cursor.executemany("""
                INSERT INTO expected_runs_2025 (
                    unique_game_id, game_pk, game_id, game_date, game_type,
                    home_team, away_team, inning_topbot,
                    team_batting, team_pitching, batter, pitcher,
                    hit_distance_sc, launch_angle, launch_speed,
                    outs_when_up, runners_on_base, runs_scored,
                    expected_runs, events, description,
                    AB_id, inning, park_run_factor
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s,
                          %s, %s, %s, %s, %s, %s, %s,
                          %s, %s, %s, %s, %s, %s, %s, %s)
                ON DUPLICATE KEY UPDATE
                    game_pk = VALUES(game_pk),
                    game_id = VALUES(game_id),
                    game_date = VALUES(game_date),
                    game_type = VALUES(game_type),           
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
            """, data_xruns[i:i+5000])
            conn.commit()

        print(f"xRuns Module completed successfully! {len(data_xruns)} rows uploaded")

        # -----------------------
        # 4️⃣ Game Score Module
        # -----------------------
        print("Starting Game Score Module...")
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

        start = "2025-01-01"
        end = "2025-12-31"
        url = f"https://statsapi.mlb.com/api/v1/schedule?sportId=1&startDate={start}&endDate={end}&hydrate=linescore,probablePitcher"
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        data = response.json()
        games = []
        for date_data in data.get("dates", []):
            for game in date_data.get("games", []):
                if game.get("gameType") in ["E", "A"]:
                    continue
                game_utc_str = game["gameDate"]
                status = game["status"]["detailedState"]
                ls = game.get("linescore", {})
                if status == "Final":
                    inn_disp = ls.get("scheduledInnings", 9)
                elif status == "In Progress":
                    inn_disp = ls.get("currentInning", 0)
                else:
                    inn_disp = 0
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
                    "inning_state": ls.get("inningState", "Scheduled")
                })

        df = pd.DataFrame(games)
        if not df.empty:
            df['game_time'] = pd.to_datetime(df['game_time']) - timedelta(hours=4)
            df['game_time'] = df['game_time'].dt.strftime("%I:%M %p") + " ET"
            df['game_date'] = pd.to_datetime(df['game_date']) - timedelta(hours=4)
            df['game_date'] = df['game_date'].dt.strftime("%Y-%m-%d")
            df['unique_game_id'] = df['game_pk'].astype(str) + '_' + df['game_date'].astype(str)

        if not df.empty:
            data_values = [
                (
                    row['unique_game_id'], row["game_pk"], row["game_date"], row["game_time"],
                    row["game_status"], row["home_team"], row["away_team"],
                    row["home_pitcher"], row["away_pitcher"], row["home_score_final"],
                    row["away_score_final"], row["current_inning"], row["inning_state"]
                )
                for _, row in df.iterrows()
            ]

            cursor.execute("""
            CREATE TABLE IF NOT EXISTS actual_game_scores_2026 (
                unique_game_id VARCHAR(50) PRIMARY KEY,
                game_pk INT,
                game_date DATE,
                game_time VARCHAR(20),
                game_status VARCHAR(50),
                home_team VARCHAR(10),
                away_team VARCHAR(10),
                home_pitcher VARCHAR(100),
                away_pitcher VARCHAR(100),
                home_score_final INT,
                away_score_final INT,
                current_inning INT,
                inning_state VARCHAR(20)
            )
            """)

            for i in range(0, len(data_values), 1000):
                cursor.executemany("""
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
                """, data_values[i:i+1000])
                conn.commit()

        print(f"Loaded {len(data_values)} games successfully.")

        # -----------------------
        # 5️⃣ Player Names Module
        # -----------------------
        print("Starting Player Names Module...")
        cursor.execute("SELECT DISTINCT batter FROM expected_runs_2025 UNION SELECT DISTINCT pitcher FROM expected_runs_2025")
        all_ids = [r[0] for r in cursor.fetchall() if r[0] is not None]

        dfs = []
        for char in '0123456789abcdef':
            url = f"https://raw.githubusercontent.com/chadwickbureau/register/master/data/people-{char}.csv"
            res = requests.get(url)
            if res.status_code == 200:
                dfs.append(pd.read_csv(StringIO(res.text), usecols=['key_mlbam', 'name_first', 'name_last']))

        player_register = pd.concat(dfs, ignore_index=True)
        player_lookup = player_register[player_register['key_mlbam'].isin(all_ids)].copy()
        player_lookup['full_name'] = player_lookup['name_first'] + ' ' + player_lookup['name_last']
        data_names = player_lookup[['key_mlbam', 'full_name']].drop_duplicates().values.tolist()

        cursor.execute("""
        CREATE TABLE IF NOT EXISTS player_names (
            player_id BIGINT PRIMARY KEY,
            full_name VARCHAR(100)
        )
        """)

        cursor.executemany("""
        INSERT INTO player_names (player_id, full_name)
        VALUES (%s, %s)
        ON DUPLICATE KEY UPDATE
            full_name = VALUES(full_name)
        """, data_names)
        conn.commit()

        print(f"Loaded {len(data_names)} player names successfully.")

        cursor.close()
        conn.close()
        print("✅ ALL MODULES COMPLETED SUCCESSFULLY!")

    except Exception as e:
        print(f"ETL pipeline failed: {e}")
if __name__ == "__main__":
    main()

"""
Input: List of ticks
Ouput: List of state dictionaries for each tick
"""
import snappy
import json
import numpy as np
from demoparser2 import DemoParser

def find_last_carrier_tick(target_tick, bomb_carrier_by_tick, round):
    for t in range(target_tick, -1, -1):  
        if t in bomb_carrier_by_tick:  
            if round != bomb_carrier_by_tick[t][4]:
                break
            return [bomb_carrier_by_tick[t][1], bomb_carrier_by_tick[t][2], bomb_carrier_by_tick[t][3]]
    return None

def check_steamid_consistency(json_data):
    base_ids = {}
    for tick in json_data:
        ids = [p["steamid"] for p in tick["players_info"]]
        round_id = tick["round"]
        if round_id not in base_ids:
            base_ids[round_id] = ids
        if ids != base_ids[round_id]:
            raise ValueError("steamid 顺序不一致")

def extract_states(demo_path: str, ticks: list[int]) -> list[dict]:
    parser = DemoParser(demo_path)
    states = []

    # sort ticks
    ticks = sorted(ticks)
    
    df_events = parser.parse_events(["smokegrenade_detonate", "smokegrenade_expired", "inferno_startburn", "inferno_expire", "bomb_planted"], other=["game_start_time", "total_rounds_played", "game_time"])


    round_starts = parser.parse_event("round_freeze_end")  
    round_start_ticks = round_starts["tick"].to_numpy().astype(int).tolist()
    round_start_ticks.sort()
    df = parser.parse_ticks(
        wanted_props=[
            "game_time",
            "total_rounds_played",
        ]
    )
    # deduplicate (10 players per tick)
    df = df[["tick", "game_time", "total_rounds_played"]] \
            .drop_duplicates() \
            .sort_values("tick")
    df_round_starts_time = [None for _ in range(df["total_rounds_played"].max() + 1)]
    df_round_starts_tick = [None for _ in range(df["total_rounds_played"].max() + 1)]
    for idx, round_id in enumerate(df[df['tick'].isin(round_start_ticks)]["total_rounds_played"].to_numpy()):
        df_round_starts_time[round_id] = df[df['tick'].isin(round_start_ticks)]["game_time"].to_numpy()[idx]
        df_round_starts_tick[round_id] = round_start_ticks[idx]
        # print(f"Round {round_id} starts at tick {round_start_ticks[idx]}, time {df_round_starts_time[round_id]}")

    df_round_starts_time = np.array(df_round_starts_time)

    # print("Round starts time:", df_round_starts_time)
    # print("Round start ticks:", round_start_ticks)

    df["round_start_time"] = df_round_starts_time[df["total_rounds_played"].to_numpy()]



    df_grenades = parser.parse_grenades(grenades=False)  

    # get all numes
    # print(set(df_grenades['grenade_type'].tolist()))

    # print(df_grenades)

    map_name = parser.parse_header().get("map_name", "unknown_map")

    smoke_events = []
    inferno_events = []

    bomb_events = {}
    round_result = {}

    for name, df in df_events:
        for row in df.itertuples():
            if name == "smokegrenade_detonate":
                smoke_events.append({
                    "game_time": row.game_time,
                    "round": row.total_rounds_played,
                    "position": (row.x, row.y, row.z),
                    "type": "start",
                    "entityid": row.entityid,
                    })
            elif name == "smokegrenade_expired":
                smoke_events.append({
                    "game_time": row.game_time,
                    "round": row.total_rounds_played,
                    "type": "end",
                    "entityid": row.entityid,
                    })
            elif name == "inferno_startburn":
                inferno_events.append({
                    "game_time": row.game_time,
                    "round": row.total_rounds_played,
                    "position": (row.x, row.y, row.z),
                    "type": "start",
                    "entityid": row.entityid,
                    })
            elif name == "inferno_expire":
                inferno_events.append({
                    "game_time": row.game_time,
                    "round": row.total_rounds_played,
                    "type": "end",
                    "entityid": row.entityid,
                    })
            elif name == "bomb_planted":
                now_round = row.total_rounds_played
                if df_round_starts_time[now_round] is None:
                    continue
                planted_time = row.game_time - df_round_starts_time[now_round]
                if planted_time > 0:
                    bomb_events[now_round] = planted_time

    # sort smoke and inferno events by game_time
    smoke_events = sorted(smoke_events, key=lambda x: x["game_time"])
    inferno_events = sorted(inferno_events, key=lambda x: x["game_time"])

    smoke_round_start = {}
    inferno_round_start = {}
    for idx in range(len(smoke_events)):
        round = smoke_events[idx]["round"]
        if round not in smoke_round_start:
            smoke_round_start[round] = idx
    for idx in range(len(inferno_events)):
        round = inferno_events[idx]["round"]
        if round not in inferno_round_start:
            inferno_round_start[round] = idx


    df_end = parser.parse_event("round_end")

    round_end_ticks = df_end['tick'].tolist()
    df_round_end_ticks = parser.parse_ticks(wanted_props=["health", "kills_this_round", "deaths_this_round", "assists_this_round", "damage_this_round", "team_num"], ticks=round_end_ticks)

    for index, row in df_end.iterrows():

        # print(f"End at tick {row['tick']}. Index: {index}")

        round_id = -1

        for i in range(len(df_round_starts_tick)):
            if df_round_starts_tick[i] is not None and row['tick'] > df_round_starts_tick[i]:
                round_id = i

        if round_id == -1:
            continue

        # player_df = df_round_end_ticks[df_round_end_ticks['tick'] == row['tick']]
        # assert len(player_df) == 10  # 10 players
        # player_info = []
        # for p_row in player_df.itertuples():
        #     player_info.append({
        #         "steamid": p_row.steamid,
        #         "name": p_row.name,
        #         "health": p_row.health,
        #         "kills_this_round": p_row.kills_this_round,
        #         "deaths_this_round": p_row.health == 0,
        #         "assists_this_round": p_row.assists_this_round,
        #         "damage_this_round": p_row.damage_this_round,
        #         "team_num": "CT" if p_row.team_num == 3 else "T"
        #     })

        round_info = {
            "winner": row.winner,
            "reason": row.reason
        }


        if round_id != -1:
            round_result[round_id] = {
                "round_info": round_info,
                # "player_info": player_info
            }

    all_ticks = list(range(0, max(ticks) + 1, 1))
    df_all = parser.parse_ticks(  
        wanted_props=["inventory", "X", "Y", "Z", "total_rounds_played"],  
        ticks=all_ticks  
    )
    bomb_carrier_by_tick = {}  
    for row in df_all.itertuples():  
        if isinstance(row.inventory, list) and "C4 Explosive" in row.inventory:  
            bomb_carrier_by_tick[row.tick] = (row.steamid, row.X, row.Y, row.Z, row.total_rounds_played)

    df_ticks = parser.parse_ticks(wanted_props=["game_time", "game_start_time", "total_rounds_played", "X", "Y", "Z", "weapon_name", "inventory", "inventory_as_ids", "pitch", "yaw", "is_alive", "health", "flash_duration", "flash_max_alpha", "team_num", "last_place_name", "armor", "has_helmet", "has_defuser", "is_bomb_planted", "is_bomb_dropped"], ticks=ticks)

    results = []

    death_ticks = parser.parse_event("player_death", other=["total_rounds_played", "game_time"])

    damage_ticks = parser.parse_event("player_hurt", other=["total_rounds_played", "game_time"])

    

    for tick in ticks:
        info = {}
        df_tick = df_ticks[df_ticks['tick'] == tick]
        assert len(df_tick) == 10  # 10 players
        info['round'] = df_tick.iloc[0]['total_rounds_played']
        info['tick'] = tick
        info['round_label'] = round_result.get(info['round'], {})
        info['map_name'] = map_name
        info['round_seconds'] = df_tick.iloc[0]['game_time'] - df_round_starts_time[info['round']]
        info['is_bomb_planted'] = df_tick.iloc[0]['is_bomb_planted']  
        info['is_bomb_dropped'] = df_tick.iloc[0]['is_bomb_dropped']
        if info['round'] in bomb_events:
            info['bomb_planted_time'] = bomb_events[info['round']]
        else:
            info['bomb_planted_time'] = None

        if info['is_bomb_planted'] and info['round'] in bomb_events:  
            info['bomb_planted_duration'] = info['round_seconds'] - bomb_events[info['round']]  
        else:
            info['bomb_planted_duration'] = None

        info['entity_grenades'] = []
        info['bomb_position'] = find_last_carrier_tick(tick, bomb_carrier_by_tick, info['round'])
        assert info['bomb_position'] is not None, f"Bomb carrier not found for tick {tick} in round {info['round']}"
        now_entity = df_grenades[(df_grenades['tick'] == tick)]
        for row in now_entity.itertuples():

            # skip if position is NaN
            if np.isnan(row.x) or np.isnan(row.y) or np.isnan(row.z):
                continue

            info['entity_grenades'].append({
                "name": row.name,
                "steamid": row.steamid,
                "entityid": row.grenade_entity_id,
                "type": row.grenade_type,
                "position": (row.x, row.y, row.z),
            })
        players_info = []

        for row in df_tick.itertuples():

            players_info.append({
                "steamid": row.steamid,
                "name": row.name,
                "X": row.X,
                "Y": row.Y,
                "Z": row.Z,
                "last_place_name": row.last_place_name,
                "weapon_name": row.weapon_name,
                "inventory": row.inventory,
                "inventory_as_ids": row.inventory_as_ids,
                "pitch": row.pitch,
                "yaw": row.yaw,
                "is_alive": row.is_alive,
                "health": row.health,
                "flash_duration": row.flash_duration,
                "flash_max_alpha": row.flash_max_alpha,
                "armor": row.armor,
                "has_helmet": row.has_helmet,
                "has_defuser": row.has_defuser,
                "team_num": "CT" if row.team_num == 3 else "T",
            })
        info['players_info'] = players_info
        info['projectiles'] = []
        smoke_bin = {}
        # for event in smoke_events:
        for idx in range(smoke_round_start.get(info['round'], 0), len(smoke_events)):
            event = smoke_events[idx]
            if event["game_time"] > df_tick.iloc[0]['game_time']:
                break
            
            if event["game_time"] <= df_round_starts_time[info['round']]:
                continue

            if event["round"] != info['round']:
                continue
            if event["type"] == "start":
                smoke_bin[event["entityid"]] = event
            else:
                if event["entityid"] in smoke_bin:
                    del smoke_bin[event["entityid"]]
        inferno_bin = {}
        # for event in inferno_events:
        for idx in range(inferno_round_start.get(info['round'], 0), len(inferno_events)):
            event = inferno_events[idx]
            if event["game_time"] > df_tick.iloc[0]['game_time']:
                break

            if event["game_time"] <= df_round_starts_time[info['round']]:
                continue

            if event["round"] != info['round']:
                continue
            if event["type"] == "start":
                inferno_bin[event["entityid"]] = event
            else:
                if event["entityid"] in inferno_bin:
                    del inferno_bin[event["entityid"]]
        for entityid, event in smoke_bin.items():
            info['projectiles'].append({
                "type": "smokegrenade",
                "entityid": entityid,
                "position": event["position"],
                "duration": df_tick.iloc[0]['game_time'] - event["game_time"]
            })
        for entityid, event in inferno_bin.items():
            info['projectiles'].append({
                "type": "inferno",
                "entityid": entityid,
                "position": event["position"],
                "duration": df_tick.iloc[0]['game_time'] - event["game_time"]
            })

        # for entityid, event in smoke_events.items():
        #     if "start" in event and event["start"] <= tick and "end" in event and event["end"] >= tick:
        #         info['projectiles'].append({
        #             "type": "smokegrenade",
        #             "entityid": entityid,
        #             "position": event["position"]
        #         })
        # for entityid, event in inferno_events.items():
        #     if "start" in event and event["start"] <= tick and "end" in event and event["end"] >= tick:
        #         info['projectiles'].append({
        #             "type": "inferno",
        #             "entityid": entityid,
        #             "position": event["position"]
        #         })

    

        # next_kill_info = {}
        future_deaths = death_ticks[(death_ticks['tick'] > tick) & (death_ticks['total_rounds_played'] == info['round'])]
        # if not future_deaths.empty:
        #     next_death = future_deaths.iloc[0]
        #     next_kill_info = {
        #         "attacker_name": next_death['attacker_name'],
        #         "attacker_steamid": next_death['attacker_steamid'],
        #         "assister_name": next_death['assister_name'],
        #         "assister_steamid": next_death['assister_steamid'],
        #         "victim_name": next_death['user_name'],
        #         "victim_steamid": next_death['user_steamid'],
        #         "assistedflash": next_death['assistedflash'],
        #         "attackerblind": next_death['attackerblind'],
        #         "attackerinair": next_death['attackerinair'],
        #         "dmg_health": next_death['dmg_health'],
        #         "headshot": next_death['headshot'],
        #         "thrusmoke": next_death['thrusmoke'],
        #         "weapon": next_death['weapon'],
        #         "time": next_death['game_time'] - df_round_starts_time[info['round']],
        #     }
        # info['next_kill'] = next_kill_info
        future_kill_info = []
        for _, future_death in future_deaths.iterrows():
            future_kill_info.append({
                "attacker_name": future_death['attacker_name'],
                "attacker_steamid": future_death['attacker_steamid'],
                "assister_name": future_death['assister_name'],
                "assister_steamid": future_death['assister_steamid'],
                "victim_name": future_death['user_name'],
                "victim_steamid": future_death['user_steamid'],
                "assistedflash": future_death['assistedflash'],
                "attackerblind": future_death['attackerblind'],
                "attackerinair": future_death['attackerinair'],
                "dmg_health": future_death['dmg_health'],
                "headshot": future_death['headshot'],
                "thrusmoke": future_death['thrusmoke'],
                "weapon": future_death['weapon'],
                "time": future_death['game_time'] - df_round_starts_time[info['round']],
            })
        info['future_kills'] = future_kill_info

        future_damage = damage_ticks[(damage_ticks['tick'] > tick) & (damage_ticks['total_rounds_played'] == info['round'])]
        future_damage_info = []
        for _, future_dmg in future_damage.iterrows():
            future_damage_info.append({
                "attacker_name": future_dmg['attacker_name'],
                "attacker_steamid": future_dmg['attacker_steamid'],
                "victim_name": future_dmg['user_name'],
                "victim_steamid": future_dmg['user_steamid'],
                "dmg_health": future_dmg['dmg_health'],
                "weapon": future_dmg['weapon'],
                "time": future_dmg['game_time'] - df_round_starts_time[info['round']],
            })
        info['future_damage'] = future_damage_info

        results.append(info)

    check_steamid_consistency(results)

    return results


def extract_states_by_group(demo_path: str, ticks_group: list[list[int]]) -> list[dict]:
    parser = DemoParser(demo_path)
    states = []
    
    df_events = parser.parse_events(["smokegrenade_detonate", "smokegrenade_expired", "inferno_startburn", "inferno_expire", "bomb_planted"], other=["game_start_time", "total_rounds_played", "game_time"])


    round_starts = parser.parse_event("round_freeze_end")  
    round_start_ticks = round_starts["tick"].to_numpy().astype(int).tolist()
    round_start_ticks.sort()
    df = parser.parse_ticks(
        wanted_props=[
            "game_time",
            "total_rounds_played",
        ]
    )
    # deduplicate (10 players per tick)
    df = df[["tick", "game_time", "total_rounds_played"]] \
            .drop_duplicates() \
            .sort_values("tick")
    df_round_starts_time = [None for _ in range(df["total_rounds_played"].max() + 1)]
    df_round_starts_tick = [None for _ in range(df["total_rounds_played"].max() + 1)]
    for idx, round_id in enumerate(df[df['tick'].isin(round_start_ticks)]["total_rounds_played"].to_numpy()):
        df_round_starts_time[round_id] = df[df['tick'].isin(round_start_ticks)]["game_time"].to_numpy()[idx]
        df_round_starts_tick[round_id] = round_start_ticks[idx]
        # print(f"Round {round_id} starts at tick {round_start_ticks[idx]}, time {df_round_starts_time[round_id]}")

    df_round_starts_time = np.array(df_round_starts_time)


    df["round_start_time"] = df_round_starts_time[df["total_rounds_played"].to_numpy()]



    df_grenades = parser.parse_grenades(grenades=False)  

    # get all numes
    # print(set(df_grenades['grenade_type'].tolist()))

    # print(df_grenades)

    map_name = parser.parse_header().get("map_name", "unknown_map")

    smoke_events = []
    inferno_events = []

    bomb_events = {}
    round_result = {}

    for name, df in df_events:
        for row in df.itertuples():
            if name == "smokegrenade_detonate":
                smoke_events.append({
                    "game_time": row.game_time,
                    "round": row.total_rounds_played,
                    "position": (row.x, row.y, row.z),
                    "type": "start",
                    "entityid": row.entityid,
                    })
            elif name == "smokegrenade_expired":
                smoke_events.append({
                    "game_time": row.game_time,
                    "round": row.total_rounds_played,
                    "type": "end",
                    "entityid": row.entityid,
                    })
            elif name == "inferno_startburn":
                inferno_events.append({
                    "game_time": row.game_time,
                    "round": row.total_rounds_played,
                    "position": (row.x, row.y, row.z),
                    "type": "start",
                    "entityid": row.entityid,
                    })
            elif name == "inferno_expire":
                inferno_events.append({
                    "game_time": row.game_time,
                    "round": row.total_rounds_played,
                    "type": "end",
                    "entityid": row.entityid,
                    })
            elif name == "bomb_planted":
                now_round = row.total_rounds_played
                if df_round_starts_time[now_round] is None:
                    continue
                planted_time = row.game_time - df_round_starts_time[now_round]
                if planted_time > 0:
                    bomb_events[now_round] = planted_time

    # sort smoke and inferno events by game_time
    smoke_events = sorted(smoke_events, key=lambda x: x["game_time"])
    inferno_events = sorted(inferno_events, key=lambda x: x["game_time"])

    smoke_round_start = {}
    inferno_round_start = {}
    for idx in range(len(smoke_events)):
        round = smoke_events[idx]["round"]
        if round not in smoke_round_start:
            smoke_round_start[round] = idx
    for idx in range(len(inferno_events)):
        round = inferno_events[idx]["round"]
        if round not in inferno_round_start:
            inferno_round_start[round] = idx


    df_end = parser.parse_event("round_end")

    round_end_ticks = df_end['tick'].tolist()
    df_round_end_ticks = parser.parse_ticks(wanted_props=["health", "kills_this_round", "deaths_this_round", "assists_this_round", "damage_this_round", "team_num"], ticks=round_end_ticks)

    for index, row in df_end.iterrows():

        # print(f"End at tick {row['tick']}. Index: {index}")

        round_id = -1

        for i in range(len(df_round_starts_tick)):
            if df_round_starts_tick[i] is not None and row['tick'] > df_round_starts_tick[i]:
                round_id = i

        if round_id == -1:
            continue

        # player_df = df_round_end_ticks[df_round_end_ticks['tick'] == row['tick']]
        # assert len(player_df) == 10  # 10 players
        # player_info = []
        # for p_row in player_df.itertuples():
        #     player_info.append({
        #         "steamid": p_row.steamid,
        #         "name": p_row.name,
        #         "health": p_row.health,
        #         "kills_this_round": p_row.kills_this_round,
        #         "deaths_this_round": p_row.health == 0,
        #         "assists_this_round": p_row.assists_this_round,
        #         "damage_this_round": p_row.damage_this_round,
        #         "team_num": "CT" if p_row.team_num == 3 else "T"
        #     })

        round_info = {
            "winner": row.winner,
            "reason": row.reason
        }


        if round_id != -1:
            round_result[round_id] = {
                "round_info": round_info,
                # "player_info": player_info
            }


    results_group = []

    death_ticks = parser.parse_event("player_death", other=["total_rounds_played", "game_time"])

    damage_ticks = parser.parse_event("player_hurt", other=["total_rounds_played", "game_time"])

    from tqdm import tqdm
    
    all_ticks = list(range(0, max([tick for tick_group in ticks_group for tick in tick_group]) + 1, 1))
    df_all = parser.parse_ticks(  
        wanted_props=["inventory", "X", "Y", "Z", "total_rounds_played"],  
        ticks=all_ticks  
    )
    bomb_carrier_by_tick = {}  
    for row in df_all.itertuples():  
        if isinstance(row.inventory, list) and "C4 Explosive" in row.inventory:  
            bomb_carrier_by_tick[row.tick] = (row.steamid, row.X, row.Y, row.Z, row.total_rounds_played)

    for ticks in tqdm(ticks_group, desc="Extracting states by round"):
        
        try:
            # sort ticks
            ticks = sorted(ticks)


            df_ticks = parser.parse_ticks(wanted_props=["game_time", "game_start_time", "total_rounds_played", "X", "Y", "Z", "weapon_name", "inventory", "inventory_as_ids", "pitch", "yaw", "is_alive", "health", "flash_duration", "flash_max_alpha", "team_num", "last_place_name", "armor", "has_helmet", "has_defuser", "is_bomb_planted", "is_bomb_dropped"], ticks=ticks)

            results = []

            

            for tick in ticks:
                info = {}
                df_tick = df_ticks[df_ticks['tick'] == tick]
                assert len(df_tick) == 10  # 10 players
                info['round'] = df_tick.iloc[0]['total_rounds_played']
                info['tick'] = tick
                info['round_label'] = round_result.get(info['round'], {})
                info['map_name'] = map_name
                info['round_seconds'] = df_tick.iloc[0]['game_time'] - df_round_starts_time[info['round']]
                info['is_bomb_planted'] = df_tick.iloc[0]['is_bomb_planted']  
                info['is_bomb_dropped'] = df_tick.iloc[0]['is_bomb_dropped']
                if info['round'] in bomb_events:
                    info['bomb_planted_time'] = bomb_events[info['round']]
                else:
                    info['bomb_planted_time'] = None

                if info['is_bomb_planted'] and info['round'] in bomb_events:  
                    info['bomb_planted_duration'] = info['round_seconds'] - bomb_events[info['round']]  
                else:
                    info['bomb_planted_duration'] = None

                info['entity_grenades'] = []
                info['bomb_position'] = find_last_carrier_tick(tick, bomb_carrier_by_tick, info['round'])
                assert info['bomb_position'] is not None, f"Bomb carrier not found for tick {tick} in round {info['round']}"
                now_entity = df_grenades[(df_grenades['tick'] == tick)]
                for row in now_entity.itertuples():

                    # skip if position is NaN
                    if np.isnan(row.x) or np.isnan(row.y) or np.isnan(row.z):
                        continue

                    info['entity_grenades'].append({
                        "name": row.name,
                        "steamid": row.steamid,
                        "entityid": row.grenade_entity_id,
                        "type": row.grenade_type,
                        "position": (row.x, row.y, row.z),
                    })
                players_info = []

                for row in df_tick.itertuples():

                    players_info.append({
                        "steamid": row.steamid,
                        "name": row.name,
                        "X": row.X,
                        "Y": row.Y,
                        "Z": row.Z,
                        "last_place_name": row.last_place_name,
                        "weapon_name": row.weapon_name,
                        "inventory": row.inventory,
                        "inventory_as_ids": row.inventory_as_ids,
                        "pitch": row.pitch,
                        "yaw": row.yaw,
                        "is_alive": row.is_alive,
                        "health": row.health,
                        "flash_duration": row.flash_duration,
                        "flash_max_alpha": row.flash_max_alpha,
                        "armor": row.armor,
                        "has_helmet": row.has_helmet,
                        "has_defuser": row.has_defuser,
                        "team_num": "CT" if row.team_num == 3 else "T",
                    })
                info['players_info'] = players_info
                info['projectiles'] = []
                smoke_bin = {}
                # for event in smoke_events:
                for idx in range(smoke_round_start.get(info['round'], 0), len(smoke_events)):
                    event = smoke_events[idx]
                    if event["game_time"] > df_tick.iloc[0]['game_time']:
                        break
                    
                    if event["game_time"] <= df_round_starts_time[info['round']]:
                        continue

                    if event["round"] != info['round']:
                        continue
                    if event["type"] == "start":
                        smoke_bin[event["entityid"]] = event
                    else:
                        if event["entityid"] in smoke_bin:
                            del smoke_bin[event["entityid"]]
                inferno_bin = {}
                # for event in inferno_events:
                for idx in range(inferno_round_start.get(info['round'], 0), len(inferno_events)):
                    event = inferno_events[idx]
                    if event["game_time"] > df_tick.iloc[0]['game_time']:
                        break

                    if event["game_time"] <= df_round_starts_time[info['round']]:
                        continue

                    if event["round"] != info['round']:
                        continue
                    if event["type"] == "start":
                        inferno_bin[event["entityid"]] = event
                    else:
                        if event["entityid"] in inferno_bin:
                            del inferno_bin[event["entityid"]]
                for entityid, event in smoke_bin.items():
                    info['projectiles'].append({
                        "type": "smokegrenade",
                        "entityid": entityid,
                        "position": event["position"],
                        "duration": df_tick.iloc[0]['game_time'] - event["game_time"]
                    })
                for entityid, event in inferno_bin.items():
                    info['projectiles'].append({
                        "type": "inferno",
                        "entityid": entityid,
                        "position": event["position"],
                        "duration": df_tick.iloc[0]['game_time'] - event["game_time"]
                    })

                # for entityid, event in smoke_events.items():
                #     if "start" in event and event["start"] <= tick and "end" in event and event["end"] >= tick:
                #         info['projectiles'].append({
                #             "type": "smokegrenade",
                #             "entityid": entityid,
                #             "position": event["position"]
                #         })
                # for entityid, event in inferno_events.items():
                #     if "start" in event and event["start"] <= tick and "end" in event and event["end"] >= tick:
                #         info['projectiles'].append({
                #             "type": "inferno",
                #             "entityid": entityid,
                #             "position": event["position"]
                #         })

            

                # next_kill_info = {}
                future_deaths = death_ticks[(death_ticks['tick'] > tick) & (death_ticks['total_rounds_played'] == info['round'])]
                # if not future_deaths.empty:
                #     next_death = future_deaths.iloc[0]
                #     next_kill_info = {
                #         "attacker_name": next_death['attacker_name'],
                #         "attacker_steamid": next_death['attacker_steamid'],
                #         "assister_name": next_death['assister_name'],
                #         "assister_steamid": next_death['assister_steamid'],
                #         "victim_name": next_death['user_name'],
                #         "victim_steamid": next_death['user_steamid'],
                #         "assistedflash": next_death['assistedflash'],
                #         "attackerblind": next_death['attackerblind'],
                #         "attackerinair": next_death['attackerinair'],
                #         "dmg_health": next_death['dmg_health'],
                #         "headshot": next_death['headshot'],
                #         "thrusmoke": next_death['thrusmoke'],
                #         "weapon": next_death['weapon'],
                #         "time": next_death['game_time'] - df_round_starts_time[info['round']],
                #     }
                # info['next_kill'] = next_kill_info
                future_kill_info = []
                for _, future_death in future_deaths.iterrows():
                    future_kill_info.append({
                        "attacker_name": future_death['attacker_name'],
                        "attacker_steamid": future_death['attacker_steamid'],
                        "assister_name": future_death['assister_name'],
                        "assister_steamid": future_death['assister_steamid'],
                        "victim_name": future_death['user_name'],
                        "victim_steamid": future_death['user_steamid'],
                        "assistedflash": future_death['assistedflash'],
                        "attackerblind": future_death['attackerblind'],
                        "attackerinair": future_death['attackerinair'],
                        "dmg_health": future_death['dmg_health'],
                        "headshot": future_death['headshot'],
                        "thrusmoke": future_death['thrusmoke'],
                        "weapon": future_death['weapon'],
                        "time": future_death['game_time'] - df_round_starts_time[info['round']],
                    })
                info['future_kills'] = future_kill_info

                future_damage = damage_ticks[(damage_ticks['tick'] > tick) & (damage_ticks['total_rounds_played'] == info['round'])]
                future_damage_info = []
                for _, future_dmg in future_damage.iterrows():
                    future_damage_info.append({
                        "attacker_name": future_dmg['attacker_name'],
                        "attacker_steamid": future_dmg['attacker_steamid'],
                        "victim_name": future_dmg['user_name'],
                        "victim_steamid": future_dmg['user_steamid'],
                        "dmg_health": future_dmg['dmg_health'],
                        "weapon": future_dmg['weapon'],
                        "time": future_dmg['game_time'] - df_round_starts_time[info['round']],
                    })
                info['future_damage'] = future_damage_info

                results.append(info)

            check_steamid_consistency(results)

            results_group.append(results)

        except Exception as e:
            print(f"Error processing round {info['round']}: {e}")
            results_group.append([])

    return results_group

def convert_to_python_type(obj):
    if isinstance(obj, dict):
        return {k: convert_to_python_type(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_python_type(v) for v in obj]
    elif isinstance(obj, (np.int32, np.int64)):
        return int(obj)
    elif isinstance(obj, (np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, (np.bool_)):
        return bool(obj)
    else:
        return obj
    
def save_as_json(results, path, compression=False):
    results_py = convert_to_python_type(results)
    if compression:
        compressed = snappy.compress(json.dumps(results_py, ensure_ascii=False).encode('utf-8'))
        with open(path, "wb") as f:
            f.write(compressed)
    else:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(results_py, f, indent=4, ensure_ascii=False)


if __name__ == "__main__":

    results = extract_states("demo/parivision-vs-vitality-m1-overpass.dem", [30128, 42419, 48174])


    save_as_json(results, "results.json")

    # parser = DemoParser("demo/test.dem")

    # df = parser.parse_event("player_death", other=["total_rounds_played"])


    # df.to_csv("player_death.csv")


    # df_ticks = parser.parse_ticks(wanted_props=["health", "kills_this_round", "deaths_this_round", "assists_this_round", "damage_this_round", "team_num"], ticks=[53005])


    # df_ticks.to_csv("ticks.csv")

    # player_info = parser.parse_player_info()

    # print(player_info)

    # df['seconds'] = df['tick'] / 64.0  # 每秒64个tick
    # df['game_seconds'] = df['game_start_time'] + df['seconds'] - df['round_start_time']

    # print(df)

    # df = parser.parse_ticks(  
    # ["round_start_time"],  
    # ticks=[1000, 2000, 3000]
    # )

    # print(df)  

    # print(extract_states("demo/test.dem", [1000, 2000, 3000]))
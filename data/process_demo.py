# .demo -> .json
from demoparser_utils.state_extract import extract_states, save_as_json
from demoparser2 import DemoParser
import numpy as np
import argparse
from pathlib import Path
import time

# import matplotlib.pyplot as plt

# def plot_game_time_distribution(demo_path):
#     demo_parser = DemoParser(str(demo_path))
#     df = demo_parser.parse_ticks(wanted_props=["tick", "game_time", "total_rounds_played"])
#     df = df[["tick", "game_time", "total_rounds_played"]].drop_duplicates().sort_values("tick")

#     plt.figure(figsize=(12, 6))

#     # 分局画
#     for round_id, df_round in df.groupby("total_rounds_played"):
#         ticks = df_round["tick"].to_numpy()
#         game_times = df_round["game_time"].to_numpy()
#         plt.plot(ticks, game_times, marker='o', linestyle='-', label=f"Round {round_id}")

#     plt.xlabel("Tick")
#     plt.ylabel("Game Time (seconds)")
#     plt.title("Game Time Distribution per Round")
#     plt.legend()
#     plt.grid(True)
#     plt.show()

def get_important_ticks(parser: DemoParser, interval=0.5):
    """
    Sample ticks every `interval` seconds INSIDE EACH ROUND.

    Returns:
        list[int]
    """

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

    all_ticks = []

    round_starts = parser.parse_event("round_freeze_end")  
    round_start_ticks = round_starts["tick"].to_numpy().astype(int).tolist()
    round_start_ticks.sort()
    # df_round_starts_time = df[df['tick'].isin(round_start_ticks)]["game_time"].to_numpy()
    # print("Round start ticks:", round_start_ticks)
    # print(df[df['tick'].isin(round_start_ticks)]["total_rounds_played"].to_numpy().tolist())
    # print("Round start times:", df_round_starts_time.tolist())
    df_round_starts_time = [None for _ in range(df["total_rounds_played"].max() + 1)]
    for idx, round_id in enumerate(df[df['tick'].isin(round_start_ticks)]["total_rounds_played"].to_numpy()):
        df_round_starts_time[round_id] = df[df['tick'].isin(round_start_ticks)]["game_time"].to_numpy()[idx]

    # print(df_round_starts_time)
    df_round_starts_time = np.array(df_round_starts_time)

    df["round_start_time"] = df_round_starts_time[df["total_rounds_played"].to_numpy()]


    # print unique round_start_time
    # print(f"Total rounds: {df['total_rounds_played'].nunique()}")
    # print(f"Total unique round start times: {df['round_start_time'].nunique()}")
    # print("Unique round start times:", df["round_start_time"].unique().tolist())

    # process per round
    for round_id, df_round in df.groupby("total_rounds_played"):

        times = df_round["game_time"].to_numpy()
        round_start = df_round["round_start_time"].iloc[-1]
        ticks = df_round["tick"].to_numpy()


        # round-relative seconds

        if round_start is None:
            continue

        round_seconds = times - round_start

        if len(round_seconds) == 0:
            continue

        t_end = round_seconds[-1]

        target_times = np.arange(0.5, t_end, interval)

        # print("####")
        # print(df_round)
        # print(round_seconds.tolist())
        # print(times)
        # print(round_start)

        idx = 0
        for t in target_times:
            while idx + 1 < len(round_seconds) and round_seconds[idx + 1] < t:
                idx += 1

            if idx + 1 < len(round_seconds):
                if abs(round_seconds[idx + 1] - t) < abs(round_seconds[idx] - t):
                    all_ticks.append(int(ticks[idx + 1]))
                else:
                    all_ticks.append(int(ticks[idx]))
            else:
                all_ticks.append(int(ticks[idx]))

    return sorted(set(all_ticks))

def main():
    parser = argparse.ArgumentParser(description="Process a CS demo into JSON states")
    parser.add_argument("-path", type=str, required=True, help="Path to .demo file")
    parser.add_argument("-interval", type=float, default=0.5, help="Sampling interval in seconds")
    parser.add_argument("-out", type=str, required=True, help="Output JSON file path")
    parser.add_argument("-debug", type=bool, required=False, default=0, help="Output processing information")
    parser.add_argument("-compression", type=bool, required=False, default=0, help="Use compression for JSON output")

    args = parser.parse_args()

    if args.debug:
        print(args)

    start_time = time.perf_counter()

    demo_path = Path(args.path)

    output_path = Path(args.out)
    interval = args.interval
    debug = args.debug

    if not demo_path.exists():
        print(f"Error: {demo_path} does not exist")
        return
    if debug:
        print(f"Parsing demo: {demo_path}")
    demo_parser = DemoParser(str(demo_path))
    if debug:   
        print(f"Sampling ticks every {interval} seconds per round...")
    ticks = get_important_ticks(demo_parser, interval=interval)
    if debug:
        print(f"Extracting states for {len(ticks)} ticks...")
    states = extract_states(str(demo_path), ticks)
    if debug:
        print(f"Saving states to {output_path}..., compression={args.compression}")
    save_as_json(states, str(output_path), compression=args.compression)

    end_time = time.perf_counter()  # 结束计时
    elapsed = end_time - start_time

    if debug:
        print(f"Done. The size of output file is {output_path.stat().st_size / 1024 / 1024:.2f} MB")
        print(f"Total processing time: {elapsed:.2f} seconds")
    


if __name__ == "__main__":
    main()
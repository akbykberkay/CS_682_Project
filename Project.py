"""
CS 682 Project — Tennis wOBA-style Metric
=========================================



Three steps:

  Step 1.  V(s) — probability that the server eventually wins the game,
           estimated empirically for each of the 18 non-terminal game
           states from point-by-point data.

  Step 2.  w(e) — average ΔV for each point-ending event type, where
                  ΔV =  V(s_after) - V(s_before)    if the event is
                                                    recorded by the server,
                  ΔV =  V(s_before) - V(s_after)    if recorded by the
                                                    returner.

  Step 3.  X(player) = (1/N) * Σ w(e_i) * C_i
           where C_i is the count of the player's classified points of
           event type e_i, and N is the total number of classified
           points for the player.

Data:   The Match Charting Project (Jeff Sackmann) point-by-point CSVs
        in `tennis_MatchChartingProject-master/`.  We use the 2020s
        files as the reference population.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

HERE = Path(__file__).parent
MCP_DIR = HERE / "tennis_MatchChartingProject-master"
OUT_DIR = HERE / "output"

# The 18 non-terminal game states, server's points first.
REGULAR_STATES = [
    "0-0",  "0-15",  "0-30",  "0-40",
    "15-0", "15-15", "15-30", "15-40",
    "30-0", "30-15", "30-30", "30-40",
    "40-0", "40-15", "40-30", "40-40",
    "AD-40", "40-AD",
]
STATE_SET = set(REGULAR_STATES)

# MCP shot-letter alphabet (forehand, backhand, slices, volleys, smashes, drop shots, lobs, half-volleys, trick shots, swinging volleys, etc.).
SHOT_LETTERS = set("fbrsvzopuylmhijkt")
# Serve-fault types: net, wide, deep, wide+deep
SERVE_FAULT_CHARS = set("nwdxe")
ENDING_CHARS = set("*@#")

EVENT_TYPES = [
    "ace_or_winner",   # ace or service winner — server perspective
    "double_fault",    # both serves faulted — server perspective
    "return_error",    # error on the return shot only — returner perspective
    "winner",          # rally winner (incl. return winner) — hitter perspective
    "forced_error",    # forced error drawn — drawer (winner) perspective
    "unforced_error",  # unforced error during rally — errorer perspective
]

# ---------------------------------------------------------------------------
# Shot-string parsing & event classification
# ---------------------------------------------------------------------------

def _strip_lets(s: str) -> str:
    """Strip leading 'c' characters used to mark let cords."""
    i = 0
    while i < len(s) and s[i] == "c":
        i += 1
    return s[i:]


def is_pure_serve_fault(code: str) -> bool:
    """True iff `code` describes a serve that failed (no rally shots)."""
    if not code:
        return False
    code = _strip_lets(code)
    if not code:
        return False
    if any(c in SHOT_LETTERS for c in code):
        return False
    return code[-1] in SERVE_FAULT_CHARS


def classify_event(first: str, second: str):

    if first is None or (isinstance(first, float) and first != first):
        first = ""
    if second is None or (isinstance(second, float) and second != second):
        second = ""
    first = str(first).strip()
    second = str(second).strip()

    if first in ("", "S", "R", "P", "Q"):
        return None

    f_clean = _strip_lets(first)
    s_clean = _strip_lets(second)

    if s_clean:
        if is_pure_serve_fault(s_clean):
            return ("double_fault", "server")
        rally = s_clean
    else:
        rally = f_clean

    if is_pure_serve_fault(rally):
        return None

    i = 0
    while i < len(rally) and rally[i].isdigit():
        i += 1
    while i < len(rally) and rally[i] in "+=-":
        i += 1
    after_serve = rally[i:]
    if not after_serve:
        return None

    last = after_serve[-1]
    if last not in ENDING_CHARS:
        return None

    num_rally_shots = sum(1 for c in after_serve if c in SHOT_LETTERS)
    total_shots = 1 + num_rally_shots                  # serve: rally shots
    last_by_server = (total_shots % 2 == 1)            # odd: server hit it

    if num_rally_shots == 0:
        # Pure serve outcome: ace (`*`) or unreturnable serve winner (`#`)
        if last in ("*", "#"):
            return ("ace_or_winner", "server")
        return None

    if last == "*":
        return ("winner", "server" if last_by_server else "returner")

    if last == "@":
        # Unforced error — credited (negatively) to the player who hit it
        if num_rally_shots == 1:
            return ("return_error", "returner")
        return ("unforced_error", "server" if last_by_server else "returner")

    if last == "#":
        # Forced error — credited (positively) to the *opponent* who drew it
        if num_rally_shots == 1:
            return ("return_error", "returner")
        return ("forced_error", "returner" if last_by_server else "server")

    return None


# Loading

def load_points(*paths: Path) -> pd.DataFrame:
    frames = [pd.read_csv(p, low_memory=False, dtype=str) for p in paths]
    df = pd.concat(frames, ignore_index=True)
    for col in ("Pt", "Svr", "PtWinner"):
        df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64")
    return df


def load_matches(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, low_memory=False, dtype=str)


def filter_regular_points(df: pd.DataFrame) -> pd.DataFrame:
    return df[df["Pts"].isin(STATE_SET)].copy()


# ---------------------------------------------------------------------------
# Step 1 — V(s)
# ---------------------------------------------------------------------------

def compute_V(df: pd.DataFrame):

    df = df.sort_values(["match_id", "Pt"]).reset_index(drop=True)

    grouped = df.groupby(["match_id", "Gm#"], sort=False)
    outcomes = grouped.agg(
        svr_game=("Svr", "first"),
        last_winner=("PtWinner", "last"),
    ).reset_index()
    outcomes["server_won"] = (outcomes["svr_game"] == outcomes["last_winner"]).astype(int)

    df = df.merge(outcomes[["match_id", "Gm#", "server_won"]],
                  on=["match_id", "Gm#"], how="left")

    stats = df.groupby("Pts")["server_won"].agg(["count", "sum"])
    stats["V"] = stats["sum"] / stats["count"]
    stats = stats.reindex(REGULAR_STATES)
    V_dict = stats["V"].to_dict()
    return V_dict, df, stats



# Step 2 — w(e)

def add_event_and_delta(df: pd.DataFrame, V_dict: dict) -> pd.DataFrame:
    """Add columns `event`, `perspective`, `V_before`, `V_after`, `delta_V`."""
    df = df.sort_values(["match_id", "Pt"]).reset_index(drop=True)

    df["next_state"] = df.groupby(["match_id", "Gm#"], sort=False)["Pts"].shift(-1)
    df["V_before"] = df["Pts"].map(V_dict)
    df["V_after"] = df["next_state"].map(V_dict)
    # Game ended on this point: terminal value: 1 if server won, else 0
    terminal = df["V_after"].isna()
    df.loc[terminal, "V_after"] = df.loc[terminal, "server_won"].astype(float)

    events = [classify_event(f, s) for f, s in zip(df["1st"].tolist(),
                                                     df["2nd"].tolist())]
    df["event"] = [e[0] if e else None for e in events]
    df["perspective"] = [e[1] if e else None for e in events]

    sign = np.where(df["perspective"] == "server", 1.0,
            np.where(df["perspective"] == "returner", -1.0, np.nan))
    df["delta_V"] = sign * (df["V_after"] - df["V_before"])
    return df


def compute_w(df: pd.DataFrame) -> pd.DataFrame:
    """Average ΔV (and count) per event type."""
    valid = df.dropna(subset=["event", "delta_V"])
    w = valid.groupby("event")["delta_V"].agg(["mean", "count"])
    w = w.rename(columns={"mean": "w", "count": "N"}).reindex(EVENT_TYPES)
    return w


# Step 3 — X(player)

def compute_player_X(player_name: str,
                     df: pd.DataFrame,
                     matches: pd.DataFrame,
                     w_dict: dict) -> Optional[dict]:
    in_matches = matches[
        (matches["Player 1"] == player_name) |
        (matches["Player 2"] == player_name)
    ]
    if in_matches.empty:
        return None

    p1_map = dict(zip(in_matches["match_id"], in_matches["Player 1"]))
    pnum = {mid: (1 if p1_map[mid] == player_name else 2)
            for mid in in_matches["match_id"]}

    sub = df[df["match_id"].isin(pnum)].copy()
    if sub.empty:
        return None
    sub["player_num"] = sub["match_id"].map(pnum).astype("Int64")
    sub["is_server"] = sub["player_num"] == sub["Svr"]

    valid = sub.dropna(subset=["event", "perspective"])
    owned = (
        ((valid["perspective"] == "server")   &  valid["is_server"]) |
        ((valid["perspective"] == "returner") & ~valid["is_server"])
    )
    player_pts = valid[owned]
    N = len(player_pts)
    if N == 0:
        return None
    counts = player_pts["event"].value_counts().to_dict()
    X = sum(w_dict.get(e, 0.0) * c for e, c in counts.items()) / N
    return {"player": player_name, "X": X, "N": N, "counts": counts,
            "matches": int(in_matches.shape[0])}



# Tests

def _self_test() -> None:
    """Sanity tests for the parser and end-to-end pipeline."""
    # Ace on first serve
    assert classify_event("4*", "") == ("ace_or_winner", "server")
    assert classify_event("6*", "") == ("ace_or_winner", "server")
    # Unreturnable service winner
    assert classify_event("4#", "") == ("ace_or_winner", "server")
    # Double fault: both serves out
    assert classify_event("4w", "5d") == ("double_fault", "server")
    assert classify_event("4n", "4n") == ("double_fault", "server")
    # Ace on 2nd serve (1st was wide)
    assert classify_event("4w", "5*") == ("ace_or_winner", "server")
    # Forced error on return
    assert classify_event("6f#", "") == ("return_error", "returner")
    # Unforced error on return (return shot has direction + miss)
    assert classify_event("4f3d@", "") == ("return_error", "returner")
    # Rally — server hits forehand winner after return
    # serve + return(f3) + server's forehand winner(f3*) → 3 shots, last is server
    assert classify_event("4f37f3*", "") == ("winner", "server")
    # Rally — returner hits a winner
    # serve + return(f3) + server reply(b1) + returner winner(f3*) → 4 shots, last by returner
    assert classify_event("4f37b1f3*", "") == ("winner", "returner")
    # Unforced error by returner in rally
    assert classify_event("4f37b1f3n@", "") == ("unforced_error", "returner")
    # Forced error by returner in rally (server drew it)
    assert classify_event("4f37b1f3n#", "") == ("forced_error", "server")

    tiny = pd.DataFrame([
        {"match_id": "A", "Pt": 1, "Gm#": "1", "Pts": "0-0", "Svr": 1,
         "1st": "4*", "2nd": "", "PtWinner": 1},
        {"match_id": "A", "Pt": 2, "Gm#": "1", "Pts": "15-0", "Svr": 1,
         "1st": "4f37b1f3*", "2nd": "", "PtWinner": 1},
        {"match_id": "A", "Pt": 3, "Gm#": "1", "Pts": "15-15", "Svr": 1,
         "1st": "4n", "2nd": "5w", "PtWinner": 2},   # double fault
        {"match_id": "A", "Pt": 4, "Gm#": "1", "Pts": "15-30", "Svr": 1,
         "1st": "4*", "2nd": "", "PtWinner": 1},
        {"match_id": "A", "Pt": 5, "Gm#": "1", "Pts": "30-30", "Svr": 1,
         "1st": "4*", "2nd": "", "PtWinner": 1},
        {"match_id": "A", "Pt": 6, "Gm#": "1", "Pts": "40-30", "Svr": 1,
         "1st": "4*", "2nd": "", "PtWinner": 1},
    ])
    tiny["Pt"] = tiny["Pt"].astype("Int64")
    tiny["Svr"] = tiny["Svr"].astype("Int64")
    tiny["PtWinner"] = tiny["PtWinner"].astype("Int64")

    V, tiny, _ = compute_V(tiny)
    assert (tiny["server_won"] == 1).all()
    for s in ("0-0", "15-0", "15-15", "15-30", "30-30", "40-30"):
        assert V[s] == 1.0, (s, V[s])

    tiny = add_event_and_delta(tiny, V)
    assert tiny["delta_V"].notna().all()
    w = compute_w(tiny)
    assert int(w.loc["double_fault", "N"]) == 1

    print("[self-test] OK")


# Plots

def _save_plot_V(stats: pd.DataFrame, title: str, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(11, 4.5))
    ax.bar(stats.index, stats["V"], color="#3a7bd5")
    ax.axhline(0.5, color="grey", linestyle="--", linewidth=0.8)
    ax.set_ylim(0, 1)
    ax.set_ylabel("V(s) = P(server wins game | state)")
    ax.set_xlabel("Game state (server-returner)")
    ax.set_title(title)
    for i, (state, v) in enumerate(stats["V"].items()):
        if not np.isnan(v):
            ax.text(i, v + 0.01, f"{v:.2f}", ha="center", fontsize=8)
    fig.autofmt_xdate(rotation=45)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def _save_plot_w(w: pd.DataFrame, title: str, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 4.5))
    colors = ["#1e8a3a" if v >= 0 else "#c0392b" for v in w["w"].fillna(0)]
    ax.bar(w.index, w["w"], color=colors)
    ax.axhline(0, color="black", linewidth=0.6)
    ax.set_ylabel("w(e) — average ΔV")
    ax.set_title(title)
    for i, (e, row) in enumerate(w.iterrows()):
        if not np.isnan(row["w"]):
            ax.text(i, row["w"] + (0.005 if row["w"] >= 0 else -0.012),
                    f"{row['w']:+.3f}\nN={int(row['N']):,}",
                    ha="center", fontsize=8)
    fig.autofmt_xdate(rotation=15)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def _save_plot_X(rows: list, title: str, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = sorted(rows, key=lambda r: r["X"])
    names = [r["player"] for r in rows]
    xs = [r["X"] for r in rows]
    fig, ax = plt.subplots(figsize=(9, max(3, 0.45 * len(rows) + 1)))
    colors = ["#1e8a3a" if x >= 0 else "#c0392b" for x in xs]
    ax.barh(names, xs, color=colors)
    ax.axvline(0, color="black", linewidth=0.6)
    ax.set_xlabel("X — tennis wOBA-style score")
    ax.set_title(title)
    
    lo, hi = min(xs + [0.0]), max(xs + [0.0])
    span = max(hi - lo, 1e-6)
    pad = span * 0.45
    ax.set_xlim(lo - pad, hi + pad)
    for i, r in enumerate(rows):
        offset = span * 0.01
        ax.text(r["X"] + (offset if r["X"] >= 0 else -offset), i,
                f"{r['X']:+.4f} (N={r['N']:,})",
                va="center",
                ha="left" if r["X"] >= 0 else "right",
                fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


# Pipeline

def run(tour: str = "m",
        points_files: Optional[list] = None,
        players: Optional[list] = None) -> None:
    """Run the full pipeline for a tour ('m' = ATP, 'w' = WTA)."""
    if points_files is None:
        points_files = [f"charting-{tour}-points-2020s.csv"]
    if players is None:
        players = (
            ["Novak Djokovic", "Carlos Alcaraz", "Jannik Sinner",
             "Daniil Medvedev", "Stefanos Tsitsipas",
             "Alexander Zverev", "Andrey Rublev", "Casper Ruud",
             "Hubert Hurkacz", "Rafael Nadal"]
            if tour == "m" else
            ["Iga Swiatek", "Aryna Sabalenka", "Coco Gauff",
             "Elena Rybakina", "Jessica Pegula", "Mirra Andreeva",
             "Ons Jabeur", "Maria Sakkari", "Karolina Pliskova"]
        )

    label = "ATP (men's)" if tour == "m" else "WTA (women's)"
    points_paths = [MCP_DIR / f for f in points_files]
    matches_path = MCP_DIR / f"charting-{tour}-matches.csv"

    print(f"\n=== {label} — Tennis wOBA-style metric ===\n")
    print("Loading data…")
    pts = load_points(*points_paths)
    matches = load_matches(matches_path)
    print(f"  {len(pts):,} raw points, {len(matches):,} matches loaded.")

    pts = filter_regular_points(pts)
    print(f"  {len(pts):,} regular (non-tiebreak) points after filtering.")

    print("\nStep 1 — computing V(s) for each game state…")
    V, pts, V_stats = compute_V(pts)
    print(V_stats.to_string(float_format=lambda x: f"{x:.4f}"))

    print("\nStep 2 — classifying events and computing w(e)…")
    pts = add_event_and_delta(pts, V)
    coverage = pts["event"].notna().mean()
    print(f"  Event classifier coverage: {coverage*100:.1f}% of points")
    w = compute_w(pts)
    print(w.to_string(float_format=lambda x: f"{x:.4f}"))
    w_dict = w["w"].to_dict()

    print(f"\nStep 3 — computing X for selected {label} players…")
    rows = []
    for name in players:
        res = compute_player_X(name, pts, matches, w_dict)
        if res is None:
            print(f"  {name}: not enough data — skipped.")
            continue
        rows.append(res)
        print(f"  {name:<22}  X = {res['X']:+.4f}   "
              f"(N = {res['N']:,} pts, {res['matches']} matches)")

    if not rows:
        print("\nNo players had usable data. Exiting.")
        return

    OUT_DIR.mkdir(exist_ok=True)
    V_stats.to_csv(OUT_DIR / f"V_states_{tour}.csv")
    w.to_csv(OUT_DIR / f"w_events_{tour}.csv")
    pd.DataFrame([{**{k: v for k, v in r.items() if k != "counts"},
                   **{f"count_{k}": v for k, v in r["counts"].items()}}
                  for r in rows]).to_csv(
        OUT_DIR / f"X_players_{tour}.csv", index=False)

    _save_plot_V(V_stats, f"V(s) — {label} reference population (2020s MCP)",
                 OUT_DIR / f"V_states_{tour}.png")
    _save_plot_w(w, f"w(e) — average ΔV per event — {label} (2020s MCP)",
                 OUT_DIR / f"w_events_{tour}.png")
    _save_plot_X(rows, f"X — tennis wOBA-style score — {label}",
                 OUT_DIR / f"X_players_{tour}.png")

    print(f"\nWrote tables and plots to: {OUT_DIR}/")


def main(argv: list) -> None:
    _self_test()
    tour = argv[1] if len(argv) > 1 else "m"
    if tour not in ("m", "w", "both"):
        print(f"unknown tour {tour!r}; expected one of m / w / both")
        sys.exit(2)
    if tour == "both":
        run("m"); run("w")
    else:
        run(tour)


if __name__ == "__main__":
    main(sys.argv)

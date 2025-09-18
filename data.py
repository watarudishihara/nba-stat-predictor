from __future__ import annotations
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

@dataclass
class DataConfig:
    csv_dir: str = "."
    players_csv: str = "Players.csv"
    player_stats_csv: str = "PlayerStatistics.csv"
    team_stats_csv: str = "TeamStatistics.csv"
    games_csv: str = "Games.csv"
    K: int = 10
    N_max: int = 12
    min_games_hist: int = 1

def _read_csv(path: str) -> pd.DataFrame:
    return pd.read_csv(path, low_memory=False)

def build_master_table(cfg: DataConfig) -> pd.DataFrame:
    d = cfg
    ps = _read_csv(f"{d.csv_dir}/{d.player_stats_csv}")
    ts = _read_csv(f"{d.csv_dir}/{d.team_stats_csv}")
    gm = _read_csv(f"{d.csv_dir}/{d.games_csv}")
    pl = _read_csv(f"{d.csv_dir}/{d.players_csv}")

    # Dates
    for df in (ps, ts, gm):
        if "gameDate" in df.columns:
            df["gameDate"] = pd.to_datetime(df["gameDate"], errors="coerce")

    # Coerce numeric player stat columns (handles blanks/strings)
    num_cols = [
        "numMinutes","points","assists","blocks","steals",
        "fieldGoalsAttempted","fieldGoalsMade","fieldGoalsPercentage",
        "threePointersAttempted","threePointersMade","threePointersPercentage",
        "freeThrowsAttempted","freeThrowsMade","freeThrowsPercentage",
        "reboundsDefensive","reboundsOffensive","reboundsTotal",
        "foulsPersonal","turnovers","plusMinusPoints"
    ]
    for c in num_cols:
        if c in ps.columns:
            ps[c] = pd.to_numeric(ps[c], errors="coerce").fillna(0.0)

    # home/win as ints
    for c in ["home","win"]:
        if c in ps.columns:
            ps[c] = pd.to_numeric(ps[c], errors="coerce").fillna(0).astype("int8")

    # Availability from minutes
    ps["is_active"] = (ps["numMinutes"] > 0).astype("int8")

    # Join to get teamId/opponentTeamId for player rows
    if "playerteamName" not in ps.columns:
        raise ValueError("PlayerStatistics.csv must have 'playerteamName'")
    join_df = ts[["gameId","teamId","teamName","opponentTeamId"]].drop_duplicates()
    ps = ps.merge(join_df, left_on=["gameId","playerteamName"],
                  right_on=["gameId","teamName"], how="left")
    ps.rename(columns={"teamId":"playerTeamId","opponentTeamId":"oppTeamId"}, inplace=True)
    ps.drop(columns=["teamName"], inplace=True, errors="ignore")

    # Home flag (prefer ps.home; otherwise derive via Games)
    if "home" in ps.columns:
        ps["is_home"] = ps["home"].astype("int8")
    else:
        home_map = gm.set_index("gameId")["hometeamId"].to_dict()
        ps["is_home"] = (ps["playerTeamId"] == ps["gameId"].map(home_map)).astype("int8")

    # Player bio (height/weight/position one-hots may be strings; coerce)
    keep = ["personId","height","bodyWeight","guard","forward","center",
            "birthdate","country","draftYear","draftRound","draftNumber"]
    for c in ["height","bodyWeight","guard","forward","center","draftYear","draftRound","draftNumber"]:
        if c in pl.columns:
            pl[c] = pd.to_numeric(pl[c], errors="coerce").fillna(0.0)
    for c in ["guard","forward","center"]:
        if c in pl.columns:
            pl[c] = pl[c].astype(int)
        else:
            pl[c] = 0
    ps = ps.merge(pl[[c for c in keep if c in pl.columns]],
                  on="personId", how="left")

    # Rest days per player
    ps = ps.sort_values(["personId","gameDate"])
    ps["prev_game_date"] = ps.groupby("personId")["gameDate"].shift(1)
    ps["player_rest_days"] = (ps["gameDate"] - ps["prev_game_date"]).dt.days
    ps["player_rest_days"] = ps["player_rest_days"].fillna(10).clip(lower=0).astype("float32")

    # Rolling features (shifted to avoid leakage)
    grp = ps.groupby("personId", group_keys=False)
    W = 5
    roll_src = ["numMinutes","points","assists","reboundsTotal","turnovers",
                "fieldGoalsAttempted","freeThrowsAttempted","threePointersAttempted"]
    for c in roll_src:
        ps[f"roll{W}_{c}_mean"] = grp[c].shift(1).rolling(W, min_periods=1).mean().astype("float32")
        ps[f"roll{W}_{c}_std"]  = grp[c].shift(1).rolling(W, min_periods=1).std().fillna(0).astype("float32")
    ps["usage_actions"] = ps["fieldGoalsAttempted"] + 0.44*ps["freeThrowsAttempted"] + ps["turnovers"]
    ps[f"roll{W}_usage_mean"] = grp["usage_actions"].shift(1).rolling(W, min_periods=1).mean().astype("float32")

    # Roster lists for attention (active only)
    active = ps[ps["is_active"]==1][["gameId","personId","playerTeamId","oppTeamId"]].dropna()
    active = active.dropna(subset=["playerTeamId"])
    active["playerTeamId"] = active["playerTeamId"].astype(int)

    team_rosters = (active.groupby(["gameId","playerTeamId"])["personId"]
                    .apply(list).rename("team_roster").reset_index())
    opp_rosters  = (active.groupby(["gameId","oppTeamId"])["personId"]
                    .apply(list).rename("opponent_roster").reset_index()
                    .rename(columns={"oppTeamId":"teamId"}))

    ps = ps.merge(team_rosters, left_on=["gameId","playerTeamId"],
                  right_on=["gameId","playerTeamId"], how="left")
    ps = ps.merge(opp_rosters, left_on=["gameId","oppTeamId"],
                  right_on=["gameId","teamId"], how="left")
    ps.drop(columns=["teamId"], inplace=True, errors="ignore")

    # Ensure lists and remove self from teammate list
    ps["team_roster"] = ps["team_roster"].apply(lambda x: x if isinstance(x, list) else [])
    ps["opponent_roster"] = ps["opponent_roster"].apply(lambda x: x if isinstance(x, list) else [])
    ps["teammates"] = ps.apply(lambda r: [p for p in r["team_roster"] if p != r["personId"]], axis=1)

    # Require minimal history
    ps["games_played_before"] = grp.cumcount()
    ps = ps[ps["games_played_before"] >= cfg.min_games_hist].reset_index(drop=True)

    # Clean engineered numerics
    fill_numeric = ["player_rest_days"] + [c for c in ps.columns if c.startswith("roll5_")]
    for c in fill_numeric:
        ps[c] = pd.to_numeric(ps[c], errors="coerce").fillna(0.0)

    return ps

class IdVocab:
    def __init__(self, ids: List[int]):
        uniq = sorted(set(int(x) for x in ids if pd.notna(x)))
        self.id2idx = {i:(k+1) for k,i in enumerate(uniq)}  # 0=PAD
        self.idx2id = {v:k for k,v in self.id2idx.items()}
        self.pad = 0
    def encode(self, arr: List[int], max_len: int) -> Tuple[np.ndarray, np.ndarray]:
        idxs = [self.id2idx.get(int(i), 0) for i in arr][:max_len]
        mask = [1]*len(idxs)
        if len(idxs) < max_len:
            idxs += [0]*(max_len-len(idxs))
            mask += [0]*(max_len-len(mask))
        return np.array(idxs, dtype=np.int64), np.array(mask, dtype=np.int64)

class NBADataset(Dataset):
    """One example per (gameId, personId) with pregame features, history, rosters, and targets."""
    def __init__(self, cfg: DataConfig):
        super().__init__()
        self.cfg = cfg
        self.df = build_master_table(cfg)

        # Player vocab and ordering
        self.p_vocab = IdVocab(self.df["personId"].unique().tolist())
        self.df = self.df.sort_values(["personId","gameDate"]).reset_index(drop=True)
        self.player_groups = self.df.groupby("personId")
        self.within_idx = self.player_groups.cumcount().values

        # History features
        self.hist_cols = [
            "numMinutes","points","reboundsTotal","assists",
            "fieldGoalsAttempted","freeThrowsAttempted","threePointersAttempted",
            "threePointersMade","steals","blocks","turnovers",
        ]
        self.df[self.hist_cols] = self.df[self.hist_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0)
        self.hist_dim = len(self.hist_cols)

        # Pregame features
        bio_cols = [c for c in ["height","bodyWeight","guard","forward","center"] if c in self.df.columns]
        W = 5
        roll_cols = [f"roll{W}_{c}_mean" for c in ["numMinutes","points","assists","reboundsTotal","turnovers","fieldGoalsAttempted","freeThrowsAttempted","threePointersAttempted"]] + \
                    [f"roll{W}_{c}_std"  for c in ["numMinutes","points","assists","reboundsTotal","turnovers","fieldGoalsAttempted","freeThrowsAttempted","threePointersAttempted"]] + \
                    [f"roll{W}_usage_mean"]
        other_cols = ["player_rest_days","is_home"]
        self.pregame_cols = bio_cols + roll_cols + other_cols
        # Coerce/Fill
        self.df[self.pregame_cols] = self.df[self.pregame_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0)

        # Targets
        self.df["numMinutes"] = pd.to_numeric(self.df["numMinutes"], errors="coerce").fillna(0.0)
        for c in ["points","reboundsTotal","assists"]:
            self.df[c] = pd.to_numeric(self.df[c], errors="coerce").fillna(0.0)
        self.target_minutes = self.df["numMinutes"].values.astype("float32")
        self.target_stats = self.df[["points","reboundsTotal","assists"]].values.astype("float32")

    def __len__(self): return len(self.df)

    def _get_history(self, idx: int) -> Tuple[np.ndarray, np.ndarray]:
        K = self.cfg.K
        row = self.df.iloc[idx]
        pid = row["personId"]
        order = self.within_idx[idx]
        start = max(0, order - K)
        hist = self.player_groups.get_group(pid).iloc[start:order][self.hist_cols].values.astype("float32")
        L = hist.shape[0]
        mask = np.zeros((K,), dtype=np.int64)
        out = np.zeros((K, self.hist_dim), dtype="float32")
        if L > 0:
            out[-L:, :] = hist[-K:]
            mask[-L:] = 1
        return out, mask

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        p_idx = self.p_vocab.id2idx.get(int(row["personId"]), 0)

        tm_ids, tm_mask = self.p_vocab.encode(row["teammates"], self.cfg.N_max)
        op_ids, op_mask = self.p_vocab.encode(row["opponent_roster"], self.cfg.N_max)

        x_pre = row[self.pregame_cols].values.astype("float32")
        hist_seq, hist_mask = self._get_history(idx)

        sample = {
            "player_idx": torch.tensor(p_idx, dtype=torch.long),
            "pregame": torch.tensor(x_pre, dtype=torch.float32),
            "hist_seq": torch.tensor(hist_seq, dtype=torch.float32),
            "hist_mask": torch.tensor(hist_mask, dtype=torch.long),
            "team_ids": torch.tensor(tm_ids, dtype=torch.long),
            "team_mask": torch.tensor(tm_mask, dtype=torch.long),
            "opp_ids": torch.tensor(op_ids, dtype=torch.long),
            "opp_mask": torch.tensor(op_mask, dtype=torch.long),
            "is_home": torch.tensor(int(row["is_home"]), dtype=torch.long),
            "game_id": row["gameId"],
            "team_id": int(row["playerTeamId"]) if pd.notna(row["playerTeamId"]) else -1,
            "y_minutes": torch.tensor(self.target_minutes[idx], dtype=torch.float32),
            "y_stats": torch.tensor(self.target_stats[idx], dtype=torch.float32),
        }
        return sample

def collate_batch(batch: List[dict]):
    out = {}
    for key in ["player_idx","pregame","hist_seq","hist_mask",
                "team_ids","team_mask","opp_ids","opp_mask","is_home",
                "y_minutes","y_stats"]:
        out[key] = torch.stack([b[key] for b in batch], dim=0)
    out["game_id"] = [b["game_id"] for b in batch]
    out["team_id_meta"] = [b["team_id"] for b in batch]
    return out

def make_loaders(cfg: DataConfig, batch_size=64, num_workers=0, shuffle=True):
    ds = NBADataset(cfg)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers,
                    collate_fn=collate_batch, drop_last=False)
    return ds, dl

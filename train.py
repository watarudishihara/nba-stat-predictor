
# train.py — Simple training loop with conservation penalty for minutes
from __future__ import annotations
import argparse, math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import random_split
from data import DataConfig, make_loaders, NBADataset
from model import NBAStatModel

def poisson_nll(y_true, rate):
    # y_true >= 0, rate > 0
    return rate - y_true * torch.log(rate + 1e-8) + torch.lgamma(y_true + 1)

def train(args):
    cfg = DataConfig(csv_dir=args.csv_dir, K=args.K, N_max=args.N_max, min_games_hist=1)
    ds = NBADataset(cfg)
    n = len(ds)
    n_val = max(1000, int(0.1 * n))
    n_train = n - n_val
    train_ds, val_ds = random_split(ds, [n_train, n_val], generator=torch.Generator().manual_seed(42))

    def make_loader(subset, shuffle):
        from data import collate_batch
        return torch.utils.data.DataLoader(subset, batch_size=args.batch_size, shuffle=shuffle, collate_fn=collate_batch)

    train_dl = make_loader(train_ds, True)
    val_dl   = make_loader(val_ds, False)

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")

    model = NBAStatModel(n_players=len(ds.p_vocab.id2idx),
                         pregame_dim=len(ds.pregame_cols),
                         hist_dim=ds.hist_dim,
                         emb_dim=args.emb_dim,
                         lstm_hidden=args.lstm_hidden).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    start_epoch = 0
    if args.resume_ckpt is not None:
        print(f"Loading checkpoint {args.resume_ckpt}")
        ckpt = torch.load(args.resume_ckpt, map_location=device)
        model.load_state_dict(ckpt["model_state"])
        opt.load_state_dict(ckpt["optimizer_state"])
        start_epoch = ckpt["epoch"] + 1

    for epoch in range(start_epoch, args.epochs):
        model.train()
        running = {"loss":0.0, "minutes":0.0, "points":0.0, "reb":0.0, "ast":0.0, "cons":0.0}
        for step, batch in enumerate(train_dl, 1):
            for k in ["player_idx","pregame","hist_seq","hist_mask",
                      "team_ids","team_mask","opp_ids","opp_mask","is_home",
                      "y_minutes","y_stats"]:
                batch[k] = batch[k].to(device)

            out = model(batch)

            # Minutes loss (MSE)

            # Stat losses (Poisson NLL on counts for v1: points, rebs, asts)
            loss_min = F.mse_loss(out["minutes"], batch["y_minutes"])
            y_pts = batch["y_stats"][:,0]
            y_reb = batch["y_stats"][:,1]
            y_ast = batch["y_stats"][:,2]
            loss_pts = poisson_nll(y_pts, out["points_rate"]).mean()
            loss_reb = poisson_nll(y_reb, out["reb_rate"]).mean()
            loss_ast = poisson_nll(y_ast, out["ast_rate"]).mean()

            # Conservation loss: sum minutes per (game_id, team_id) ≈ 240
            # We only have batch; approximate by grouping within-batch.
            cons = torch.tensor(0.0, device=device)
            if args.lambda_conserve > 0:
                # group indices by (game_id, team_id_meta)
                meta = {}
                for i,(gid,tid) in enumerate(zip(batch["game_id"], batch["team_id_meta"])):
                    key = (gid, tid)
                    if tid == -1:  # safety
                        continue
                    meta.setdefault(key, []).append(i)
                for ids in meta.values():
                    pred = out["minutes"][ids].sum()
                    cons = cons + (pred - 240.0).pow(2)
                if len(meta)>0:
                    cons = cons / len(meta)

            loss = loss_min + loss_pts + loss_reb + loss_ast + args.lambda_conserve * cons

            opt.zero_grad(set_to_none=True)
            loss.backward()
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            opt.step()

            running["loss"] += float(loss.item())
            running["minutes"] += float(loss_min.item())
            running["points"] += float(loss_pts.item())
            running["reb"] += float(loss_reb.item())
            running["ast"] += float(loss_ast.item())
            running["cons"] += float(cons.item())

            if step % args.log_every == 0:
                denom = args.log_every
                print(f"epoch {epoch+1} step {step}: "
                      f"loss {running['loss']/denom:.3f} | min {running['minutes']/denom:.3f} | "
                      f"pts {running['points']/denom:.3f} reb {running['reb']/denom:.3f} ast {running['ast']/denom:.3f} | "
                      f"cons {running['cons']/denom:.3f}")
                running = {k:0.0 for k in running}
            ckpt = {
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optimizer_state": opt.state_dict(),
                "loss": loss.item(),
            }
            torch.save(ckpt, f"checkpoints/epoch_{epoch}.pt")

        # quick val
        model.eval()
        with torch.no_grad():
            vloss = 0.0; vcnt = 0
            for batch in val_dl:
                for k in ["player_idx","pregame","hist_seq","hist_mask",
                          "team_ids","team_mask","opp_ids","opp_mask","is_home",
                          "y_minutes","y_stats"]:
                    batch[k] = batch[k].to(device)
                out = model(batch)
                loss_min = F.mse_loss(out["minutes"], batch["y_minutes"])
                y_pts = batch["y_stats"][:,0]
                y_reb = batch["y_stats"][:,1]
                y_ast = batch["y_stats"][:,2]
                loss_pts = (out["points_rate"] - y_pts * torch.log(out["points_rate"]+1e-8)).mean()
                loss_reb = (out["reb_rate"] - y_reb * torch.log(out["reb_rate"]+1e-8)).mean()
                loss_ast = (out["ast_rate"] - y_ast * torch.log(out["ast_rate"]+1e-8)).mean()
                loss = loss_min + loss_pts + loss_reb + loss_ast
                vloss += float(loss.item()); vcnt += 1
            print(f"[val] epoch {epoch+1}: loss {vloss/max(1,vcnt):.3f}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--lambda_min", type=float, default=1.0)
    ap.add_argument("--lambda_pts", type=float, default=1.0)
    ap.add_argument("--lambda_reb", type=float, default=1.0)
    ap.add_argument("--lambda_ast", type=float, default=1.0)
    ap.add_argument("--lambda_conserve", type=float, default=1e-4)

    ap.add_argument("--csv_dir", type=str, default=".")
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--grad_clip", type=float, default=1.0)
    ap.add_argument("--emb_dim", type=int, default=128)
    ap.add_argument("--lstm_hidden", type=int, default=128)
    ap.add_argument("--K", type=int, default=10)
    ap.add_argument("--N_max", type=int, default=12)
    ap.add_argument("--log_every", type=int, default=50)
    ap.add_argument("--cpu", action="store_true")
    ap.add_argument("--resume_ckpt", type=str, default=None,
                help="Path to checkpoint .pt file to resume training from")
    args = ap.parse_args()
    train(args)

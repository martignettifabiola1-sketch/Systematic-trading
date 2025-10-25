import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
signals_path = ROOT/"inputs"/"signals"/"signals_daily.csv"
gates_path   = ROOT/"inputs"/"gates"/"gates_daily.csv"
wt_path      = ROOT/"outputs"/"weights"/"weights_target.csv"
wf_path      = ROOT/"outputs"/"weights"/"weights_final.csv"

def main():
    sig = pd.read_csv(signals_path, parse_dates=["date"])
    gates = pd.read_csv(gates_path, parse_dates=["date"])

    last = sig["date"].max()
    s = sig[sig["date"] == last].copy()

    # pesi target giocattolo: solo segnali positivi, normalizzati
    s["pos"] = s["signal_smooth"].clip(lower=0)
    tot = s["pos"].sum()
    s["weight_target"] = s["pos"]/tot if tot > 0 else 0.0
    wt = s[["date","ticker","weight_target"]].copy()

    # applica gate (0→1)
    g = gates[gates["date"] == last]
    gate = float(g.iloc[0]["gate"]) if len(g) else 1.0
    wf = wt.copy()
    wf["weight_final"] = wf["weight_target"] * gate
    wf = wf[["date","ticker","weight_final"]]

    wt.to_csv(wt_path, index=False)
    wf.to_csv(wf_path, index=False)

if __name__ == "__main__":
    main()

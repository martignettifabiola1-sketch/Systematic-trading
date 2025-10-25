import pandas as pd
from pathlib import Path
from subprocess import check_call
import sys

ROOT = Path(__file__).resolve().parents[1]

def test_signals_columns():
    df = pd.read_csv(ROOT/"inputs/signals/signals_daily.csv")
    need = {"date","ticker","signal_3m","signal_6m","signal_12m","signal_combined","signal_smooth"}
    assert need.issubset(df.columns)

def test_gates_columns_and_bounds():
    df = pd.read_csv(ROOT/"inputs/gates/gates_daily.csv")
    need = {"date","gate","reason"}
    assert need.issubset(df.columns)
    assert (df["gate"].between(0,1)).all()

def test_stub_produces_weights():
    check_call([sys.executable, str(ROOT/"src/h2_stub.py")])
    wt = pd.read_csv(ROOT/"outputs/weights/weights_target.csv")
    wf = pd.read_csv(ROOT/"outputs/weights/weights_final.csv")
    assert set(["date","ticker","weight_target"]).issubset(wt.columns)
    assert set(["date","ticker","weight_final"]).issubset(wf.columns)
    assert (wt["weight_target"].between(0,1)).all()
    assert (wf["weight_final"].between(0,1)).all()

# factor-strategy-midcap

Defensive multi-factor strategy on US mid and small caps (DMF-US-02). It uses the same
signals, weights, trading rules, benchmark and pass checks as the S&P 500 project
(`..\factor-strategy`, DMF-US-01), but picks from US stocks ranked 501 to 1,500 by
market value. The full specification is in `docs/Strategy_Design_Document_DMF-US-02_v0.1.docx`.

The S&P 500 project is kept unchanged as a record of trials 1 and 2.

## What changed from DMF-US-01 (all decided before running)

| Setting | DMF-US-01 (S&P 500) | DMF-US-02 (this project) |
|---|---|---|
| Universe | S&P 500 members at each month-end | US common stocks ranked 501-1,500 by market value at each month-end |
| Liquidity minimum | USD 20m a day | USD 2m a day |
| Trading costs (spread + slippage) | 7 bps | 20 bps |
| Sector limits | S&P 500 sector weight +/- 5 pp | Universe sector weight +/- 5 pp |
| Trading rules | Trial 1, then trial 2 | Trial 2 rules from the start (2-month wait; trim only above limit + 1 pp) |
| Trial log | Trials 1-2 | Starts with DMF-US-01's two trials; this run is trial 3 |

## Steps

Run each line in VS Code's terminal (PowerShell) and wait for it to finish before the next.

**1. Set up the project** (creates `.venv`, the Git repository and its first commit).
Create a private, empty GitHub repository called `factor-strategy-midcap` first.

```powershell
powershell -ExecutionPolicy Bypass -File "$HOME\Documents\factor-strategy-midcap\setup.ps1"
```

**2. Copy the cleaned data and the API key from the S&P 500 project** (about 700 MB, never committed):

```powershell
cd "$HOME\Documents\factor-strategy-midcap"
New-Item -ItemType Directory -Force data\clean | Out-Null
Copy-Item -Recurse ..\factor-strategy\data\clean\sharadar, ..\factor-strategy\data\clean\fama_french, ..\factor-strategy\data\clean\benchmark data\clean\
Copy-Item ..\factor-strategy\.env .env
```

**3. Check the code:** `.\.venv\Scripts\python.exe -m pytest`

**4. Build the universe** and read `reports\universe_report.txt` (must say PASS):
`.\.venv\Scripts\python.exe -m src.data.size_universe`

**5. Sign Gates 0 and 1** in the design document (Review and sign-off table).

**6. Build the scores:** `.\.venv\Scripts\python.exe -m src.signals.build`

**7. Run the backtest once:** `.\.venv\Scripts\python.exe -m src.backtest.run`
Close `reports\backtest_insample.png` first if it is open.

## Rules

- One run. If it fails Section 10.3, stop and invest in an S&P 500 UCITS ETF; no further rethinks on this data.
- The hold-out (2016 onwards) is only used at Gate 2, once.
- Sharadar data must be deleted from both projects within 30 days of the subscription ending.
- Data, credentials and account numbers are never committed.

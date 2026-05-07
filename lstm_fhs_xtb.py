"""
Hybrydowy model LSTM + FHS (Filtered Historical Simulation) dla XTB.WA.

Wariant z lstm_var_xtb.py, ale zamiast parametrycznego t-Studenta uzywamy
EMPIRYCZNEGO kwantyla standaryzowanych reszt:

    VaR_alpha,t+1 = -sigma_pred(t+1) * Q_alpha(z)

gdzie:
  - sigma_pred(t+1) — prognoza z LSTM (rolling refit co REFIT_STEP dni)
  - z = r / sigma_pred — standaryzowane reszty na zbiorze treningowym
    (per refit; pelny in-sample, dla stabilnej estymacji 1%-kwantyla)
  - Q_alpha(z) — empiryczny lewy alpha-kwantyl reszt

Motywacja:
  1. ZWROTY AKCJI MAJA NEGATYWNY SKEW. Symetryczny t (loc=0) zaniza lewy
     ogon i moze tlumaczyc systematyczne za-male/zle-zgrupowane przekroczenia
     w poprzednim wariancie. Empiryczny kwantyl natywnie obsluguje asymetrie.
  2. Brak parametrycznego zalozenia — gladko radzi sobie z grubymi ogonami.
  3. Mniej hiperparametrow do tuningu (znikaja NU_FLOOR/NU_CAP, scale_z).

Zachowane wzgledem v3 (lstm_var_xtb.py):
  - rolling refit, target log|r_{t+1}|, L1 loss
  - gradient clipping, defensive best_state
  - reset_seed per pipeline (deterministyczne porownanie konfiguracji)
  - grid po WINDOW x LAMBDA_EWMA x SIGMA_FLOOR
"""

import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from scipy.stats import chi2
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import warnings
warnings.filterwarnings('ignore')

DEVICE = 'mps' if torch.backends.mps.is_available() else 'cpu'
print(f'Urzadzenie: {DEVICE}')


def reset_seed(seed=24):
    np.random.seed(seed)
    torch.manual_seed(seed)
    try:
        if torch.backends.mps.is_available() and hasattr(torch.mps, 'manual_seed'):
            torch.mps.manual_seed(seed)
    except Exception:
        pass


reset_seed(24)

# ------------------- Stale modelu -------------------
ALPHAS      = [0.05, 0.01]
WINDOW      = 60
SIGMA_WIN   = 30
LAMBDA_EWMA = 0.94
EPS         = 1e-8
TRAIN_FRAC  = 0.80
EPOCHS      = 200
BATCH_SIZE  = 32
LR          = 1e-3
PATIENCE    = 20
SIGMA_FLOOR = 0.25

FEATURES   = ['log_ret', 'log_vol', 'sigma_cl', 'log_r2', 'ewma_sig', 'dlog_vol']
REFIT_STEP = 90

# Ile ostatnich dni reszt uzywac do empirycznego kwantyla (None = caly trening).
# Dluzsze okno = stabilniejszy 1%-kwantyl, ale wolniej reaguje na zmiany rezimu.
RESID_WIN = None

# ------------------- 1. Pobranie danych -------------------
data = yf.download('XTB.WA', start='2018-01-01', end='2025-12-31',
                   progress=False, auto_adjust=False)
if isinstance(data.columns, pd.MultiIndex):
    data.columns = data.columns.get_level_values(0)
close = data['Close'].squeeze().dropna()
close.name = None
volume = data['Volume'].reindex(close.index).fillna(0)
volume.name = None

log_ret = np.log(close / close.shift(1)).dropna().rename('log_ret')
volume  = volume.loc[log_ret.index]


# ------------------- 2. Inzynieria cech -------------------
def build_df(lambda_ewma):
    log_vol  = np.log(volume + 1.0).rename('log_vol')
    sigma_cl = log_ret.rolling(SIGMA_WIN).std().rename('sigma_cl')
    log_r2   = np.log(log_ret.pow(2) + EPS).rename('log_r2')
    ewma_var = log_ret.pow(2).ewm(alpha=1 - lambda_ewma, adjust=False).mean()
    ewma_sig = np.sqrt(ewma_var).rename('ewma_sig')
    dlog_vol = log_vol.diff().rename('dlog_vol')
    target   = (log_ret.abs() + EPS).apply(np.log).shift(-1).rename('y')
    d = pd.concat([log_ret, log_vol, sigma_cl, log_r2, ewma_sig, dlog_vol, target],
                  axis=1).dropna()
    return d


def fit_scalers(d):
    return {
        'ret':  StandardScaler().fit(d[['log_ret']]),
        'sig':  StandardScaler().fit(d[['sigma_cl']]),
        'vol':  MinMaxScaler(feature_range=(0, 1)).fit(d[['log_vol']]),
        'r2':   StandardScaler().fit(d[['log_r2']]),
        'ewma': StandardScaler().fit(d[['ewma_sig']]),
        'dvol': StandardScaler().fit(d[['dlog_vol']]),
    }


def apply_scalers(d, sc):
    return np.column_stack([
        sc['ret'].transform(d[['log_ret']]).ravel(),
        sc['vol'].transform(d[['log_vol']]).ravel(),
        sc['sig'].transform(d[['sigma_cl']]).ravel(),
        sc['r2'].transform(d[['log_r2']]).ravel(),
        sc['ewma'].transform(d[['ewma_sig']]).ravel(),
        sc['dvol'].transform(d[['dlog_vol']]).ravel(),
    ])


# ------------------- 3. Sliding windows -------------------
def make_windows(X_flat, y_flat, w):
    X, y = [], []
    for i in range(w, len(X_flat)):
        X.append(X_flat[i - w:i])
        y.append(y_flat[i - 1])
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)


# ------------------- 4. Architektura LSTM -------------------
class LSTMVol(nn.Module):
    """LSTM(32) -> Dropout(0.3) -> Dense(1). Predykcja log(sigma_{t+1})."""
    def __init__(self, input_size=6, hidden=32, dropout=0.3):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden, batch_first=True)
        self.drop = nn.Dropout(dropout)
        self.fc   = nn.Linear(hidden, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        out    = self.drop(out[:, -1, :])
        return self.fc(out).squeeze(-1)


# ------------------- 5. Trening (jeden fit) -------------------
def train_one(df_tr_subset, window, verbose=False):
    sc = fit_scalers(df_tr_subset)
    X_flat = apply_scalers(df_tr_subset, sc)
    y_flat = df_tr_subset['y'].values
    X_tr, y_tr = make_windows(X_flat, y_flat, window)

    n_val = max(1, int(0.15 * len(X_tr)))
    X_val_t = torch.from_numpy(X_tr[-n_val:]).to(DEVICE)
    y_val_t = torch.from_numpy(y_tr[-n_val:]).to(DEVICE)
    X_tr_t  = torch.from_numpy(X_tr[:-n_val]).to(DEVICE)
    y_tr_t  = torch.from_numpy(y_tr[:-n_val]).to(DEVICE)

    loader  = DataLoader(TensorDataset(X_tr_t, y_tr_t),
                         batch_size=BATCH_SIZE, shuffle=True)

    model     = LSTMVol(input_size=len(FEATURES)).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.L1Loss()

    best_val, patience_cnt = np.inf, 0
    best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    tr_losses, val_losses = [], []

    for epoch in range(1, EPOCHS + 1):
        model.train()
        batch_loss = []
        for xb, yb in loader:
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            batch_loss.append(loss.item())
        tr_l = float(np.mean(batch_loss))

        model.eval()
        with torch.no_grad():
            v_l = criterion(model(X_val_t), y_val_t).item()

        tr_losses.append(tr_l); val_losses.append(v_l)
        if verbose and epoch % 10 == 0:
            print(f'    epoch {epoch:3d}/{EPOCHS}  train MAE={tr_l:.6f}  val MAE={v_l:.6f}')

        if v_l < best_val - 1e-7:
            best_val, patience_cnt = v_l, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            patience_cnt += 1
            if patience_cnt >= PATIENCE:
                if verbose:
                    print(f'    EarlyStopping po epoce {epoch}')
                break

    model.load_state_dict(best_state)
    return model, sc, tr_losses, val_losses


def predict_block(model, sc, df_full, start_k, end_k, window):
    block  = df_full.iloc[start_k - window:end_k]
    X_flat = apply_scalers(block, sc)
    X = np.array([X_flat[i - window:i] for i in range(window, len(X_flat))],
                 dtype=np.float32)
    model.eval()
    with torch.no_grad():
        ls = model(torch.from_numpy(X).to(DEVICE)).cpu().numpy()
    return np.clip(np.exp(ls), 1e-6, None)


def predict_train(model, sc, df_full, end_k, window):
    block  = df_full.iloc[:end_k]
    X_flat = apply_scalers(block, sc)
    X = np.array([X_flat[i - window:i] for i in range(window, len(X_flat))],
                 dtype=np.float32)
    model.eval()
    with torch.no_grad():
        ls = model(torch.from_numpy(X).to(DEVICE)).cpu().numpy()
    return np.clip(np.exp(ls), 1e-6, None)


# ------------------- 6. Rolling refit + empiryczny kwantyl reszt -------------------
def run_pipeline(window, lambda_ewma, verbose=True, seed=24):
    """Rolling refit LSTM. Per refit zapisuje pelne standaryzowane reszty
    z = r/sigma_LSTM (FHS), zeby kwantyle dawalo sie liczyc post-hoc dla
    dowolnego RESID_WIN bez ponownego treningu modelu."""
    reset_seed(seed)
    df_loc  = build_df(lambda_ewma)
    n_train = int(TRAIN_FRAC * len(df_loc))

    sigma_chunks    = []
    z_resid_blocks  = []   # pelne reszty per refit (uciecie do RESID_WIN robimy post-hoc)
    block_lens      = []   # ile dni testowych obejmuje dany refit
    train_losses, val_losses = [], []

    cur_end  = n_train
    refit_no = 0
    while cur_end < len(df_loc):
        block_end  = min(cur_end + REFIT_STEP, len(df_loc))
        refit_no  += 1
        if verbose:
            print(f'  Refit #{refit_no:2d}: train -> {df_loc.index[cur_end-1].date()} '
                  f'({cur_end} obs)   |   pred {df_loc.index[cur_end].date()}..'
                  f'{df_loc.index[block_end-1].date()} ({block_end - cur_end} dni)')

        model_k, sc_k, tr_l, v_l = train_one(df_loc.iloc[:cur_end], window,
                                             verbose=False)
        if refit_no == 1:
            train_losses, val_losses = tr_l, v_l

        sigma_in = predict_train(model_k, sc_k, df_loc, cur_end, window)
        r_in     = df_loc['log_ret'].values[window:cur_end]
        z_resid  = r_in / np.maximum(sigma_in, 1e-8)
        z_resid  = z_resid[np.isfinite(z_resid)]
        z_resid_blocks.append(z_resid)

        sig_b = predict_block(model_k, sc_k, df_loc, cur_end, block_end, window)
        sigma_chunks.append(sig_b)
        block_lens.append(len(sig_b))

        cur_end = block_end

    return {
        'df':              df_loc,
        'n_train':         n_train,
        'sigma_pred_raw':  np.concatenate(sigma_chunks),
        'z_resid_blocks':  z_resid_blocks,
        'block_lens':      np.array(block_lens),
        'ewma_te':         df_loc['ewma_sig'].values[n_train:],
        'r_actual':        df_loc['log_ret'].values[n_train:],
        'dates_te':        df_loc.index[n_train:],
        'train_losses':    train_losses,
        'val_losses':      val_losses,
    }


def quantiles_for(pipe, resid_win):
    """Liczy empiryczne kwantyle alpha post-hoc.
    `resid_win` moze byc:
      - None / int  -> jedno okno wspolne dla wszystkich alpha
      - dict {alpha: window} -> osobne okno per poziom (np. krotkie dla 5%,
        dlugie dla 1%, gdzie rzadkie obserwacje wymagaja stabilnosci)."""
    if not isinstance(resid_win, dict):
        resid_win = {a: resid_win for a in ALPHAS}

    q_arr        = {a: [] for a in ALPHAS}
    q_per_refit  = {a: [] for a in ALPHAS}
    n_resid_used = {a: [] for a in ALPHAS}

    for z_full, blen in zip(pipe['z_resid_blocks'], pipe['block_lens']):
        for a in ALPHAS:
            w = resid_win[a]
            z = z_full if (w is None or len(z_full) <= w) else z_full[-w:]
            q = float(np.quantile(z, a))
            q_arr[a].append(np.full(blen, q))
            q_per_refit[a].append(q)
            n_resid_used[a].append(len(z))

    return {
        'q_arr':       {a: np.concatenate(q_arr[a])    for a in ALPHAS},
        'q_per_refit': {a: np.array(q_per_refit[a])    for a in ALPHAS},
        'n_resid':     {a: np.array(n_resid_used[a])   for a in ALPHAS},
    }


_pipeline_cache = {}
def get_pipeline(window, lambda_ewma, verbose=False):
    key = (int(window), round(float(lambda_ewma), 4))
    if key not in _pipeline_cache:
        print(f'\n>>> Pipeline LSTM-FHS: WINDOW={window}, LAMBDA_EWMA={lambda_ewma}')
        _pipeline_cache[key] = run_pipeline(window, lambda_ewma, verbose=verbose)
    return _pipeline_cache[key]


# ------------------- 7. Backtesty (Kupiec POF + Christoffersen) -------------------
def kupiec_pof(viol, alpha):
    x, n = int(viol.sum()), len(viol)
    if x == 0 or x == n:
        return np.nan, np.nan, x
    p_hat = x / n
    lr = -2 * (np.log((1 - alpha)**(n - x) * alpha**x)
               - np.log((1 - p_hat)**(n - x) * p_hat**x))
    return lr, 1 - chi2.cdf(lr, 1), x


def christoffersen_ind(viol):
    """LR test niezaleznosci. Obsluguje przypadki brzegowe (np. n11=0 przy
    rozproszonych naruszeniach — to wlasnie sygnatura niezaleznosci, nie
    powod, zeby zwracac NaN). Konwencja 0*log(0)=0."""
    v = viol.astype(int)
    n00 = n01 = n10 = n11 = 0
    for i in range(1, len(v)):
        if v[i-1] == 0 and v[i] == 0: n00 += 1
        if v[i-1] == 0 and v[i] == 1: n01 += 1
        if v[i-1] == 1 and v[i] == 0: n10 += 1
        if v[i-1] == 1 and v[i] == 1: n11 += 1

    n_viol = n01 + n11
    n_total = n00 + n01 + n10 + n11
    if n_viol == 0 or n_total == 0:
        return np.nan, np.nan

    pi01 = n01 / (n00 + n01) if (n00 + n01) > 0 else 0.0
    pi11 = n11 / (n10 + n11) if (n10 + n11) > 0 else 0.0
    pi_  = n_viol / n_total

    def xlogy(x, p):
        # 0 * log(0) = 0 (konwencja); x>0 i p=0 -> -inf (sygnal anomalii)
        if x == 0:
            return 0.0
        if p <= 0.0:
            return -np.inf
        return x * np.log(p)

    log_l0 = xlogy(n00 + n10, 1 - pi_) + xlogy(n_viol, pi_)
    log_l1 = (xlogy(n00, 1 - pi01) + xlogy(n01, pi01)
              + xlogy(n10, 1 - pi11) + xlogy(n11, pi11))

    if not np.isfinite(log_l0) or not np.isfinite(log_l1):
        return np.nan, np.nan

    lr = -2.0 * (log_l0 - log_l1)
    lr = max(lr, 0.0)  # LR moze byc nieznacznie ujemne przez bledy numeryczne
    return lr, 1 - chi2.cdf(lr, 1)


def compute_var_and_tests(pipe, sigma_floor, resid_win=None):
    """VaR_alpha = -sigma_used * Q_alpha(z), gdzie Q_alpha pochodzi z FHS.
    `resid_win` jak w quantiles_for: skalar, None, lub dict per alpha."""
    qd = quantiles_for(pipe, resid_win)
    sigma_used = np.maximum(pipe['sigma_pred_raw'], sigma_floor * pipe['ewma_te'])
    out = {'sigma': sigma_used, 'vars': {}, 'tests': {},
           'q_arr': qd['q_arr'], 'q_per_refit': qd['q_per_refit']}
    for a in ALPHAS:
        q_a   = qd['q_arr'][a]                          # juz lewe (ujemne) kwantyle
        var_a = -(sigma_used * q_a)
        viol  = (pipe['r_actual'] < -var_a).astype(int)
        _, p_p, n_v = kupiec_pof(viol, a)
        _, p_i      = christoffersen_ind(viol)
        out['vars'][a]  = var_a
        out['tests'][a] = {
            'n_viol': n_v,
            'freq':   n_v / len(viol),
            'p_kup':  p_p,
            'p_chr':  p_i,
            'k_ok':   (not np.isnan(p_p)) and (p_p > 0.05),
            'c_ok':   (not np.isnan(p_i)) and (p_i > 0.05),
        }
    return out


# ------------------- 8. Grid search: WINDOW x LAMBDA_EWMA x SIGMA_FLOOR x RESID_WIN -------------------
WINDOW_GRID      = [30, 50, 80]
LAMBDA_EWMA_GRID = [0.94, 0.97]
SIGMA_FLOOR_GRID = [0.0, 0.3, 0.5, 0.7, 0.9]

# Per-alpha okno empirycznych reszt (None = caly trening).
# Dla 5% krotsze okno tnie wplyw skrajnych obs sprzed lat -> q95 mniej skrajne
# -> wiecej naruszen, bardziej w okolicy 5%.
# Dla 1% potrzeba duzo punktow, zeby kwantyl byl stabilny.
RESID_WIN_GRID = [
    # (RW95, RW99)
    (None, None),
    (250,  None),
    (500,  None),
    (750,  None),
    (1000, None),
    (250,  1500),
    (500,  1500),
]

n_lstm_runs = len(WINDOW_GRID) * len(LAMBDA_EWMA_GRID)
print('\n====== Grid search FHS: WINDOW x LAMBDA_EWMA x SIGMA_FLOOR x RESID_WIN ======')
print(f'(LSTM trenowany dla kazdej pary (WINDOW, LAMBDA_EWMA) — {n_lstm_runs} razy. '
      f'SIGMA_FLOOR i RESID_WIN aplikowane post-hoc.)\n')


def _fmt_rw(w):
    return 'all' if w is None else str(w)


grid_rows = []
for win in WINDOW_GRID:
    for lam in LAMBDA_EWMA_GRID:
        pipe = get_pipeline(win, lam, verbose=True)
        n_obs = len(pipe['r_actual'])
        exp95 = int(round(0.05 * n_obs))
        exp99 = int(round(0.01 * n_obs))
        print(f'  N test = {n_obs}, oczekiwane przekroczenia: 95%={exp95}, 99%={exp99}')
        for sf in SIGMA_FLOOR_GRID:
            for rw95, rw99 in RESID_WIN_GRID:
                rw_dict = {0.05: rw95, 0.01: rw99}
                res = compute_var_and_tests(pipe, sf, resid_win=rw_dict)
                t95 = res['tests'][0.05]
                t99 = res['tests'][0.01]
                fr95 = t95['freq']
                fr99 = t99['freq']
                freq_dev = abs(fr95 - 0.05) / 0.05 + abs(fr99 - 0.01) / 0.01
                q95_arr = res['q_per_refit'][0.05]
                q99_arr = res['q_per_refit'][0.01]
                grid_rows.append({
                    'WIN':     win,
                    'LAMBDA':  lam,
                    'SIG_FLR': sf,
                    'RW95':    _fmt_rw(rw95),
                    'RW99':    _fmt_rw(rw99),
                    'q95_avg': f'{q95_arr.mean():.3f}',
                    'q99_avg': f'{q99_arr.mean():.3f}',
                    'n95':     t95['n_viol'],
                    'fr95':    f'{fr95:.2%}',
                    'Kup95':   f'{t95["p_kup"]:.3f}' if not np.isnan(t95['p_kup']) else '-',
                    'Chr95':   f'{t95["p_chr"]:.3f}' if not np.isnan(t95['p_chr']) else '-',
                    'OK95':    'TAK' if (t95['k_ok'] and t95['c_ok']) else 'NIE',
                    'n99':     t99['n_viol'],
                    'fr99':    f'{fr99:.2%}',
                    'Kup99':   f'{t99["p_kup"]:.3f}' if not np.isnan(t99['p_kup']) else '-',
                    'Chr99':   f'{t99["p_chr"]:.3f}' if not np.isnan(t99['p_chr']) else '-',
                    'OK99':    'TAK' if (t99['k_ok'] and t99['c_ok']) else 'NIE',
                    'freq_dev': f'{freq_dev:.3f}',
                    '_score':  (
                        (0.0 if np.isnan(t95['p_kup']) else min(t95['p_kup'], 1.0))
                        + (0.0 if np.isnan(t99['p_kup']) else min(t99['p_kup'], 1.0))
                        + (0.0 if np.isnan(t95['p_chr']) else min(t95['p_chr'], 1.0))
                        + (0.0 if np.isnan(t99['p_chr']) else min(t99['p_chr'], 1.0))
                    ),
                    '_freq_dev': freq_dev,
                    '_rw95':   rw95,
                    '_rw99':   rw99,
                })

df_grid = pd.DataFrame(grid_rows)
_aux_cols = ['_score', '_freq_dev', '_rw95', '_rw99']

df_show = df_grid.sort_values('_score', ascending=False).drop(columns=_aux_cols)
print('\n--- Pelna tabela (sortowana wg sumy p-wartosci, malejaco) — top 30 ---')
print(df_show.head(30).to_string(index=False))

df_pass = df_show[(df_show['OK95'] == 'TAK') & (df_show['OK99'] == 'TAK')]
print('\n--- Tylko konfiguracje OK95 = TAK i OK99 = TAK ---')
if len(df_pass) == 0:
    print('  (brak konfiguracji przechodzacej oba testy na obu poziomach)')
else:
    print(df_pass.to_string(index=False))

df_freq = df_grid.sort_values('_freq_dev', ascending=True).drop(columns=_aux_cols).head(10)
print('\n--- Top 10 po freq_dev (najmniejsze odchylenie od 5%/1%) ---')
print(df_freq.to_string(index=False))

# Najlepsza konfiguracja: priorytet — przechodzenie obu testow na obu poziomach
# (OK95 i OK99 = TAK), nastepnie score po p-wartosciach, tie-break po freq_dev.
# Bez priorytetu OK byloby ryzyko, ze model z marginalnym Kupcem dla 95% wygra.
df_grid['_pass_both'] = ((df_grid['OK95'] == 'TAK') & (df_grid['OK99'] == 'TAK')).astype(int)
best = df_grid.sort_values(['_pass_both', '_score', '_freq_dev'],
                           ascending=[False, False, True]).iloc[0]
def _to_rw(v):
    # pandas zamienia None w kolumnie mieszanej int/None na NaN
    return None if v is None or pd.isna(v) else int(v)

BEST_WIN  = int(best['WIN'])
BEST_LAM  = float(best['LAMBDA'])
BEST_SF   = float(best['SIG_FLR'])
BEST_RW95 = _to_rw(best['_rw95'])
BEST_RW99 = _to_rw(best['_rw99'])
BEST_RW   = {0.05: BEST_RW95, 0.01: BEST_RW99}
print(f'\nNajlepsza konfiguracja (pass_both={int(best["_pass_both"])}, '
      f'score = {best["_score"]:.3f}, freq_dev = {best["_freq_dev"]:.3f}): '
      f'WINDOW={BEST_WIN}, LAMBDA_EWMA={BEST_LAM}, SIGMA_FLOOR={BEST_SF}, '
      f'RW95={_fmt_rw(BEST_RW95)}, RW99={_fmt_rw(BEST_RW99)}')

best_pipe       = get_pipeline(BEST_WIN, BEST_LAM)
best_res        = compute_var_and_tests(best_pipe, BEST_SF, resid_win=BEST_RW)
sigma_pred      = best_res['sigma']
vars_pred       = best_res['vars']
dates_te        = best_pipe['dates_te']
r_actual        = best_pipe['r_actual']
ewma_te         = best_pipe['ewma_te']
sigma_pred_raw  = best_pipe['sigma_pred_raw']
df_best         = best_pipe['df']
n_train         = best_pipe['n_train']
train_losses    = best_pipe['train_losses']
val_losses      = best_pipe['val_losses']

# ------------------- 9. Wykresy -------------------
fig, axes = plt.subplots(3, 1, figsize=(14, 11), sharex=False)

axes[0].plot(train_losses, label='train MAE')
axes[0].plot(val_losses,   label='val MAE')
axes[0].set_title(f'Krzywa uczenia (MAE) — WIN={BEST_WIN}, LAMBDA={BEST_LAM}')
axes[0].legend(); axes[0].grid(alpha=0.3)

axes[1].plot(dates_te, sigma_pred,     color='steelblue', lw=1,
             label=f'sigma LSTM (floor={BEST_SF})')
axes[1].plot(dates_te, sigma_pred_raw, color='steelblue', lw=0.6, alpha=0.4,
             label='sigma LSTM (raw)')
axes[1].plot(dates_te, ewma_te,        color='gray', lw=0.8, alpha=0.7,
             label=f'EWMA sigma (lambda={BEST_LAM})')
for k in range(n_train + REFIT_STEP, len(df_best), REFIT_STEP):
    axes[1].axvline(df_best.index[k], color='black', ls=':', lw=0.5, alpha=0.4)
axes[1].set_title('Prognoza zmiennosci $\\sigma_{t+1}$ (rolling refit)')
axes[1].legend(); axes[1].grid(alpha=0.3)

axes[2].plot(dates_te, r_actual, color='steelblue', lw=0.5, alpha=0.7, label='$R_t$')
for a, c in zip(ALPHAS, ['orange', 'red']):
    var_a = vars_pred[a]
    viol  = r_actual < -var_a
    axes[2].plot(dates_te, -var_a, color=c, lw=1, label=f'-VaR {int((1-a)*100)}% (FHS)')
    axes[2].scatter(dates_te[viol], r_actual[viol], color=c, s=18, zorder=5)
axes[2].set_title(f'LSTM-FHS VaR best config '
                  f'(WIN={BEST_WIN}, LAM={BEST_LAM}, SF={BEST_SF}, '
                  f'RW95={_fmt_rw(BEST_RW95)}, RW99={_fmt_rw(BEST_RW99)}) vs zwroty')
axes[2].legend(); axes[2].grid(alpha=0.3)

plt.tight_layout()
print('\nWykres gotowy.')
plt.show()

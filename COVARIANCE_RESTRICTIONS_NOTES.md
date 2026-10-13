# PyBLP Covariance Restrictions (MacKay–Miller 2025) 実装メモ

> covariance restriction（共分散制約 `E[ξω·Z_C] = 0`）で**価格係数を推定**するときの、PyBLP特有のハマりどころと正しい設定手順。別環境での実装用リファレンス。

---

## TL;DR

- **価格係数（alpha）は covariance restriction で推定できる。** マニュアルの efficiency 警告と矛盾しない。
- PyBLP の言う **「linear parameters」= concentrate out（消去）される係数**であって、価格係数のことではない。
- 価格係数を推定に乗せるには、`solve()` の **`beta` 引数で価格係数の位置に有限の初期値を与える**（→ `theta` 入り）。NaN のままにすると消去対象になり、供給側があるとそもそもエラーになる。
- covariance restriction は `xi*omega` を使うので **供給側（X3 と firm_ids）が必須**。

---

## 一番のハマりどころ：「linear / nonlinear」の用語が2つの意味を持つ

| 文脈 | 「linear」の意味 |
|---|---|
| **モデル構造**（効用が線形） | X1 に入る係数（beta）。価格係数もここ。 |
| **PyBLP の推定実装** | concentrate out（消去）される係数。Z_D / Z_S だけで線形IVで推定され、covariance moment は効かない。 |

マニュアルの

> In the current implementation, these covariance restrictions only affect the nonlinear parameters. The linear parameters are estimated using other moments. In the case of overidentification, the estimator may not be fully efficient...

の **「linear parameters」は後者（消去される係数）**を指す。価格係数は X1（モデル上は線形）にありながら、推定上は **`theta`（PyBLP の言う "nonlinear parameters"）に置かれる**ため、covariance moment が効く。→ **MacKay–Miller の使い方そのもの。**

---

## なぜそうなるか（PyBLP のパラメタ分類の仕組み）

beta の各要素が「消去」か「theta入り」かは、**初期値が NaN かどうか**だけで決まる。

- `eliminated_beta_index = np.isnan(beta)` … NaN なら消去対象
- 消去対象 → 線形IV回帰（Z_D / Z_S のみ）で推定
- 初期値あり → `theta` に入り、非線形最適化で推定（**ここに covariance moment が効く**）

さらに **供給側がある場合、価格係数の消去は禁止**されており、NaN だと次のエラーになる:

```
A supply side was specified, so alpha should not be concentrated out.
That is, initial values should be specified for all parameters in beta on
X1 characteristics involving prices.
```

そして covariance moment は `E[ξ·ω·Z_C]`、つまり供給側の構造誤差 `ω` を使う → **供給側が必須**。

結果として、**covariance restriction を使う場面では価格係数は必ず `theta` に入る**。

### 実装の流れ（内部）

1. Z_D / Z_S だけの重み行列で線形IV回帰 → 消去対象の係数（外生 X1 の beta、供給側 gamma）を推定
2. その**後で** covariance moment `xi*omega - m` と `Z_C` を moment 群に追加
3. `theta`（価格係数を含む）を、需要・供給・covariance を**すべて含む** GMM 目的関数で最適化
   - covariance moment の勾配 = `xi_jacobian*omega + xi*omega_jacobian`。価格係数が theta にあると `xi_jacobian` に価格係数列が立つので、勾配は非ゼロ → 最適化器が covariance moment で価格係数を動かす。

---

## 実装チェックリスト

- [ ] X1 の `Formulation` に `prices` を含める
- [ ] **供給側 X3 の `Formulation` を指定**（限界費用シフター）。`product_data` に `firm_ids`
- [ ] `product_data` に `covariance_instruments`（= Z_C）を追加
- [ ] `solve()` の **`beta` 引数で、価格係数の位置に有限の初期値**を与える（他の外生係数は NaN で消去してよい）
- [ ] （任意）符号担保のため `beta_bounds` で価格係数に上限 0、または `rc_type='log'` で対数正規係数
- [ ] demand/supply の moment type はデフォルト（`'levels'`）のまま（非デフォルト型は covariance restriction 非対応）

---

## 最小コード例

```python
import numpy as np
import pyblp

# --- product_formulations = (X1, X2, X3) ---
product_formulations = (
    pyblp.Formulation('1 + prices + x'),   # X1: 線形特性（価格を含む）
    pyblp.Formulation('0 + prices'),       # X2: ランダム係数（任意。無ければ X2 は省略可）
    pyblp.Formulation('1 + w'),            # X3: 限界費用シフター（供給側 → 必須）
)

# product_data に必要な列:
#   market_ids, shares, prices, firm_ids,
#   demand_instruments*, supply_instruments*,
#   covariance_instruments*   ← これが Z_C
#
# Z_C はシンプルには定数1列（無条件の E[ξω]=0）。
# 追加の需要/費用シフターを入れて過剰識別にもできる。

problem = pyblp.Problem(
    product_formulations,
    product_data,
    agent_formulation=pyblp.Formulation('0 + income'),  # 任意
    agent_data=agent_data,                              # 任意
)

# X1 = [1, prices, x] の並び。価格係数(prices, index=1)だけ有限の初期値 → theta 入り。
# 残り(intercept, x)は NaN → concentrate out。
results = problem.solve(
    sigma=0.5,                      # ランダム係数を使う場合（任意）
    beta=[np.nan, -1.0, np.nan],    # ← 価格係数に初期値 -1.0（消去させない）
    # 符号担保（任意）: 価格係数の上限を 0 にして下方傾斜の根を選ぶ
    beta_bounds=(
        [np.nan, np.nan, np.nan],   # 下限（NaN → -inf）
        [np.nan,  0.0,   np.nan],   # 上限（価格係数のみ 0）
    ),
    # covariance_moments_mean=0.0,  # 感度分析用に m≠0 にもできる（既定 0）
)
```

ポイント:

- `beta` の **位置は X1 の列順**に対応。自分の formulation の列順に合わせて価格係数のインデックスを決める。
- 初期値に **0 は使わない**（0 は「固定」扱いになる）。
- price は外生扱いされないので自分自身は instrument に自動追加されない。**価格係数を識別するのが Z_C の役割**。

---

## 符号（upper / lower root）への対処

covariance restriction で価格上のパラメタを識別すると、**2つの解**が出ることがある。「upper」解は正（=右上がり需要）になりうる。「lower」解が正しいケースでは:

- `beta_bounds` で価格係数に片側制約（例：上限 0）を課す（価格係数が `beta` にある場合）、または
- `rc_type='log'`（`Problem` 引数）で価格に対数正規係数を使い符号を保証する

---

## efficiency 警告の正しい解釈

過剰識別時に「fully efficient ではないかもしれない」とあるのは:

- 完全効率な GMM なら covariance moment を**消去される線形係数（外生 beta・供給側 gamma）**の推定にも使うはず
- PyBLP は「Z_D/Z_S だけで線形係数を消去 → 後から covariance moment を theta 推定に追加」の2段構成なので、その情報を線形係数側に使い切らない

→ 影響範囲は**消去される外生係数・gamma の推定効率がわずかに落ちうる**ことだけ。

- 一致性（consistency）の問題ではない
- **価格係数の識別の問題ではない**

---

## ハマりポイント早見表

| 症状 / 疑問 | 原因 | 対処 |
|---|---|---|
| 「価格係数は covariance で推定できない？」 | 「linear parameters」を誤解 | 価格係数は `theta` 入り。`beta` に初期値を与える |
| `alpha should not be concentrated out` エラー | 供給側があるのに価格係数を NaN にした | `beta` の価格位置に有限の初期値 |
| covariance moment が効いていない気がする | 供給側 X3 / firm_ids 未指定で `ω` が無い | X3（限界費用式）と firm_ids を指定 |
| 価格係数が正（右上がり） | upper root を拾った | `beta_bounds` で上限 0、または `rc_type='log'` |
| `Non-default ... moment_types and covariance restrictions are not currently supported` | demand/supply_moment_types を非デフォルトにした | `'levels'` のまま使う |

---

## コードベースでの検証箇所（pyblp 本体）

別環境の PyBLP でも同じ実装か確認したいとき:

- `pyblp/parameters.py:253` — `eliminated_beta_index = np.isnan(self.beta)`（NaN=消去）
- `pyblp/parameters.py:265-272` — 供給側があると価格係数の消去を禁止
- `pyblp/parameters.py:382-397` — `store()`: 消去 / theta入り(unfixed) / 固定 の振り分け
- `pyblp/economies/problem.py:1395, 1477` — 線形IVは消去対象 beta 列のみ
- `pyblp/economies/problem.py:1463` — 線形IVの重みは `W[:MD+MS, :MD+MS]`（Z_C を含まない）
- `pyblp/economies/problem.py:1517-1521` — covariance moment は IV回帰の後に追加。勾配 `xi_jacobian*omega + xi*omega_jacobian`
- `pyblp/markets/market.py:1022-1024` — theta内 `BetaParameter`（価格係数）が製品特性=価格を通じて確率微分に寄与
- `tests/test_blp.py:1856-1957` — covariance moment の勾配を、線形パラメタを theta に入れた状態（`eliminate='no'`）で検証
- `pyblp/economies/problem.py:1764-1771` — 価格上パラメタの符号問題と `beta_bounds` / `rc_type` の案内（公式 note）

参考文献: MacKay, Alexander and Nathan Miller (2025). "Estimating models of supply and demand: Instruments and covariance restrictions." *AEJ: Microeconomics, 17*(1), 238–281.

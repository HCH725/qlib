# QLib fork 完整性、安全同步與回測保存契約

狀態：SCOPED_VERIFIED / NOT_DEPLOYED。2026-10-10，Kanban t_1769c0f1。
這是實作端交付，不是獨立 auditor verdict，也不是完整功能或正式 runtime 的 PASS。

## 1. 範圍與 GitHub 交付

- 基準：HCH725/qlib main `1c0204c9`；upstream main 本輪查驗為
  `54355232463878d2eebb91fe0ee5fa7fa1f5976c`，比 fork 的共同祖先新增兩個提交。
- 安全同步 PR：https://github.com/HCH725/qlib/pull/4。
  HEAD `7ac2350597dadafcfd71286fd6f9bec7aff4d373`。
- 此保存契約 PR 疊在安全同步分支上，合併順序為安全同步 → 保存契約。
  第二個 PR 只新增本報告、實測 evidence JSON、六個 artifact 測試；沒有新服務、DB 或 Runner。
- upstream `qlib/` 的路徑集合在合併後沒有缺漏；fork 額外加入的 runtime 檔案只有
  `qlib/backtest/hummingbot_dca.py`。本輪 wheel 含全部 234 個已追蹤 Python 原始檔。
  原始檔齊全不等於所有 optional 功能已安裝或可用。
- 不修改正式 qlib-run、容器建置／部署、正式排程、策略 repo、Wiki 或原始 HTML。
- 獨立 auditor 子卡：t_48cef5ee；實作者不自審，不合併、不部署。

## 2. upstream 兩個提交：不是文件-only 更新

| 提交 | 真實變更與風險 | 必須查驗的相容邊界 |
|---|---|---|
| `7c95268c0c174ea596dee86e0ff1efb29c0e3d09`（#2339） | artifact hardening，breaking change。MLflowRecorder 預設 RestrictedUnpickler；移除廣泛 numpy/pandas namespace 信任，改成精確 globals；可執行物件需明示 trusted。增加 highfreq artifact-root 路徑檢查、HIST JSON mappings、線上／延遲訓練／DDGDA scoped trust 與 migration guide | DataFrame round-trip 與模型／Position／task 的可執行 pickle 是不同信任層。HIST 舊 object-array .npy mappings 不能當作安全資料直接載入。舊 workflow 可能需要針對特定讀取加 consent，不得全域開 trusted=True |
| `54355232463878d2eebb91fe0ee5fa7fa1f5976c`（#2340） | config execution hardening。以 AST expression parser 代替 eval、model/report registry 代替字串 eval、檔案 .py module 須 component 層 scoped trust；保留 package import 及合法 QLib expression 語法 | `trusted` 放 component 的 class/module_path 同層，不是 constructor kwargs；巢狀 component 分別授權。file-module consent 不授權 pickle；expression parser 不是不可信 Python 套件的沙箱 |

兩筆提交原樣 merge（無 conflict resolution）；98 個 upstream 檔案變更，
不是手工摘取幾個安全片段或任意升級整庫。遷移文件：
`docs/start/artifact_migration.rst`、`docs/start/config_migration.rst`。

GitHub 第一輪實際 CI 發現 fork 原有 DCA 的四個 mypy 錯誤（job 114182258401）。
因此同步 PR 另附最小相容修正：side/mode 增加 object 型別、Decimal sum identity，
浮點 label slice 改成等價 `slice(entry_timestamp, None)` 物件。
沒有更動 BUY/SELL、MAKER、BO/SO 時序、PnL、費用或 generic backtest。
七個原有 DCA regression 與新增六個 persistence test 通過；這不是新增 Hummingbot live parity 證據。

## 3. 「完整」功能依賴矩陣

依據：本 repo `pyproject.toml:27-116`、`.github/ci/constraints.txt`、
`.github/ci/test-requirements.txt`、`qlib/contrib/model/__init__.py` 及实际 import/call sites。
官方 extras 並沒有名為 all 的 extra；Makefile 的 all target 也不等於全功能組合
（例如其安裝列未包含 client，卻列了不適用所有平台的 pywinpty）。不要以 `make all` 作完整性證明。

| 能力／extra | 依賴與用途 | DCA 回測必要？ | CPython 3.12 / ARM64 證據與成本／衝突 |
|---|---|---|---|
| 原生 core：資料、expression、Dataset、generic backtest、workflow、MLflow | NumPy/pandas、Cython C++ rolling/expanding、MLflow<3.13、filelock<3.30、pyarrow、LightGBM、cvxpy、gym 等原生宣告 | 原生 backtest/workflow 所需；DCA math 本身用 pandas/NumPy，但 public qlib.backtest 的 import closure 仍載入其他 core 模組 | macOS ARM64/Python3.12 editable build、native extensions、599 tests PASS；Linux ARM64 尚未實跑。core 仍包含 jupyter 等較重依賴，保留 upstream 宣告，不另拆 dependency 架構 |
| analysis | plotly<7、statsmodels；報圖與 model analysis | 否；不要把有圖當作逐筆保存證據 | 本機 import PASS；Linux aarch64 resolver PASS。Plotly 6.9.0 pure wheel 約9.45MiB，statsmodels0.15.0 cp312 ARM64 wheel 約11MiB；renderer／完整所有報圖未宣稱 PASS |
| client | python-socketio<6、tables；remote client / HDF cache | 否；本地資料／DCA 不需要遠端 Qlib server | 本機 import PASS；Linux resolver PASS。tables3.11.1 的 cp311-abi3 wheel 支援 CPython3.12（不是因無 cp312 字樣就判 missing）；Linux aarch64 wheel 約6.6MiB。實際 server/Redis/socket 通訊未驗證 |
| PyTorch 神經模型（不是獨立 neural extra） | torch；HIST/GRU/LSTM/TRA 等 contrib model；各 benchmark 自己有 requirements | 否，除非策略需要神經模型；不是 torch 只能給 RL 用 | 本機 ARM64 torch2.14.1 安裝及 import、HIST 真實離線訓練/保存/回測 workflow 1項 PASS。DDGDA full workflow 在本機 SIGSEGV，不能標完整 neural/meta PASS。torch macOS wheel 約121MiB；Linux aarch64 wheel約433MiB，預設 PyPI resolver 還帶 CUDA/NVIDIA 依賴，不代表 CPU 映像所需 |
| rl | tianshou<=0.4.10、torch、numpy<2；RL trainer/policy/order execution | 否；一般 DCA 不需要 RL | 另建隔離 .venv-rl，實裝 extra、pip check、tianshou/torch/MLflow/qlib.rl.trainer/policy import PASS；Linux resolver PASS。完整 RL fixture-dependent 測試缺 intraday_saoe 資料，12項 fail，不是演算法 PASS。gym 為舊 API；不擅自改 gymnasium／新版 Tianshou |
| dev | pytest、statsmodels、舊 Python 的 tomli | 僅驗收；非正式 runtime 必要 | 本機測試安裝 PASS；dev 不包含所有 neural/RL/market fixtures，不能只裝 dev 就跑所有 suite |
| lint | black<26.1、pylint、mypy<1.5、flake8、nbqa | 否 | mypy1.4.1（56 files）、Black246 files、QLib flake8 PASS；不是聲稱所有 make lint 子項已跑。自行額外裝 pandas-stubs3.x 會改變上游既有型別基線，不是本次正式 lint dependency |
| docs | scipy<=1.15.3、Sphinx/RTD、snowballstemmer<3 | 否 | 官方對 scipy 的 docs bound 保留；SciPy1.15.3 macOS wheel 在 macOS27 有 thread_bss loader error。本機測試改用1.16.3，不能說 docs build 已 PASS，也不將本機 workaround 改成全平台核心 pin |
| package | build、twine | 只在建置時 | CPython3.12 ARM64 wheel build PASS；234/234 tracked .py 無漏檔。未測 upload；既有 MANIFEST 包入 pycache .pyc，列為非功能性 packaging hygiene，不趁本卡大改 |
| test | yahooquery、baostock、lxml<6.1.3 | 非核心，資料收集/PIT相關 tests | 不把外部市場下載／網路 fixtures 當作 native unit test；expression compatibility TestAutoData 的外部下載逾時，未計 PASS |
| model-specific：CatBoost/XGBoost/TabNet/TFT 等 | 見各 examples/benchmarks/*/requirements.txt | 僅特定模型需要 | 不能以 core+rl 安裝推論所有 contrib 模型齊全；本輪未跑所有 benchmark、GPU、資料來源、remote services，明列 NOT_VERIFIED，而非安裝一個巨型 all |

### 3.1 可重現最小建置／安裝規格（沒有改正式容器）

1. 先以已審核 Git commit 建新 image／隔離 venv；Python3.12、ARM64、QLib core。
   先驗 rolling/expanding、generic imports、pip check。正式 runtime 的既有 packages
   只採卡面已提供的 frozen inventory，不把本機測試 venv 冒充容器現況。
2. 只有需要報圖／HDF或remote才選 `.[analysis]` / `.[client]`。
   本輪 resolver 命令：
   `uv pip compile pyproject.toml --extra analysis --extra client --python-version 3.12 --python-platform aarch64-manylinux_2_28`。
   解得214 packages；resolver 成功≠Linux wheel建置、import或執行PASS。
3. RL 另行選 `.[rl]`，優先隔離 research/test image，不在 qlib-run 強塞 Torch/Tianshou。
   同樣 target 的 RL resolver 解得221 packages；numpy1.26.4、tianshou0.4.10、torch2.14.1、
   cvxpy1.7.5。須區分 CPU Torch index 和預設 CUDA 成本，不能沿用 resolver 清單直接灌 production。
4. `numpy<2` 會迫使 NumPy2-only 的新 cvxpy／contourpy 降版。
   先装 NumPy2 core 再單獨降 numpy，曾造成 pip consistency error；重新對整個 manifest
   解算後為cvxpy1.7.5、contourpy1.3.3。不是修改 metadata 或忽略 pip check。
5. 本機驗收（非Linux安裝指令）的確定基線：
   `uv pip install --python .venv/bin/python -e '.[dev,analysis,client]' 'numpy<2' 'scipy==1.16.3' torch 'mypy<1.5' 'black<26.1' build flake8`。
   從乾淨venv做單次一致解算；SciPy1.16.3只為本次macOS27驗收，非production要求。
   不需要神經模型的 runtime 不要求 torch；test venv需要是因為安全workflow tests import meta models。
6. 未新增 pyproject extras／新 core pin：既有 extras 足以表達選用功能。
   上線前由新image跑相同 acceptance，不因有 wheel 就跳過 Linux ARM64 驗證。

wheel metadata 原始來源為 PyPI JSON：
`https://pypi.org/pypi/{distribution}/{version}/json`。
本輪查驗 numpy1.26.4、scipy1.15.3、torch2.14.1、tianshou0.4.10、statsmodels0.15.0、
plotly6.9.0、tables3.11.1、python-socketio5.17.0、mlflow3.12.0、lightgbm4.7.0、
pyarrow23.0.1、cvxpy1.9.3；wheel存在與compatible metadata並未等同runtime qualification。

## 4. 原生 recorder 保存欄位矩陣

「自動保存」下列均指真的呼叫對應 generate／task_train 的 run；不是裝好QLib就會寫入。
本輪没有正式研究 run 產物，不把測試 artifact 宣稱為歷史策略交易。

| 欄位／產物 | 原生 SignalRecord / PortAnaRecord / Trainer | standalone DCA 回傳內容 | Pipeline 必須另行明寫／驗收 |
|---|---|---|---|
| 訊號prediction/label | SignalRecord.generate存pred.pkl；DatasetH時存label.pkl（可為None），不是逐筆訂單 | 輸入DataFrame，不自動呼叫SignalRecord | 有model才用SignalRecord；不假造prediction給純規則DCA |
| IC等signal analysis | SigAnaRecord：sig_analysis/ic.pkl、ric.pkl；可選long_short_r.pkl/long_avg_r.pkl，並log metrics | 無 | 需要該analysis才顯式配置record；不存在時不能冒充已產出 |
| 期別portfolio report | PortAnaRecord：portfolio_analysis/report_normal_{freq}.pkl，來源generic normal_backtest | net_pnl_quote/net_pnl_pct、filled_amount_quote、cum_fees_quote等每bar序列 | 明確區分generic report與DCA raw frame，不能拿PortAnaRecord當DCA recorder |
| positions | positions_normal_{freq}.pkl，generic Position objects | 沒有完整cash/base-position account snapshots | 原生Position讀取可需 scoped trusted=True；DCA portfolio positions需Runner/engine另提供 |
| equity | generic report 的account/cash/value類數據可形成帳戶曲線 | net_pnl_quote不等於全帳戶equity | 提供初始資金、空倉時段、並行executor、槓桿／保證金／funding及估值定義；不能把quote allocation doubled-on-exit當equity |
| 風險摘要 | port_analysis_{freq}.pkl；risk_analysis含成本／benchmark口徑；log_metrics | 沒有portfolio Sharpe/CAGR/MDD整體分析 | 指標必綁定同run、同period/window及account equity/cost model；只需有科學意義的校驗，不加任意績效gate |
| 成交indicator | indicators_normal_{freq}.pkl、indicators_normal_{freq}_obj.pkl、indicator_analysis_{freq}.pkl | 無generic Indicator | 指標含gross aggregations也不等於每筆fill journal；PortAnaRecord.list並非完整寫入清單 |
| 每筆entry/exit時間 | generic orders有start/end，這是訂單窗口，不是完整fill archive；原生record模板未存每筆fill journal | timestamp及stage filled_amount_quote_i首次非零可見bar-level fill線索；最後bar與close_type可見終止 | 真正逐event entry/exit timestamp與bar/availability時間另存；不能逆推後冒充交易所事件 |
| entry/exit價格 | exchange在update_order處取得trade_price，但record模板不自動保存每fill price | close、配置prices、current_position_average_price；不是獨立每筆executed-price欄 | engine須提供真正模拟成交價及價源；不存在就NOT_AVAILABLE/UNSUPPORTED，不以close偷偷補 |
| 數量／方向 | generic holdings/order amount可得，未自動成fill export | amounts_quote、stage quote allocations；side為input，frame未存side | 明確base/quote units、side及notional；不將quote allocation寫成base quantity |
| 手續費／費率 | generic報表cost是聚合值，配置open_cost/close_cost可記task | cum_fees_quote = round-trip估計；trade_cost為input，非entry/exit逐筆實際扣費 | 存fee_model、rate、fee_currency、per-event fee；round-trip假設標記，不編造funding/liquidation |
| BO/SO | generic模板不識別DCA stage | filled_amount_quote_0..N、net_pnl_quote_0..N；未fill stage可不存在，frame保留各stage序列 | 原frame完整保存並記stage index、BO/SO語義與original config；不能只存terminal metrics |
| close_type | 原生generic無DCA close_type | 獨立tuple字串（不是frame欄位） | 額外保存，否則return tuple第二項丟失 |
| model/dataset/task | task_train：task、params.pkl（model物件）、dataset、flattened參數；R.start的run生命週期 | 純函式不建立recorder／run | run_key與recorder ID顯式綁定；不能把saved model當config參數 |
| source/version/config/data | MLflow有run id、params/tags與uncommitted-code等線索；不保證固定strategy SHA/data snapshot與image digest | 未保存 | 固定git SHA、QLib版本、image/source hash、實際有效config、data snapshot、fee model |
| run_key / IS/OOS | QLib原生run ID不等於Pipeline不可變run_key；沒有通用OOS隔離gate | 無 | period_basis、window_hash、研究輪次、frozen lineage另存，與submit request逐欄相符 |
| 原始檔URI/hash | MLflow artifact URI與保存API存在；原生record不產HTML三URI與hash manifest | 只回傳記憶體dataframe/string | 明確寫檔、下載讀回、驗bytes/hash/schema再發布完整manifest |

來源呼叫鏈：`qlib/model/trainer.py:_log_task_info/_exe_task/task_train` →
`RecordTemp.save` → `MLflowRecorder.save_objects`。
`SignalRecord.generate`、`ACRecordTemp.generate`、`PortAnaRecord._generate/list` 在
`qlib/workflow/record_temp.py`；generic report/account見 `qlib/backtest/account.py` 與 `report.py`。
`ACRecordTemp.generate`可能因dependency缺檔只warning+return；Runner不能用「generate未raise」判完整。
`RecordTemp.check`只驗artifact列舉，不驗內容／hash／identity。

## 5. 新 Runner 的現況與最小保存契約

### 5.1 找到什麼／沒找到什麼

QLib repo實際 DCA callers只有public export及七個測試。
搜尋 `run_key|metrics_uri|trades_uri|equity_uri|backtest_hummingbot_dca|simulate_hummingbot_dca`，
沒有HTML所提新Pipeline submit/status/result Runner，也沒有DCA→recorder production call site。
原生qrun/model trainer是模型workflow入口，不是該HTTP Runner。
因此不新造Runner/Manager/DB，不擅自跨到策略repo或舊host bridge補線。
本卡交付的是契約＋可執行保存範例測試；實際Pipeline wiring仍待後續施工。

### 5.2 當Runner存在時，成功前必須做的事

- 同一run_key固定對应recorder run ID；初始化明示既有tracking URI，避免誤寫到default cwd。
- 在existing `R.start`上下文內執行真正engine；有model才顯式配置
  SignalRecord/SigAnaRecord/PortAnaRecord。standalone DCA不是PortAnaRecord的回測路徑。
- DCA立即用既有`recorder.save_objects(artifact_path="hummingbot_dca", **{"ledger.pkl": ledger})`
  保存完整raw frame；close_type、side/mode、完整prices/amounts_quote/TP/SL/trailing/time_limit/trade_cost
  以及版本／資料／期間metadata分開存。Decimal config用可重現字串，不自行轉float後丟精度。
- 新engine沒有per-fill event journal就不要發布`trades_uri`指向raw frame冒充完整fills；
  journal須包含event/executor/position ID、stage、方向、entry/exit時間與價格、base/quote數量及費用。
  equity須是真正account series；這兩項尚未產生時complete=false，不是策略績效Reject。
- 使用既有MLflowRecorder.save_objects(local_path=...)存JSON/CSV/Parquet原始檔；
  `load_object`是pickle reader，不拿它讀JSON。摘要／manifest用JSON避免需要可執行pickle信任。
- 成功前以固定required set校驗，不接受manifest自報「只有我有的檔才required」。
  逐artifact下載到隔離位置，檢查exists/file、size、SHA256、schema/row counts/units/period範圍。
  bytes/hash針對真正落盤artifact，不對記憶體DataFrame或pickle重序列化計算。
- ledger完整（含BO/SO序列）是必需；zero-fill可合法保存空交易journal／無交易摘要，
  不因空檔就捏造成交。檔案完整與是否可績效晉級分別判定。
- 在R.start結束前驗收；失敗raise以保留FAILED，不讓上下文自動留下看似FINISHED的run。
  save_objects上傳為同步，async metrics在end_run等待；兩者都不是Pipeline的complete自動保證。
- manifest最後發布；所有required產物＋run identity校驗後才complete=true。
  上傳/驗收失敗不刪原始證據、不計策略Reject、不重算不同run_key繞過故障。
  既有store重試沿用同run；如已published内容/hash不同應停下查證，不覆寫成功證據。
- hash提供完整性，不提供不可信來源的認證。Runner以自己受控store的manifest核對submit context；
  n8n只接受經信任且身份對應的Runner ACK，不藉重新下載所有大檔增加workflow負擔。

### 5.3 六個實跑測試（不是部署的validator）

`tests/backtest/test_hummingbot_dca_artifacts.py`使用明示synthetic bars，
真實backtest_hummingbot_dca、真實MLflow file store與MLflowRecorder，不mock保存成功。
保存raw stage ledger、close type/config metadata與hash manifest，讀回比較DataFrame與metadata。
另五個負測試：missing file、truncated bytes、wrong hash、manifest missing required entry、wrong run ID。
測試從下載檔案重新計算 size/hash，再與呼叫端獨立持有的預期 manifest 比較；run ID 也逐欄核對，避免僅因整份字典不相等就提前通過負測試。
這是固定兩artifact的最小範例；production要由真實 submit/run context 驗證身份，並提供自己的metrics/fills/equity/identity schema。
不把示例manifest的run_id或raw ledger冒充完整Pipeline result。

## 6. HTML v8.6 只讀核對及04/05精確補強文案

原檔：`quant_pipeline_2026-10-10_v8_6_minimal_remediation_draft.html`。
SHA256 `88e973e33d8e3e68b0dcd584cb0edaebace424d4eefe705da2063ae886b1ab6d`；114085 bytes。
沒有修改HTML。本輪審的是文字契約，不以HTML可開啟宣稱已跑n8n。

已有：218/312行URI-only raw evidence/Wiki摘要邊界；335、350–368行submit/status/result；
641–659行04/05 nodeSpecs、period/window identity、有界Wait、IS/OOS分流。
缺口：沒有明文QLib recorder call、DCA tuple完整落盤、逐fill event/schema、account equity產源、
成功前artifact bytes/hash驗收。05範例只信complete=true與身份，不能發現Runner漏保存。

### 建議追加到04 settings／guard（原文可直接轉交，暫不回寫HTML）

「04 Runner在同run_key所綁定的QLib/MLflow recorder內執行回測。
模型workflow顯式生成SignalRecord及需要的PortAnaRecord；DCA使用既有standalone入口，
另以save_objects保存完整BO/SO raw ledger及close_type/effective execution config，
不能以PortAnaRecord自動包含DCA逐筆交易作假設。
逐事件fills與account equity必須由真正engine/Runner提供且語義標明；
目前source沒有該產源時標示結果不完整，不用bar close/quote allocation偽造base fills。
在宣布SUCCEEDED/result.complete=true以前，required artifacts逐檔讀回驗存在、大小、SHA256及
schema/period/run identity，並凍結result manifest；缺檔/失配是技術待修，不是策略Reject。
未建置Runner或未驗實接口時保持待施工，不新造DB/Manager/Workflow。」

### 建議追加到05 settings／output（原文可直接轉交，暫不回寫HTML）

「05只接受受信任Runner在保存驗收後發布的小型manifest。
核對run_key/run_id/family/source SHA/config hash/data snapshot/fee model/period_basis/window_hash；
metrics_uri、trades_uri、equity_uri須各有artifact kind、schema version、size_bytes、sha256、row_count，
另索引DCA ledger/metadata，不能把raw stage frame當完整fills或PnL當account equity。
完整性ACK與manifest hash、required set均來自該固定run；既有artifact URI沒有存在/hash驗收不算完整。
n8n不載完整K線/trades/equity，artifact bytes驗證在Runner端做；
缺產物或有hash/identity錯配→Needs Review，只有完整IS→06，完整OOS→11。
OOS不得進研究MoA，QLib FINISHED或n8n execution success均不單獨構成完整結果。」

### Result manifest 最小欄位（待Runner施工時落地，不是現有API）

- identity：run_key、run_id/recorder_id、family_id、source_sha、engine_version/source/image digest、
  effective_config_hash、campaign_hash、data_snapshot_id、fee_model_hash、period_basis、window_hash、
  round_index及已要求的parent/hypothesis/frozen lineage。
- artifacts：metrics、trades、equity、DCA raw ledger及execution metadata，各帶kind/URI/SHA256/
  size_bytes/schema_version/row_count；generic positions/analysis按實際path有則另列。
- completion：required_artifact_kinds、complete、validation summary及manifest hash。
  required set由實際execution path的契約決定，不由run結果縮減；沒有fills/equity產源時complete=false。
- trades：時間單位／可用時間、價格源、base/quote units、side/stage/executor/position IDs、費率/費用幣別。
- equity：完整固定評估區間、初始權益、空倉／並行executor估值、資金與成本模型；
  funding/liq未支援即明示，不能依賴OOS或policy推論其存在。
- 42格campaign若外層聚合：manifest列全部evaluation-unit結果及status；child身份包含
  symbol/timeframe/leverage/effective_config，不把同一family不同格混成一個run或只挑最佳格。

追加乾跑驗收（不改原T01–T29）：真實寫檔再讀回；ledger/trades/equity任一缺失；
size/hash/schema錯配；close_type漏存；同key不同artifact内容；IS/OOS身份錯配；
零成交但完整證據；stage allocations最後bar doubled不被當新BO/SO或equity；
失敗寫檔不留complete=true。這些是必要科學/身份/落盤檢查，不是新績效gate。

## 7. 驗證結果、blocking與非blocking

機器可核對快照：`docs/start/fork_completeness_evidence.json`。
source HEAD為安全同步的7ac23505；新增測試SHA也記在快照，避免把未提交測試與head混淆。

已驗：一致dependency環境pip check；599 tests + 82 subtests；mypy56 files；
Black246 files與qlib flake8；wheel build及234/234原始檔；HIST full workflow單獨1項。
所有資料都是測試fixture，不是策略實績或真實交易。
未完整跑`make lint`（包括所有pylint/mypy stub安裝/nbqa）、Sphinx docs及所有benchmark。
Makefile的make test其實是安裝test extra，不是pytest；因此明示真正pytest命令，避免假驗收。

Blocking（對部署／完整功能聲稱，不攔本契約交付）：
1. 獨立auditor尚未給verdict；GitHub每個PR必須讀回exact-head CI，不因本機pytest綠就合併。
2. 本repo不存在新Pipeline Runner與完整fill/equity exporter；不能宣稱正式回測已無缺資料。
3. 未實跑Linux ARM64新image；resolver/wheel metadata不能作target-platform PASS。
4. 在本macOS27 host，DDGDA full workflow SIGSEGV；需要該能力者不可視為已資格化。

Material/conditional：RL完整fixture測試未完成；Torch CPU/GPU依賴成本與NumPy重解算；
scoped trusted consent遷移；SciPy docs bound與macOS wheel問題。
這些在需要該feature時阻塞其qualification，但不應強塞依賴阻擋純DCA研究。

Optional/nonblocking：wheel pycache hygiene、未用client server與報圖renderer、
其他model-specific套件／不需要的資料收集benchmark。沒有為它們新增runtime gate。

此報告與測試可以交獨立auditor審查；它們不取代後續Runner施工、平台驗收或部署授權。

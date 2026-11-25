# 关联抽样工具（Join Sampler）

两张明细表按键关联后，合法关联对可能达到数亿甚至上百亿行，无法整体预览。
本工具在 **不先对各表随机抽行、不物化完整关联结果** 的前提下，对关联对做
两级均匀抽样，并为每个被抽中的结果对给出可核验的入选依据。

仅依赖 Python 3.11+ 标准库；Web 前端为原生 HTML/CSS/JS，无构建步骤。

## 1. 抽样模型

内关联，仅两侧都出现的非空键参与（NULL / NaN 键被排除）。

1. **键级**：键 `k` 的权重与其 *可接受关联对数* 成正比；
2. **键内级**：在该键的可接受行对中均匀抽一个 `(左行号, 右行号)`。

无条件时，权重 `w_k = nL(k) · nR(k)`，每个关联对的入选概率严格相等：

```
P(对) = w_k / Σ_j w_j · 1 / w_k = 1 / Σ_j nL(j)·nR(j)
```

### 成对条件下的接受率校正（不能沿用无条件概率）

条件作用于整行对（如 `abs(L.v - R.v) <= 100`）。每个键单独处理：

| 键类型 | 处理 | 接受率 | 校正权重 |
|---|---|---|---|
| 小键（对数 ≤ `max_pairs_per_key`） | 全枚举所有对 | 精确 `a_k = 接受数/对数` | `pair_count · a_k` |
| 大热点键 | 均匀探针估计（`estimator_probes` 个，去重计不同接受对） | 估计 `â_k`，标记为估计值 | `pair_count · â_k` |
| 探针零命中的热点键 | add-one 保守平滑 `a = 1/(probes+pair_count)`，标记 `rate_smoothed` | 极小正下界 | 正权重，稀有接受对仍可被抽中 |
| 零结果键 | 枚举/探针均不接受 | 0 | 0，永不被抽中 |

校正后每个 **接受对** 仍等概率（测试 `test_every_accepted_pair_has_identical_joint_probability`
以 6022 个接受对、3 万次抽样做卡方验证，p ≈ 0.9）。估计键的结果与全枚举参考
分布做卡方对比，估计接受率落在二项标准误的 3σ 区间内。

每条结果记录给出的是**真实条件入选概率**而非无条件概率：

- `p_select_key` = 抽中该键时其 *存活* 校正权重 / 当时总权重（WOR 下随扣权变化）；
- `p_select_pair_within_key` = 1 / 该键 *存活* 可接受对数；
- `p_joint` = 两者乘积。精确模式下每个接受对的 `p_joint` 严格等于 `1/M`
  （M 为当前可接受对总数），估计模式下为基于估计容量的近似值并如实标注。

### 有放回 / 无放回

- **WR（有放回）**：每次抽取独立；即使源表存在完全重复的行，每行有独立的
  扁平行号身份，重复行可各自被反复抽中。
- **WOR（无放回）**：Fenwick 树按存活对扣权（精确权重整数扣减，估计权重
  按估计容量等比扣减并保留相对极小正下限，不提前扣到 0），键内通过全局
  已选集合排除重复对；精确权重下任意存活对的条件入选概率恒等，可一直抽到
  全部可接受对耗尽。
- 键内取对只走两条严格均匀的路径——「均匀探针 + 接受/未用判定」的拒绝抽样，
  或在完整存活接受对列表上均匀选取——**绝不在有偏的小蓄水池上直接取对**。
  当拒绝抽样触达自适应上限时，改为惰性枚举该键当前未使用的全部接受对
  （`alive_cache`，运行期派生物，不写入检查点，恢复后按需重建），之后在
  存活列表上 O(1) 均匀抽取；若某键真实存活对耗尽，清零其残余 Fenwick
  权重并重抽其他键，由有界重试保证终止。

## 2. 关键工程约束与实现位置

| 约束 | 实现 |
|---|---|
| 不能先各表随机抽几行再关联 | 两级抽样直接在关联对空间进行（`sampler.py`），表行从不预抽 |
| 不能存完整关联结果 | 只保存键计数、键内行号索引与受限的接受对缓存；`materialized_join_rows` 恒为 0 |
| 计数与取样同一快照 | `JoinSnapshot` 构造时刻固化键集合/统计/指纹；`SideData` 之后追加段不影响已建快照 |
| 条件必须校正接受率 | 精确枚举 + 探针估计两条路径，零结果键权重清零 |
| 恢复保留随机状态与已选对 | 检查点保存 `random.Random` 完整内部状态、Fenwick 剩余权重、初始总权重、已选对集合与全部记录 |
| 不能换种子重挑样本 | 恢复校验快照指纹、种子与**完整配置逐项等值**；任一字段缺失/篡改抛 `ResumeError`，严格 JSON（拒绝 NaN/Infinity） |
| 热点键、空值键、整键零结果 | 热点走估计路径；NULL/NaN 不参与关联；零结果键统计并排除；探针零命中走平滑而非误判为零 |
| 资源上限 | 构建期支持求值次数 / 秒数 / 内存字节三类预算，超限抛 `ResourceLimitExceeded` 并返回部分统计 |

## 3. 目录结构

```
join_sampler/
  store.py        分段列式存储、键->行号索引、空值键判定、内存估算
  snapshot.py     不可变关联快照与逐键统计
  conditions.py   AST 白名单成对条件求值器（无 eval/exec，无属性逃逸）
  fenwick.py      整数/浮点 Fenwick 树：加权抽键与 WOR 扣权
  sampler.py      两级抽样器：接受率构建、WR/WOR、资源上限、恢复状态
  checkpoint.py   检查点编解码（键类型标签化，支持 tuple 键）
  data.py         CSV 装载、类型推断、合成数据生成器
  stats.py        卡方/二项检验（纯 stdlib 的不完全 Gamma）、两版对比
  cli.py          命令行
  webapp.py       stdlib http.server 后端（自实现 multipart 解析，无已废弃的 cgi）
  web_static/     前端（index.html / styles.css / app.js）
tests/            52 个 unittest 用例（含受限缓存/估计/安全加固边界集）
```

## 4. 命令行用法

```bash
# 生成演示数据（含热点键与空值键）
python -m join_sampler.cli synthetic --out-dir ./demo --seed 42

# 抽样并写检查点
python -m join_sampler.cli sample \
  --left demo/left.csv --right demo/right.csv --key k \
  --condition "abs(L.v - R.v) <= 200" \
  --n 200 --mode WOR --seed 7 \
  --checkpoint run.cp.json --out run.json

# 从检查点恢复后追加一批（种子必须一致，随机状态延续）
python -m join_sampler.cli resume \
  --checkpoint run.cp.json --more 100 \
  --left demo/left.csv --right demo/right.csv --key k \
  --seed 7 --out run2.json
```

结果 JSON 中每条记录包含 `p_select_key`、`p_select_pair_within_key`、
`p_joint`、`weight_mode`（exact/estimate/unconditioned）、`rate_is_estimate`、
`rate_smoothed`（探针零命中平滑标记）、`rejection_attempts` 等入选依据字段。

## 5. Web 工作台

```bash
python -m join_sampler.webapp --host 127.0.0.1 --port 8765
# 浏览器打开 http://127.0.0.1:8765
```

功能：

- 合成数据或上传两侧 CSV（multipart），配置条件、WR/WOR、种子与资源上限；
- 每个结果对一行入选依据（三级概率、权重模式、拒绝尝试次数、两侧行内容）；
- 键分布 SVG 柱状图（命中次数 vs 校正权重，双系列带图例与 hover tooltip，
  数据同时以表格呈现，浅/深主题调色板经 CVD 校验）；
- 内存面板：两侧表存储、抽样器结构、接受对缓存、物化关联行（恒 0）；
- 「追加一批」走真实检查点恢复路径，历史事件记录种子未变；
- 两版运行比较：共同/独有行对数、逐键命中差值、快照指纹是否一致。

## 6. 条件表达式

白名单语法：字面量、`L.<列>` / `R.<列>`、算术（`+ - * / // % **`）、
比较（`== != < <= > >=`）、布尔（`and/or/not`）、三元表达式，以及函数
`abs/min/max/round/floor/ceil/sqrt/pow`。禁止调用、属性逃逸、lambda、导入。

安全加固：

- 列名以下划线（含 `__dunder__`）开头、或与映射方法同名（`keys/values/items/
  get/copy/update/pop`）在编译期直接拒绝，堵死 `L.__class__`、`R.keys` 之类的
  属性逃逸与语义混淆；
- `**` 与 `pow` 走带资源上限的安全封装：指数绝对值 ≤ 100、整数底数/结果位数
  ≤ 4096、浮点结果必须有限，超限在求值期按该对不接受处理，防止
  `2 ** 100000000` 之类的 CPU/内存耗尽；
- 缺列、类型不兼容、除零、溢出（含触发上限的 `MemoryError`）等运行期异常
  一律按该对不接受处理。

## 7. 测试

```bash
python -m unittest discover -s tests -v
```

52 个用例覆盖：

- **小表全枚举**：行对层面卡方 + 二项精确检验等概率；WOR 抽尽全部对且不重复；
  重复行独立身份；热点键占比；NULL/NaN 键；
- **条件校正**：精确接受率、整键零结果、估计热点键 vs 全枚举参考、
  全部接受对联合概率相等、条件沙箱；
- **受限缓存/估计边界**（`test_bounded_cache_edges.py`）：缓存预算 < 接受对数时
  蓄水池路径的 WR 严格均匀与 WOR 全量耗尽/边际均匀；估计热点键小缓存 WOR
  不崩溃、不重复并惰性构建存活缓存；探针零命中平滑仍能抽中稀有对；
  FloatFenwick 动态 epsilon 不扼杀幸存键；沙箱下划线封堵与 pow 上限；
  stats/配置入参校验；异构 tuple 键排序；损坏检查点收敛与配置等值恢复；
- **大表（2×20 万行、数亿关联对）**：不物化、内存随行数而非对数增长、
  三类资源上限中止、大表分布 sanity；
- **快照与检查点**：追加段后快照不变、WR/WOR 恢复与不中断运行逐条一致、
  拒绝异快照与换种子、tuple 键检查点往返、损坏/截断文件报 `ResumeError`；
- **Web API**：合成/上传/追加/比较/422/400/静态资源的真实 HTTP 端到端。

## 8. 已知边界

- 检查点 JSON 支持 int/float/str/bool/None/tuple 键；其他可哈希键类型会在
  保存时明确报错（CSV 推断与合成数据只产生前述类型）。
- Python 语义下 `1 == True`、`1.0 == 1` 哈希同桶，属于语言既定行为；异构键的
  排序由递归类型标签给出确定性全序，但不改变字典的归并语义。
- 估计热点键的 WOR 容量为估计值：Fenwick 权重保留正下限并在键真实耗尽时
  清零重抽，因此不会因估计偏低而提前报错；极端低接受率下惰性全枚举的
  单次代价为该键 `pair_count`，是空间受限下的有意取舍。
- Web 运行状态保存在服务端内存中；进程重启后用 CLI 检查点文件恢复。

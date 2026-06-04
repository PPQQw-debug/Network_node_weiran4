# Branch Builder 项目开发备忘录

本文用于压缩长对话上下文，记录本项目已经形成的关键经验、踩坑点和后续必须遵守的约定。

## 当前项目目标

本项目是一个用于搭建网络节点电路的图形工具。用户可以在画布中绘制二节点支路、四节点单相变压器、自定义 N 节点黑盒、Y 节点黑盒，并输出：

- 完整节点方程：`I = G V + Ihis`
- 节点消去后的等效节点方程
- 被消去内部节点的电压恢复公式
- 支路观测电流，以及节点消去后的支路电流表达式

前端主要负责绘图、交互、显示和数据组织；后端 Python 负责严格的矩阵消元、符号化简和支路观测电流化简。

## AI Agent 快速入口 / Quick Map

后续 AI agent 接手时，优先读本文件，再按任务读取相关代码。不要一上来重构整个 `index.html`。

### 本地启动

推荐启动方式：

```powershell
python local_server.py
```

默认地址：

```text
http://127.0.0.1:4177/
```

`local_server.py` 会提供静态页面和后端 API。每次调用 `/reduce-system`、`/validate-blackbox-observers` 时，它会启动对应 Python 子进程，所以修改 `reduce_api.py`、`blackbox_validation_api.py`、`elimination.py` 后通常不需要重启服务；刷新页面或重新触发计算即可。

### 主要代码入口

- `index.html`：单文件前端，包含画布、状态管理、节点组装、i18n、公式渲染、导入导出、YBox 打包拆包。
- `local_server.py`：推荐本地服务，负责静态文件、保存导出、列出电路、调用 Python API。
- `reduce_api.py`：节点方程消去 API。解析前端 payload，调用 `elimination.py`，返回 `G_red`、`Ihis_red`、`K_v`、`K_h` 和 reduced observers。
- `elimination.py`：节点消去的数学核心。当前实现是逐节点 Schur 消去，不显式构造 `inv(G_ii)`。
- `observers.py`：支路观测电流跟随节点消去后的表达式恢复。
- `blackbox_validation_api.py`：黑盒 observer 校验 API wrapper。
- `nodal_tool/blackbox_validation.py`：黑盒 observer 与用户输入 `G/Ihis` 的一致性检查。
- `server.js`：可选 Node 服务；推荐优先用 Python 服务。
- `tests/`：回归测试。数学或 API 改动后必须跑。

### 前端到后端的数据流

1. 前端根据当前画布状态把元件和导线整理为全局节点网络。
2. 前端生成：
   - `all_nodes`
   - `voltage_nodes`
   - `external_nodes`
   - `G_full`
   - `Ihis_full`
   - `observers`
3. 前端 POST 到 `/reduce-system`。
4. `local_server.py` 调用 `reduce_api.py`。
5. `reduce_api.py` 调用 `eliminate_internal_nodes(...)`。
6. 后端返回 reduced equations 和内部节点恢复公式。
7. 前端负责缓存、渲染、语言切换后的展示。

### 测试命令

核心 Python 测试：

```powershell
python -m unittest discover tests -v
```

前端公式格式化测试：

```powershell
node tests\frontend_math_formatter_cases.mjs
```

注意：在某些 Codex / Windows 沙箱中，`node.exe` 可能因为权限被拒绝而跳过或无法启动；这不代表 Python 数学后端失败。

## 数学约定

全项目统一使用：

```text
I = G V + Ihis
```

不要写成 `I + Ihis = G V`，也不要把 `Ihis` 符号反过来。

一条 `from -> to` 的支路定义为：

```text
i = G * (V_from - V_to) + Ihis
```

节点 stamp 规则：

```text
G[from, from] += G
G[to, to]     += G
G[from, to]   -= G
G[to, from]   -= G

Ihis[from] += Ihis
Ihis[to]   -= Ihis
```

节点消去必须使用 Schur complement，不能简单删除 internal node 的行列。

对于分块：

```text
I_e = G_ee V_e + G_ei V_i + Ihis_e
I_i = G_ie V_e + G_ii V_i + Ihis_i
```

内部节点无外部注入：

```text
I_i = 0
```

所以：

```text
V_i = -inv(G_ii) G_ie V_e - inv(G_ii) Ihis_i
```

消去后：

```text
G_red = G_ee - G_ei inv(G_ii) G_ie
Ihis_red = Ihis_e - G_ei inv(G_ii) Ihis_i
```

支路观测电流消去公式：

```text
i_obs = C_full V_full + d
C_full = [C_e C_i]

i_obs_red = (C_e + C_i K_v) V_e + (C_i K_h + d)
```

其中：

```text
K_v = -inv(G_ii) G_ie
K_h = -inv(G_ii) Ihis_i
```

### 逐节点 Schur 消去实现说明

数学原理仍然是 Schur complement，但后端实现不要显式求 `inv(G_ii)`。原因是：当内部节点较多且矩阵元素是符号表达式时，`G_ii.inv()` 会生成完整符号逆矩阵，中间表达式急剧膨胀，容易导致 `/reduce-system` 超时。

当前 `elimination.py` 使用逐节点消去。对某个内部节点 `k`，设其余节点为 `r`：

```text
0 = G_kk V_k + G_kr V_r + Ihis_k
V_k = -(G_kr V_r + Ihis_k) / G_kk

G_rr_new = G_rr - G_rk G_kr / G_kk
Ihis_r_new = Ihis_r - G_rk Ihis_k / G_kk
```

连续消去所有内部节点，与一次性分块 Schur complement 等价：

```text
G_red = G_ee - G_ei inv(G_ii) G_ie
Ihis_red = Ihis_e - G_ei inv(G_ii) Ihis_i
```

只是计算顺序不同：逐节点消去避免构造完整 `inv(G_ii)`，更适合符号矩阵。

内部节点恢复公式仍然保留。实现中每消去一个节点，就记录：

```text
V_k = coeffs * V_remaining + source
```

最后反向代回，得到：

```text
V_internal = K_v V_external + K_h
```

重要：不要为了“看起来更数学”把实现改回 `G_ii.inv()`。如需优化表达式美观，应在小范围输出阶段做轻量处理，不要在核心消去循环里对整矩阵反复 `simplify/cancel`。

## 后端职责

后端 Python 是数学真值来源。以下功能尽量交给后端：

- 矩阵节点消去
- 符号表达式化简
- 支路观测电流消去
- 矩阵项合并，如 `G11 + G11 -> 2*G11`，`-G12 + G12 -> 0`
- 自定义 N 节点黑盒的支路电流 observer 一致性校验

相关文件：

- `elimination.py`
- `observers.py`
- `reduce_api.py`
- `nodal_tool/blackbox_validation.py`
- `tests/test_elimination_rl.py`
- `tests/test_complex_internal_elimination.py`
- `tests/test_blackbox_observer_validation.py`

每次修改消元逻辑后，至少运行：

```bash
python -m unittest tests.test_elimination_rl -v
```

RL 串联测试是核心回归测试，不要删除。`tests/test_complex_internal_elimination.py` 覆盖了 `YBox3 + UCM_block + P/N 内部节点` 的复杂符号消去场景，防止逐节点 Schur 优化被回退后再次出现 240 秒 timeout。

`reduce_api.py` 的 `_clean_expr` / `_clean_observer_expr` 当前刻意保持轻量字符串输出。不要在 API 输出阶段对大表达式做全量 `sp.simplify()`、`sp.cancel()`、`sp.expand()`，否则可能出现“消去已完成，但格式化输出卡死”的问题。

## 前端职责

前端负责：

- 画布交互
- 元件拖拽、缩放、旋转、框选、复制粘贴、删除
- 多画布管理
- 导入导出
- 中英文切换
- 公式渲染
- 把当前电路整理成完整的 `G_full`、`Ihis_full`、节点顺序、观测电流，提交给后端

前端不应该自己实现复杂节点消去数学。

## 画布和交互经验

### 端点、根部、元件拖拽

普通支路、变压器、自定义 N 节点黑盒、YBox 都应该遵守同一类交互直觉：

- 元件整体可以拖动
- 外端点可以自由拖动
- 根部小圆圈可以自由拖动
- 拖动整个元件时，外端点和根部应该跟着元件一起移动
- 对齐按钮只在点击时修改坐标，不应该在之后持续约束用户拖动

自定义 N 节点黑盒中，端口显示位置只是图层布局，不改变数学端口顺序。数学端口顺序由用户输入的矩阵定义。

### 自定义黑盒支路电流

只给出黑盒 `G/Ihis` 时，物理支路分解不唯一，不能自动推断真实支路电流。支路电流必须由用户显式定义 observer：

```text
i_k = G_k * (V_from - V_to) + Ihis_k
```

这些 observers 只用于支路电流输出和一致性校验，不能替代黑盒主节点方程。主节点方程永远使用用户输入的 `G/Ihis`。

一致性校验由后端 `validate_blackbox_observers` 完成：

```text
Delta_G = simplify(G_bb - G_obs)
Delta_Ihis = simplify(Ihis_bb - Ihis_obs)
```

校验状态：

- `floating`：没有定义 observers，支路电流不可观测/悬空
- `matched`：observers 完整复现 `G/Ihis`
- `mismatch`：observers 不完整或不一致，但节点方程仍使用用户输入的 `G/Ihis`

如果 observer 使用了 `G/Ihis` 中没有出现过的符号，给 warning，不阻止运行。

### 自动对齐按钮

端口对齐拆成两个动作更清楚：

- `端口根部对齐`：让每个端口根部和对应外端点连线变直
- `同侧端点对齐`：读取当前同侧端点坐标，左/右侧统一 X，顶/底侧统一 Y

注意：同侧端点对齐要基于当前屏幕坐标，不要优先读取旧的 `axis` 元数据，否则会导致端点跳到奇怪位置。

### 节点命名

节点名是用户语义，不应随便自动重排。

例子：

- `N1, N2, N3` 中 `N2` 被消去后，打包时 `N3` 仍应叫 `N3`
- 独立节点 `N4, N5` 不应因为别的元件打包/拆包而被改名
- 导入外部电路如果重名，可以用颜色区分，后续由用户自己决定是否改名

不要为了“连续编号好看”而破坏用户命名。

## YBox / 打包拆包经验

YBox 是被选中电路的打包结果。

打包时：

- 外部节点成为 YBox 暴露端口
- 内部节点被消去，不再作为端口暴露
- 内部节点电压恢复公式需要保留并显示
- 支路观测电流信息需要保留
- 支路观测电流的 `from/to` 显示方向要保留原始节点名，即使某个端点后来作为内部节点被消去；`P1/P2/...` 只应作为 YBox 内部数学端口顺序，不应替代用户看到的原始方向语义
- 打包后原始支路图形不存在时，支路电流 hover 只能做轻量语义高亮：在黑盒内部按观测电流的原始 `from/to` 标签画悬浮节点和平滑连接线，不要假装恢复完整原电路拓扑
- 原始电路信息需要保存，以便拆包恢复

拆包时：

- 只能替换当前选中的 YBox
- 不能影响画布上其他无关元件
- 要尽量恢复原始节点名、端口布局、支路观测信息

### Switch cases and provenance highlighting

元件支持开关工况：

- 普通二节点支路的工况字段是 `g` / `ihis`。
- 变压器和自定义 N 节点黑盒的工况字段是 `gMatrix` / `ihisVector`。
- `branchValue(branch, field)` 会读取当前 active case；做数学组装时不要直接读 `branch.g`、`branch.ihis`、`branch.gMatrix`、`branch.ihisVector`。
- 打包黑盒 `branch.packageOriginal` 不允许外层 switch case。它的工况由 `packageOriginal.branches` 中的内部支路决定。
- 打包黑盒编辑器中的“内部工况”下拉框修改内部支路 `activeSwitchCase`，随后调用本地 SymPy 重新计算打包后的 `G/Ihis` 和 observers。

公式高亮使用 hidden provenance tags：

- 前端组装全局矩阵时，每个 stamp 项保留来源支路 id。
- 给后端的普通字段仍是 `G_full` / `Ihis_full`，用于普通显示。
- 同时发送 `G_full_tagged` / `Ihis_full_tagged`，其中 symbol 会带隐藏来源后缀，例如 `R1__bbsrc_B6`。
- `reduce_api.py` 会单独对 tagged 矩阵跑一次节点消去，并返回 `G_red_tagged`、`Ihis_red_tagged`、`K_v_tagged`、`K_h_tagged`。
- 前端显示时剥掉 `__bbsrc_*`，但根据 tag 决定哪个 symbol 上色。
- 当 reduced view 中存在当前高亮支路来源 tag 时，使用 tagged expression 作为渲染骨架，而不是用普通 simplified expression 去猜测来源。这样同名 symbol 不会串色。
- 代价是：开启公式高亮时，消去版本可能比普通显示更展开。界面里已有 warning 说明这一点。
- 不要恢复“在整个 matrix cell 里找同名 symbol 作为 fallback”的方案；它会导致同一格内重名 symbol 全部被高亮。

打包黑盒高亮：

- 如果用户高亮打包黑盒，黑盒内部所有原始支路 id 都算作这个黑盒的来源。
- 这由 `branchHighlightSourceIds(branch)` 负责。
- 不要只匹配外层 YBox 的 branch id，否则打包后的内部 symbol 无法被高亮。

布局注意：

- 小窗口下 `@media (max-width: 920px)` 会把主区域从两列改成纵向布局，并允许页面滚动。
- 不要让 `.app` 在窄屏继续强制 `height: 100vh; overflow: hidden;`，否则 output 的长矩阵会覆盖 panel。

## 导入导出经验

用户期望导出是“生成一个 JSON 文件并保存”，导入是“选择 JSON 文件并恢复/追加电路”。

注意事项：

- 保存文件名可修改
- 覆盖已有 JSON 文件不能变成空文件
- 导入后要能看到元件确实出现
- 多画布状态、节点名、端口布局、YBox 原始信息都要尽量保存

后续大功能更新前，建议用户先导出当前电路。更新后再导入一般可行，但要注意数据结构兼容。

## 多画布经验

多画布应该互相隔离：

- 每个画布有自己的元件、导线、节点顺序、内部节点标记、视角和历史
- 画布之间可以复制粘贴
- 切换画布不能串状态
- 新增功能时要记得考虑多画布状态是否需要保存

画布必须可以：

- 新建
- 删除
- 切换

## 中英文切换经验

项目默认中文，按钮可以切换英文。

后续新增任何可见文案，都要同步加入 i18n 映射，包括：

- 工具栏按钮
- 右侧面板标题
- 表单标签
- 占位提示
- 输出窗口 tab
- 弹窗
- 错误提示
- 动态生成的公式说明、加载提示、失败提示
- 异步返回后插入页面的 HTML 文案

之前出现过遗漏，新增功能时要主动检查。尤其注意：如果输出内容被缓存，缓存 key 必须包含当前语言，或者渲染函数必须在插入 HTML 时直接使用 `tr()` / 英文分支，否则切换到英文后可能继续显示旧的中文缓存。

## 公式显示经验

公式显示要尽量接近数学排版，而不是普通字符串打印。

注意事项：

- 矩阵形式要清晰
- 支路电流要用 display-style 公式
- 输出窗口可能很宽，要支持横向滚动
- 横向滚动条不能在松开鼠标后自动跳回
- 输出区域可调整高度，也可以全屏/单独查看

## 缓存和重新计算经验

节点消去、支路电流消去这些后端计算不应该因为普通点击画布就重新计算。

重新计算应该只发生在：

- 电路结构变化
- G / Ihis / 节点名 / 观测电流变化
- 内部节点集合变化
- 节点顺序变化

选择状态、普通点击、选中元件、拖动画布视角不应该触发数学重新计算。

## UI 设计经验

当前工具更像工程绘图软件，不是营销页面。

界面应保持：

- 功能直接
- 信息密度合理
- 工具按钮清楚
- 尽量使用图标按钮
- 选中面板可编辑
- 画布可缩放和平移
- 大电路时画布能扩展
- 下方输出窗口可调整比例

不要把说明文字塞满界面。功能应该靠控件本身表达。

## 已经踩过的坑

- 节点消去不能删除行列，必须 Schur complement
- `Ihis_red` 符号容易写反
- 支路观测电流消去不能忽略内部节点恢复公式
- 前端自己拼公式容易错，应交给后端符号化简
- 点击画布不应触发数学重算
- 输入框、按钮、拖拽事件容易互相吞事件，要区分 pointerdown / click / drag
- 端口布局的旧 `axis` 元数据可能覆盖自由拖动坐标
- 打包拆包不能影响无关元件
- 自动节点编号不能破坏用户已改名节点
- 新功能容易漏英文文案
- 多画布功能容易漏保存当前画布状态
- 输出窗口和画布比例调整容易把画布挤没

## 后续开发建议

每次做功能前，先判断属于哪一层：

- 数学：后端 Python
- 显示：前端公式渲染
- 交互：前端事件和状态
- 数据：导入导出 / 多画布 / 打包拆包

做完后至少检查：

- 当前功能是否正常
- 节点方程是否仍然正确
- 支路电流是否同步更新
- 中英文是否完整
- 导入导出是否保留新字段
- 多画布切换是否隔离
- 撤销/重做是否可用

前端修改后建议做脚本语法检查。数学逻辑修改后必须跑单元测试。

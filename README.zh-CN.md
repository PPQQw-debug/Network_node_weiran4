# Branch Builder 中文说明

[English README](README.md) | 中文

Branch Builder 是一个本地运行的浏览器工具，用于搭建和分析网络节点电路模型。它支持绘制支路、变压器、自定义黑盒和 YBox 打包元件，并生成完整节点方程、节点消去后的等效方程、内部节点电压恢复公式和支路观测电流表达式。

## 主要功能

- 在画布中绘制、拖拽、缩放、旋转和编辑电路元件。
- 支持二节点支路、单相变压器、自定义 N 节点黑盒和 YBox。
- 输出统一形式的节点方程：`I = G V + Ihis`。
- 使用 Schur complement 对内部节点进行消去。
- 显示内部节点电压恢复公式。
- 为黑盒元件定义支路观测电流，并进行一致性校验。
- 导入和导出电路 JSON 文件。
- 将电路保存到项目目录下的 `exports/` 文件夹。
- 支持中英文界面切换。

## 环境要求

- Python 3.10 或更新版本。
- Chrome、Edge、Firefox 等现代浏览器。
- 第一次自动安装 `sympy` 时需要网络；如果提前准备了离线 `wheels/` 文件夹，则可以离线运行。

`local_server.py` 会在启动时自动检查 `sympy` 是否已经安装。如果缺少 `sympy`，它会先尝试使用项目目录下的 `wheels/` 离线包安装；如果没有 `wheels/` 文件夹，则通过 pip 在线安装。

## 快速启动

在项目文件夹中打开终端：

```powershell
cd path\to\network_node
```

启动本地服务：

```powershell
python local_server.py
```

然后在浏览器中打开：

```text
http://127.0.0.1:4177/
```

使用过程中请保持终端窗口不要关闭。

## 在另一台电脑上运行

请把整个项目文件夹复制到另一台电脑，不要只复制 `index.html`。这个项目的后端计算、保存、读取和黑盒校验都依赖本地服务。

复制完成后，在另一台电脑上进入项目目录：

```powershell
cd path\to\network_node
```

启动：

```powershell
python local_server.py
```

打开：

```text
http://127.0.0.1:4177/
```

如果那台电脑可以联网，服务会自动下载并安装 `sympy`。

## 离线演示准备

如果演示电脑没有网络，请先在有网络的电脑上运行：

```powershell
python -m pip download sympy -d wheels
```

然后把生成的 `wheels/` 文件夹和整个项目一起复制到演示电脑。

在演示电脑上直接运行：

```powershell
python local_server.py
```

服务会自动从 `wheels/` 文件夹安装 `sympy`。

也可以手动安装：

```powershell
python -m pip install --no-index --find-links wheels sympy
```

## 可选 Node.js 启动方式

如果电脑安装了 Node.js，也可以运行：

```powershell
node server.js
```

不过第一次使用时更推荐 Python 服务，因为 `local_server.py` 可以自动检查并安装 Python 后端计算需要的 `sympy`。

## 项目结构

- `index.html`：主浏览器界面。
- `local_server.py`：推荐使用的本地服务。
- `server.js`：可选 Node.js 本地服务。
- `reduce_api.py`：节点方程消元 API。
- `elimination.py`：符号节点消去逻辑。
- `observers.py`：支路观测电流消去逻辑。
- `blackbox_validation_api.py`：黑盒观测电流校验 API。
- `nodal_tool/blackbox_validation.py`：黑盒观测电流一致性检查。
- `exports/`：保存的电路 JSON 文件。
- `tests/`：回归测试。

## 运行测试

运行 Python 测试：

```powershell
python -m unittest discover tests -v
```

运行前端公式格式化测试：

```powershell
node tests\frontend_math_formatter_cases.mjs
```

## 演示注意事项

- 先启动服务，再打开网页。
- 请使用 `http://127.0.0.1:4177/`，不要直接双击打开 `index.html`。
- 如果需要保存导出的电路，请确保项目文件夹可写。
- 如果 `4177` 端口被占用，可以换一个端口：

```powershell
$env:PORT=4180
python local_server.py
```

然后打开：

```text
http://127.0.0.1:4180/
```

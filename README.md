# Branch Builder / 节点网络电路绘图与消元工具

English | [中文说明](README.zh-CN.md)

Branch Builder is a local browser-based tool for building and analyzing network-node circuit models. It lets you draw branches, transformers, custom black boxes, and packaged Y-box components, then generate full and reduced nodal equations.

Branch Builder 是一个本地运行的浏览器工具，用于绘制和分析网络节点电路模型。你可以搭建支路、变压器、自定义黑盒和打包后的 YBox，并输出完整节点方程和节点消去后的等效方程。

## Features / 功能

- Draw and edit circuit branches on a canvas.
- Build two-node branches, single-phase transformers, custom N-node black boxes, and Y-box packages.
- Generate full nodal equations in the form `I = G V + Ihis`.
- Reduce internal nodes with Schur complement logic.
- Display internal-node voltage recovery formulas.
- Define and validate branch-current observers for black-box components.
- Import and export circuit JSON files.
- Save exported circuits into the local `exports/` folder.
- Switch between Chinese and English UI text.

中文功能概览：

- 在画布上绘制、拖拽和编辑电路元件。
- 支持二节点支路、单相变压器、自定义 N 节点黑盒和 YBox 打包元件。
- 生成统一形式的完整节点方程：`I = G V + Ihis`。
- 使用 Schur complement 对内部节点进行消去。
- 显示内部节点电压恢复公式。
- 为黑盒元件定义和校验支路观测电流。
- 导入和导出电路 JSON 文件。
- 将导出的电路保存到本地 `exports/` 文件夹。
- 支持中英文界面切换。

## Requirements / 环境要求

- Python 3.10 or newer.
- A modern browser such as Chrome, Edge, or Firefox.
- Internet access the first time the server needs to install `sympy`, unless you prepare an offline `wheels/` folder.

需要：

- Python 3.10 或更新版本。
- Chrome、Edge、Firefox 等现代浏览器。
- 第一次自动安装 `sympy` 时需要网络；如果提前准备了离线 `wheels/` 文件夹，则可以离线运行。

`local_server.py` automatically checks whether `sympy` is installed. If it is missing, the server will try to install it with pip before starting. If a local `wheels/` folder exists, the server will try that folder first.

`local_server.py` 会在启动时自动检查是否安装了 `sympy`。如果没有安装，它会先尝试使用项目目录下的 `wheels/` 离线包；如果没有离线包，则使用 pip 在线安装。

## Quick Start / 快速启动

Open a terminal in this project folder and run:

在项目文件夹中打开终端，运行：

```powershell
python local_server.py
```

Then open this URL in your browser:

然后在浏览器中打开：

```text
http://127.0.0.1:4177/
```

Keep the terminal window open while using the app.

使用过程中请保持终端窗口不要关闭。

## Running on Another Computer / 在另一台电脑运行

Copy the whole project folder to the other computer. Do not copy only `index.html`, because the app needs the local server for symbolic reduction, validation, saving, and loading.

请把整个项目文件夹复制到另一台电脑，不要只复制 `index.html`。因为符号消元、黑盒校验、保存和读取都需要本地服务。

On the other computer:

在另一台电脑上：

```powershell
cd path\to\network_node
python local_server.py
```

Then open:

然后打开：

```text
http://127.0.0.1:4177/
```

If the computer has internet access, `local_server.py` will automatically install `sympy` if needed.

如果电脑可以联网，`local_server.py` 会在需要时自动安装 `sympy`。

## Offline Setup / 离线准备

If the demonstration computer does not have internet access, prepare the dependency on a computer that does:

如果演示电脑没有网络，请先在有网络的电脑上准备离线依赖包：

```powershell
python -m pip download sympy -d wheels
```

Copy the generated `wheels/` folder together with the project. On the offline computer, simply run:

把生成的 `wheels/` 文件夹和项目一起复制到离线电脑。然后直接运行：

```powershell
python local_server.py
```

The server will detect the local `wheels/` folder and install `sympy` from it automatically.

服务会自动检测本地 `wheels/` 文件夹，并从里面安装 `sympy`。

You can also install it manually:

也可以手动安装：

```powershell
python -m pip install --no-index --find-links wheels sympy
```

## Optional Node Server / 可选 Node 服务

There is also a Node.js server:

项目也提供 Node.js 版本的本地服务：

```powershell
node server.js
```

The Python server is recommended for first-time users because it can automatically check and install the Python dependency used by the calculation backend.

第一次使用时建议运行 Python 服务，因为它可以自动检查并安装后端计算所需的 Python 依赖。

## Project Structure / 项目结构

- `index.html` - main browser interface / 主浏览器界面。
- `local_server.py` - recommended local server / 推荐使用的本地服务。
- `server.js` - optional Node.js local server / 可选 Node.js 本地服务。
- `reduce_api.py` - backend API wrapper for equation reduction / 节点方程消元 API。
- `elimination.py` - symbolic node-elimination logic / 符号节点消去逻辑。
- `observers.py` - branch-current observer reduction logic / 支路观测电流消去逻辑。
- `blackbox_validation_api.py` - API wrapper for black-box observer validation / 黑盒观测电流校验 API。
- `nodal_tool/blackbox_validation.py` - black-box observer consistency checks / 黑盒观测电流一致性检查。
- `exports/` - saved circuit JSON files / 保存的电路 JSON 文件。
- `tests/` - regression tests / 回归测试。

## Running Tests / 运行测试

Run the core Python tests with:

运行 Python 测试：

```powershell
python -m unittest discover tests -v
```

Run the frontend formatter cases with:

运行前端公式格式化测试：

```powershell
node tests\frontend_math_formatter_cases.mjs
```

## Notes for Demonstrations / 演示注意事项

- Start the server before opening the page.
- Use `http://127.0.0.1:4177/`, not a direct `file://` path.
- Keep the project folder writable if you want to save exported circuits.
- If port `4177` is already in use, start with a different port:

中文提醒：

- 先启动服务，再打开网页。
- 请使用 `http://127.0.0.1:4177/`，不要直接双击打开 `index.html`。
- 如果需要保存导出的电路，请确保项目文件夹可写。
- 如果 `4177` 端口被占用，可以换一个端口：

```powershell
$env:PORT=4180
python local_server.py
```

Then open:

然后打开：

```text
http://127.0.0.1:4180/
```

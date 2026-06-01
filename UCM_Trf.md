```python
import sympy as sp
from collections import defaultdict

sp.init_printing(use_unicode=True)

# =========================================================
# 1. 并查集：用于节点合并
# =========================================================
class UnionFind:
    def __init__(self):
        self.parent = {}

    def add(self, x):
        if x not in self.parent:
            self.parent[x] = x

    def find(self, x):
        self.add(x)
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    def union(self, a, b):
        self.add(a)
        self.add(b)
        ra = self.find(a)
        rb = self.find(b)
        if ra != rb:
            self.parent[rb] = ra

```


```python
# =========================================================
# 2. 子网络对象
#    I_local = G_local * V_local + Ihis_local
# =========================================================
class SubNetwork:
    def __init__(self, name, local_nodes, G_local, Ihis_local):
        """
        name        : 子网络名称，例如 'net1'
        local_nodes : 局部节点名列表，例如 ['a', 'm']
        G_local     : sympy Matrix
        Ihis_local  : sympy Matrix (列向量)
        """
        self.name = name
        self.local_nodes = list(local_nodes)
        self.G_local = sp.Matrix(G_local)
        self.Ihis_local = sp.Matrix(Ihis_local)

        n = len(self.local_nodes)
        if self.G_local.shape != (n, n):
            raise ValueError(f"{name}: G_local shape must be {(n, n)}, got {self.G_local.shape}")
        if self.Ihis_local.shape not in [(n, 1), (n,)]:
            raise ValueError(f"{name}: Ihis_local shape must be {(n,1)} or {(n,)}, got {self.Ihis_local.shape}")

        if self.Ihis_local.shape == (n,):
            self.Ihis_local = sp.Matrix(n, 1, list(self.Ihis_local))

    def global_node_labels(self):
        return [f"{self.name}.{nd}" for nd in self.local_nodes]
```


```python
# =========================================================
# 3. 网络装配器
# =========================================================
class NetworkAssembler:
    def __init__(self):
        self.subnets = []
        self.uf = UnionFind()
        self.all_nodes = set()
        self.ground_name = "GND"

    def add_subnetwork(self, subnet: SubNetwork):
        self.subnets.append(subnet)
        for gnode in subnet.global_node_labels():
            self.uf.add(gnode)
            self.all_nodes.add(gnode)

    def connect(self, node1, node2):
        """
        短接两个节点，例如:
        connect('net1.m', 'net2.x')
        """
        self.uf.union(node1, node2)
        self.all_nodes.add(node1)
        self.all_nodes.add(node2)

    def ground(self, node):
        """
        节点接地，例如:
        ground('net1.b')
        """
        self.uf.union(node, self.ground_name)
        self.all_nodes.add(node)
        self.all_nodes.add(self.ground_name)

    def build_global_mapping(self):
        """
        返回:
        rep_to_gid      : 等价类代表元 -> 全局节点编号 (地节点不编号)
        node_to_gid     : 每个原始节点 -> 全局节点编号(None表示地)
        global_nodes    : 全局非地节点名称（等价类代表元）
        """
        # 先确保 ground 在并查集中
        self.uf.add(self.ground_name)

        # 收集等价类
        classes = defaultdict(list)
        for node in self.all_nodes:
            root = self.uf.find(node)
            classes[root].append(node)

        gnd_root = self.uf.find(self.ground_name)

        rep_to_gid = {}
        global_nodes = []

        gid = 0
        for root in classes:
            if root == gnd_root:
                continue
            rep_to_gid[root] = gid
            global_nodes.append(root)
            gid += 1

        node_to_gid = {}
        for node in self.all_nodes:
            root = self.uf.find(node)
            if root == gnd_root:
                node_to_gid[node] = None
            else:
                node_to_gid[node] = rep_to_gid[root]

        return rep_to_gid, node_to_gid, global_nodes

    def assemble(self, verbose=True):
        """
        自动装配总系统:
            I_global = G_global * V_global + Ihis_global
        对无外部注入的节点，通常令 I_global = 0
        """
        rep_to_gid, node_to_gid, global_nodes = self.build_global_mapping()
        N = len(global_nodes)

        G_global = sp.zeros(N, N)
        Ihis_global = sp.zeros(N, 1)

        stamp_log = []

        for subnet in self.subnets:
            local_gnodes = subnet.global_node_labels()
            n = len(local_gnodes)

            for p in range(n):
                gp = node_to_gid[local_gnodes[p]]

                # 装配 Ihis
                if gp is not None:
                    Ihis_global[gp, 0] += subnet.Ihis_local[p, 0]
                    stamp_log.append(
                        f"Ihis: {subnet.name}[{p}] ({local_gnodes[p]}) -> global[{gp}] += {sp.sstr(subnet.Ihis_local[p,0])}"
                    )

                # 装配 G
                for q in range(n):
                    gq = node_to_gid[local_gnodes[q]]
                    if gp is not None and gq is not None:
                        G_global[gp, gq] += subnet.G_local[p, q]
                        stamp_log.append(
                            f"G: {subnet.name}[{p},{q}] ({local_gnodes[p]},{local_gnodes[q]}) "
                            f"-> G_global[{gp},{gq}] += {sp.sstr(subnet.G_local[p,q])}"
                        )

        # 全局电压变量
        V_symbols = sp.Matrix([sp.Symbol(f"V_{i}") for i in range(N)])
        I_symbols = sp.Matrix([sp.Symbol(f"I_{i}") for i in range(N)])

        if verbose:
            print("========== Global node classes ==========")
            for node in sorted(self.all_nodes):
                root = self.uf.find(node)
                gid = node_to_gid[node]
                if gid is None:
                    print(f"{node:15s} -> class [{root}] -> GND")
                else:
                    print(f"{node:15s} -> class [{root}] -> global node {gid}")

            print("\n========== Stamping log ==========")
            for item in stamp_log:
                print(item)

            print("\n========== Global equation ==========")
            print("I_global = G_global * V_global + Ihis_global")

        return {
            "G_global": sp.simplify(G_global),
            "Ihis_global": sp.simplify(Ihis_global),
            "V_global": V_symbols,
            "I_global": I_symbols,
            "global_nodes": global_nodes,
            "node_to_gid": node_to_gid,
            "stamp_log": stamp_log
        }

    def derive_kcl_equations(self, assembled, zero_injection_nodes=None):
        """
        生成 KCL 方程。
        默认所有全局节点都设为零注入：
            0 = G V + Ihis
        """
        G = assembled["G_global"]
        V = assembled["V_global"]
        Ihis = assembled["Ihis_global"]
        N = G.shape[0]

        if zero_injection_nodes is None:
            zero_injection_nodes = list(range(N))

        eqs = []
        expr = G * V + Ihis

        for i in range(N):
            if i in zero_injection_nodes:
                eqs.append(sp.Eq(0, sp.simplify(expr[i, 0])))
            else:
                I_i = assembled["I_global"][i, 0]
                eqs.append(sp.Eq(I_i, sp.simplify(expr[i, 0])))

        return eqs, expr

    def eliminate_internal_nodes(self, assembled, external_node_ids):
        """
        对内部节点做 Schur complement 消元。
        输入:
            external_node_ids : 保留的外部节点编号列表
        输出:
            I_e = G_eq * V_e + Ihis_eq
        """
        G = sp.Matrix(assembled["G_global"])
        J = sp.Matrix(assembled["Ihis_global"])
        N = G.shape[0]

        ext = list(external_node_ids)
        internal = [i for i in range(N) if i not in ext]

        if len(internal) == 0:
            return {
                "G_eq": G,
                "Ihis_eq": J,
                "external_ids": ext,
                "internal_ids": internal
            }

        G_ee = G.extract(ext, ext)
        G_em = G.extract(ext, internal)
        G_me = G.extract(internal, ext)
        G_mm = G.extract(internal, internal)

        J_e = J.extract(ext, [0])
        J_m = J.extract(internal, [0])

        G_eq = sp.simplify(G_ee - G_em * G_mm.inv() * G_me)
        Ihis_eq = sp.simplify(J_e - G_em * G_mm.inv() * J_m)

        return {
            "G_eq": G_eq,
            "Ihis_eq": Ihis_eq,
            "external_ids": ext,
            "internal_ids": internal
        }

```


```python
# =========================================================
# 4. 辅助打印
# =========================================================
from IPython.display import display, Math
def sanitize(name):
    return name.replace(".", "_")

def pretty_display_system_split(assembled):
    G = assembled["G_global"]
    V = assembled["V_global"]
    I = assembled["I_global"]
    Ihis = assembled["Ihis_global"]
    global_nodes = assembled["global_nodes"]

    subs = {}
    for i, node in enumerate(global_nodes):
        name_clean = sanitize(node)
        subs[V[i]] = sp.Symbol(f"V_{{{name_clean}}}")
        subs[I[i]] = sp.Symbol(f"I_{{{name_clean}}}")

    V_named = V.subs(subs)
    I_named = I.subs(subs)
    Ihis_named = Ihis.subs(subs)

    GV = sp.MatMul(G, V_named, evaluate=False)
    
    sp.init_printing(use_latex='mathjax')
    
    rhs = sp.Add(GV, Ihis_named, evaluate=False)   # 保持 GV + Ihis_named 不变
    eqn = sp.Eq(I_named, rhs, evaluate=False)      # 保持等式结构不自动改写
    
    # 用 latex(order='none') 来禁止排序
    tex = sp.latex(eqn, order='none')   

    print("\n========== System Equation ==========\n")
    display(Math(tex))   # 渲染为一行的 LaTeX

def pretty_display_reduced_system(assembled, reduced):
    """
    漂亮显示消元后的外部系统：
        I_ext = G_eq * V_ext + Ihis_eq
    其中外部节点名字来自 assembled["global_nodes"] 和 reduced["external_ids"]
    """
    G_eq = reduced["G_eq"]
    Ihis_eq = reduced["Ihis_eq"]
    ext_ids = reduced["external_ids"]
    global_nodes = assembled["global_nodes"]

    # 构造外部电压/电流变量（按 external_ids 顺序）
    V_ext = sp.Matrix([
        sp.Symbol(f"V_{{{sanitize(global_nodes[i])}}}")
        for i in ext_ids
    ])

    I_ext = sp.Matrix([
        sp.Symbol(f"I_{{{sanitize(global_nodes[i])}}}")
        for i in ext_ids
    ])

    # 用未展开的矩阵乘法和加法
    GV = sp.MatMul(G_eq, V_ext, evaluate=False)
    rhs = sp.MatAdd(GV, Ihis_eq, evaluate=False)

    # 组装成单行 LaTeX
    I_latex = sp.latex(I_ext)
    G_latex = sp.latex(G_eq)
    V_latex = sp.latex(V_ext)
    Ihis_latex = sp.latex(Ihis_eq)

    expr = f"{I_latex} = {G_latex} {V_latex} + {Ihis_latex}"

    print("\n========== Reduced External System Equation ==========\n")
    display(Math(expr))
    
def pretty_print_matrix(name, M):
    print(f"\n{name} =")
    display(sp.Matrix(M))

def pretty_print_equations(title, eqs):
    print(f"\n{title}")
    for k, eq in enumerate(eqs):
        display(sp.Eq(sp.Symbol(f"eq_{k}"), eq))
```


```python
# =========================================================
# 5. 示例：R 与 L 串联
#    netR: 节点 [a, m]
#    netL: 节点 [m2, b]
#    连接 netR.m == netL.m2
# =========================================================

# -------- 定义符号 --------
R, L, dt = sp.symbols('R L dt', positive=True)
iL_prev, vL_prev, Ihis_L = sp.symbols('iL_prev vL_prev Ihis_L')

G_R = 1 / R
G_L = dt / (2 * L)
#Ihis_L_scalar = iL_prev + G_L * vL_prev
Ihis_L_scalar = Ihis_L

# -------- 子网络 1：电阻 --------
# i = G(v_a - v_m)
G_netR = sp.Matrix([
    [ G_R, -G_R],
    [-G_R,  G_R]
])

Ihis_netR = sp.Matrix([
    [0],
    [0]
])

netR = SubNetwork(
    name="netR",
    local_nodes=["a", "m"],
    G_local=G_netR,
    Ihis_local=Ihis_netR
)

# -------- 子网络 2：电感 --------
# i = G(v_m - v_b) + Ihis
G_netL = sp.Matrix([
    [ G_L, -G_L],
    [-G_L,  G_L]
])

Ihis_netL = sp.Matrix([
    [ Ihis_L_scalar],
    [-Ihis_L_scalar]
])

netL = SubNetwork(
    name="netL",
    local_nodes=["m2", "b"],
    G_local=G_netL,
    Ihis_local=Ihis_netL
)

# -------- 装配 --------
asm = NetworkAssembler()
asm.add_subnetwork(netR)
asm.add_subnetwork(netL)

# 节点连接：R 的 m 与 L 的 m2 相连
asm.connect("netR.m", "netL.m2")

# 这里只是示例，不接地；如果要接地可用：
# asm.ground("netL.b")

assembled = asm.assemble(verbose=True)

pretty_display_system_split(assembled)

pretty_print_matrix("G_global", assembled["G_global"])
pretty_print_matrix("Ihis_global", assembled["Ihis_global"])

eqs, expr = asm.derive_kcl_equations(assembled)
pretty_print_equations("KCL equations (all nodes zero injection)", eqs)
```

    ========== Global node classes ==========
    netL.b          -> class [netL.b] -> global node 0
    netL.m2         -> class [netR.m] -> global node 1
    netR.a          -> class [netR.a] -> global node 2
    netR.m          -> class [netR.m] -> global node 1
    
    ========== Stamping log ==========
    Ihis: netR[0] (netR.a) -> global[2] += 0
    G: netR[0,0] (netR.a,netR.a) -> G_global[2,2] += 1/R
    G: netR[0,1] (netR.a,netR.m) -> G_global[2,1] += -1/R
    Ihis: netR[1] (netR.m) -> global[1] += 0
    G: netR[1,0] (netR.m,netR.a) -> G_global[1,2] += -1/R
    G: netR[1,1] (netR.m,netR.m) -> G_global[1,1] += 1/R
    Ihis: netL[0] (netL.m2) -> global[1] += Ihis_L
    G: netL[0,0] (netL.m2,netL.m2) -> G_global[1,1] += dt/(2*L)
    G: netL[0,1] (netL.m2,netL.b) -> G_global[1,0] += -dt/(2*L)
    Ihis: netL[1] (netL.b) -> global[0] += -Ihis_L
    G: netL[1,0] (netL.b,netL.m2) -> G_global[0,1] += -dt/(2*L)
    G: netL[1,1] (netL.b,netL.b) -> G_global[0,0] += dt/(2*L)
    
    ========== Global equation ==========
    I_global = G_global * V_global + Ihis_global
    
    ========== System Equation ==========
    
    


$\displaystyle \left[\begin{matrix}I_{netL_b}\\I_{netR_m}\\I_{netR_a}\end{matrix}\right] = \left[\begin{matrix}\frac{dt}{2 L} & - \frac{dt}{2 L} & 0\\- \frac{dt}{2 L} & \frac{1}{R} + \frac{dt}{2 L} & - \frac{1}{R}\\0 & - \frac{1}{R} & \frac{1}{R}\end{matrix}\right] \left[\begin{matrix}V_{netL_b}\\V_{netR_m}\\V_{netR_a}\end{matrix}\right] + \left[\begin{matrix}- Ihis_{L}\\Ihis_{L}\\0\end{matrix}\right]$


    
    G_global =
    


$\displaystyle \left[\begin{matrix}\frac{dt}{2 L} & - \frac{dt}{2 L} & 0\\- \frac{dt}{2 L} & \frac{1}{R} + \frac{dt}{2 L} & - \frac{1}{R}\\0 & - \frac{1}{R} & \frac{1}{R}\end{matrix}\right]$


    
    Ihis_global =
    


$\displaystyle \left[\begin{matrix}- Ihis_{L}\\Ihis_{L}\\0\end{matrix}\right]$


    
    KCL equations (all nodes zero injection)
    


$\displaystyle eq_{0} = 0 = \frac{- 2 Ihis_{L} L + V_{0} dt - V_{1} dt}{2 L}$



$\displaystyle eq_{1} = 0 = Ihis_{L} + \frac{V_{1}}{R} - \frac{V_{2}}{R} - \frac{V_{0} dt}{2 L} + \frac{V_{1} dt}{2 L}$



$\displaystyle eq_{2} = 0 = \frac{- V_{1} + V_{2}}{R}$



```python
# -------- 选择外部节点，仅保留端口 a,b，消去内部节点 m --------
# 先看 global_nodes 的顺序
print("\nGlobal node list:")
for idx, nd in enumerate(assembled["global_nodes"]):
    print(idx, "->", nd)

# 通常 a 与 b 为外部，m 为内部
# 这里根据实际 global_nodes 名单手动找
global_nodes = assembled["global_nodes"]
ext_ids = []
for i, nd in enumerate(global_nodes):
    # 等价类代表元名字不一定就是 netR.a/netL.b 本身，但通常会是某个类代表
    # 所以这里按成员节点映射反查更稳妥
    pass

# 手动根据 node_to_gid 找端口编号
gid_a = assembled["node_to_gid"]["netR.a"]
gid_b = assembled["node_to_gid"]["netL.b"]
ext_ids = sorted(list({gid_a, gid_b}))

reduced = asm.eliminate_internal_nodes(assembled, external_node_ids=ext_ids)

pretty_display_reduced_system(assembled, reduced)

pretty_print_matrix("G_eq (external port)", reduced["G_eq"])
pretty_print_matrix("Ihis_eq (external port)", reduced["Ihis_eq"])
```

    
    Global node list:
    0 -> netL.b
    1 -> netR.m
    2 -> netR.a
    
    ========== Reduced External System Equation ==========
    
    


$\displaystyle \left[\begin{matrix}I_{netL_b}\\I_{netR_a}\end{matrix}\right] = \left[\begin{matrix}\frac{dt}{2 L + R dt} & - \frac{dt}{2 L + R dt}\\- \frac{dt}{2 L + R dt} & \frac{dt}{2 L + R dt}\end{matrix}\right] \left[\begin{matrix}V_{netL_b}\\V_{netR_a}\end{matrix}\right] + \left[\begin{matrix}- \frac{2 Ihis_{L} L}{2 L + R dt}\\\frac{2 Ihis_{L} L}{2 L + R dt}\end{matrix}\right]$


    
    G_eq (external port) =
    


$\displaystyle \left[\begin{matrix}\frac{dt}{2 L + R dt} & - \frac{dt}{2 L + R dt}\\- \frac{dt}{2 L + R dt} & \frac{dt}{2 L + R dt}\end{matrix}\right]$


    
    Ihis_eq (external port) =
    


$\displaystyle \left[\begin{matrix}- \frac{2 Ihis_{L} L}{2 L + R dt}\\\frac{2 Ihis_{L} L}{2 L + R dt}\end{matrix}\right]$



```python
import re
import sympy as sp
from IPython.display import display, Markdown

# =========================================================
# 0. 一些辅助函数
# =========================================================

def _extract_identifiers(s: str):
    tokens = set(re.findall(r'[A-Za-z_]\w*', s))
    reserved = {
        'Matrix', 'Integer', 'Float', 'Rational',
        'sin', 'cos', 'tan', 'exp', 'log', 'sqrt',
        'Abs', 'pi', 'E', 'I', 'oo'
    }
    return sorted(t for t in tokens if t not in reserved)


def _sympify_matrix_input(s: str, symbol_table: dict):
    ids = _extract_identifiers(s)
    for name in ids:
        if name not in symbol_table:
            symbol_table[name] = sp.Symbol(name)

    expr = sp.sympify(s, locals=symbol_table)

    try:
        M = sp.Matrix(expr)
    except Exception as e:
        raise ValueError(f"无法把输入转换成 Matrix：{s}\n原错误：{e}")

    return M


def _full_node_label(net_name: str, local_node: str):
    return f"{net_name}.{local_node}"


def _alias_node_label(net_name: str, local_node: str):
    return f"{net_name}{local_node}"


def _build_node_alias_map(subnets):
    alias_map = {}
    for subnet in subnets:
        for local_node in subnet.local_nodes:
            full = _full_node_label(subnet.name, local_node)
            alias = _alias_node_label(subnet.name, local_node)

            alias_map[alias] = full
            alias_map[full] = full
            alias_map[f"{subnet.name}_{local_node}"] = full
    return alias_map


def _resolve_node_label(label: str, alias_map: dict):
    label = label.strip()
    if label in alias_map:
        return alias_map[label]
    raise ValueError(
        f"无法识别节点名: {label}\n"
        f"可用形式例如: netR2 或 netR.2"
    )


def _matrix_to_code_string(M):
    return sp.sstr(M.tolist())


def _blank_square_template(n):
    rows = []
    for _ in range(n):
        rows.append("[" + ", ".join(["&"] * n) + "]")
    return "[" + ", ".join(rows) + "]"


def _blank_vector_template(n):
    rows = []
    for _ in range(n):
        rows.append("[&]")
    return "[" + ", ".join(rows) + "]"


def _parse_size_list(size_str, n_sub):
    parts = [x.strip() for x in size_str.split(",") if x.strip()]
    sizes = [int(x) for x in parts]
    if len(sizes) != n_sub:
        raise ValueError(f"你输入了 {len(sizes)} 个尺寸，但子网络个数是 {n_sub}")
    if any(s <= 0 for s in sizes):
        raise ValueError("所有子网络大小必须为正整数")
    return sizes

def reorder_reduced_system(assembled, reduced, new_external_ids):
    """
    按用户指定的新顺序重排 reduced system
    """
    old_ext = reduced["external_ids"]

    if set(new_external_ids) != set(old_ext):
        raise ValueError("新顺序必须和原 external_ids 包含完全相同的节点，只能重排，不能增删。")

    # 建立 old_ext 中位置索引
    pos_map = {gid: i for i, gid in enumerate(old_ext)}
    perm = [pos_map[gid] for gid in new_external_ids]

    G_old = sp.Matrix(reduced["G_eq"])
    J_old = sp.Matrix(reduced["Ihis_eq"])

    G_new = G_old.extract(perm, perm)
    J_new = J_old.extract(perm, [0])

    return {
        "G_eq": G_new,
        "Ihis_eq": J_new,
        "external_ids": list(new_external_ids),
        "internal_ids": reduced["internal_ids"],
    }


def matrix_to_python_string(M, name="M"):
    return f"{name} = sp.Matrix({sp.sstr(sp.Matrix(M).tolist())})"


def print_reduced_python(reduced, name_prefix="reduced"):
    """
    打印 reduced system 的 Python 格式
    """
    G_name = f"G_{name_prefix}"
    J_name = f"Ihis_{name_prefix}"

    print("\n========== Reduced System Python Format ==========\n")
    print(matrix_to_python_string(reduced["G_eq"], G_name))
    print()
    print(matrix_to_python_string(reduced["Ihis_eq"], J_name))


def pretty_display_reduced_system(assembled, reduced):
    """
    漂亮显示 reduced system:
        I_ext = G_eq * V_ext + Ihis_eq
    """
    G_eq = reduced["G_eq"]
    Ihis_eq = reduced["Ihis_eq"]
    ext_ids = reduced["external_ids"]
    global_nodes = assembled["global_nodes"]

    V_ext = sp.Matrix([
        sp.Symbol(f"V_{{{sanitize(global_nodes[i])}}}")
        for i in ext_ids
    ])

    I_ext = sp.Matrix([
        sp.Symbol(f"I_{{{sanitize(global_nodes[i])}}}")
        for i in ext_ids
    ])

    I_latex = sp.latex(I_ext)
    G_latex = sp.latex(G_eq)
    V_latex = sp.latex(V_ext)
    Ihis_latex = sp.latex(Ihis_eq)

    expr = f"{I_latex} = {G_latex} {V_latex} + {Ihis_latex}"

    print("\n========== Reduced External System Equation ==========\n")
    display(Math(expr))

# =========================================================
# 1. 显示函数（沿用你现在的）
# =========================================================

# =========================================================
# 2. 改进版主函数
# =========================================================

def build_network_interactively_v2():
    """
    改进版交互流程：
    1. 输入子网络个数
    2. 一次性输入每个子网络的节点大小，如 2,2,3,2,2
    3. 再逐个输入名字、G、Ihis
    4. G 和 Ihis 输入前打印空模板
    """

    symbol_table = {}
    subnets = []
    generated_code_lines = []

    print("===== 第一步：输入子网络数量 =====")
    n_sub = int(input("请输入子网络个数: ").strip())

    print("\n===== 第二步：一次性输入每个子网络的节点大小 =====")
    size_str = input(
        f"请输入 {n_sub} 个子网络的节点大小，用逗号分隔，例如 2,2,3,2,2 :\n"
    ).strip()
    subnet_sizes = _parse_size_list(size_str, n_sub)

    print("\n你输入的子网络大小为：")
    for i, sz in enumerate(subnet_sizes, start=1):
        print(f"子网络 {i}: {sz} 节点")

    print("\n===== 第三步：逐个输入子网络名字、G、Ihis =====")
    for k in range(n_sub):
        n_nodes = subnet_sizes[k]

        print(f"\n--- 子网络 {k+1} / {n_sub} ---")
        name = input(f"请输入子网络名字（该网络大小为 {n_nodes}）: ").strip()
        if not name:
            raise ValueError("子网络名字不能为空")

        print("\nG 矩阵输入模板：")
        print(_blank_square_template(n_nodes))
        G_str = input("请输入 G 矩阵:\n").strip()

        print("\nIhis 向量输入模板：")
        print(_blank_vector_template(n_nodes))
        Ihis_str = input("请输入 Ihis 向量:\n").strip()

        G_local = _sympify_matrix_input(G_str, symbol_table)
        Ihis_local = _sympify_matrix_input(Ihis_str, symbol_table)

        if G_local.shape != (n_nodes, n_nodes):
            raise ValueError(
                f"{name}: G 的维度应为 {(n_nodes, n_nodes)}，实际为 {G_local.shape}"
            )

        if Ihis_local.shape not in [(n_nodes, 1), (n_nodes,)]:
            raise ValueError(
                f"{name}: Ihis 的维度应为 {(n_nodes,1)} 或 {(n_nodes,)}，实际为 {Ihis_local.shape}"
            )

        if Ihis_local.shape == (n_nodes,):
            Ihis_local = sp.Matrix(n_nodes, 1, list(Ihis_local))

        local_nodes = [str(i + 1) for i in range(n_nodes)]

        subnet = SubNetwork(
            name=name,
            local_nodes=local_nodes,
            G_local=G_local,
            Ihis_local=Ihis_local
        )
        subnets.append(subnet)

        generated_code_lines.append(f"# -------- 子网络：{name} --------")
        generated_code_lines.append(f"G_{name} = sp.Matrix({_matrix_to_code_string(G_local)})")
        generated_code_lines.append(f"Ihis_{name} = sp.Matrix({_matrix_to_code_string(Ihis_local)})")
        generated_code_lines.append(
            f'{name} = SubNetwork(\n'
            f'    name="{name}",\n'
            f'    local_nodes={local_nodes},\n'
            f'    G_local=G_{name},\n'
            f'    Ihis_local=Ihis_{name}\n'
            f')'
        )
        generated_code_lines.append("")

    # 节点别名
    alias_map = _build_node_alias_map(subnets)

    print("\n===== 自动生成的节点别名 =====")
    for subnet in subnets:
        aliases = [f"{_alias_node_label(subnet.name, nd)} <-> {_full_node_label(subnet.name, nd)}"
                   for nd in subnet.local_nodes]
        print(f"{subnet.name}: " + ", ".join(aliases))

    # 装配器
    asm = NetworkAssembler()
    generated_code_lines.append("# -------- 装配 --------")
    generated_code_lines.append("asm = NetworkAssembler()")

    for subnet in subnets:
        asm.add_subnetwork(subnet)
        generated_code_lines.append(f"asm.add_subnetwork({subnet.name})")

    # 输入连接
    print("\n===== 第四步：输入连接关系 =====")
    n_conn = int(input("请输入连接对的个数（每次连接两个节点）: ").strip())

    connection_pairs = []
    for i in range(n_conn):
        print(f"\n--- 连接 {i+1} / {n_conn} ---")
        left = input("请输入第一个节点（例如 netR2 或 netR.2）: ").strip()
        right = input("请输入第二个节点（例如 netL1 或 netL.1）: ").strip()

        full_left = _resolve_node_label(left, alias_map)
        full_right = _resolve_node_label(right, alias_map)

        asm.connect(full_left, full_right)
        connection_pairs.append((full_left, full_right))
        generated_code_lines.append(f'asm.connect("{full_left}", "{full_right}")')

    # 接地
    print("\n===== 第五步：接地（可选） =====")
    ground_input = input(
        "请输入要接地的节点，多个用逗号分隔；若没有，直接回车:\n"
    ).strip()

    grounded_nodes = []
    if ground_input:
        for item in ground_input.split(","):
            full_node = _resolve_node_label(item.strip(), alias_map)
            asm.ground(full_node)
            grounded_nodes.append(full_node)
            generated_code_lines.append(f'asm.ground("{full_node}")')

    # 装配完整系统
    print("\n===== 第六步：装配完整系统 =====")
    assembled = asm.assemble(verbose=True)
    generated_code_lines.append("")
    generated_code_lines.append('assembled = asm.assemble(verbose=True)')
    generated_code_lines.append('pretty_display_system_split(assembled)')

    pretty_display_system_split(assembled)

    # 外部节点消元
    print("\n===== 第七步：选择外部节点，生成 reduced system =====")
    print("可用节点别名如下：")
    for subnet in subnets:
        aliases = [_alias_node_label(subnet.name, nd) for nd in subnet.local_nodes]
        print(f"{subnet.name}: {aliases}")

    ext_input = input(
        "请输入要保留的外部节点，多个用逗号分隔（例如 netR1,netL2）；若保留所有节点，输入 all：\n"
    ).strip()

    if ext_input:
        if ext_input.lower() == "all":
            ext_ids = list(range(len(assembled["global_nodes"])))
            ext_labels = [assembled["global_nodes"][i] for i in ext_ids]
        else:
            ext_labels = [x.strip() for x in ext_input.split(",") if x.strip()]
            ext_ids_raw = [
                assembled["node_to_gid"][_resolve_node_label(lbl, alias_map)]
                for lbl in ext_labels
            ]
            ext_ids = sorted(set(ext_ids_raw))

            if len(ext_ids) < len(ext_ids_raw):
                print("\n注意：你输入的某些节点在前面的连接中已经合并到同一个全局节点，程序已自动去重。")

        reduced = asm.eliminate_internal_nodes(assembled, external_node_ids=ext_ids)

        generated_code_lines.append("")
        generated_code_lines.append("# -------- 选择外部节点并消元 --------")

        if ext_input.lower() == "all":
            generated_code_lines.append(
                'ext_ids = list(range(len(assembled["global_nodes"])))'
            )
        else:
            for lbl in ext_labels:
                full_lbl = _resolve_node_label(lbl, alias_map)
                generated_code_lines.append(
                    f'gid_{sanitize(lbl)} = assembled["node_to_gid"]["{full_lbl}"]'
                )
            generated_code_lines.append(
                f"ext_ids = sorted(list({set(ext_ids)}))"
            )

        generated_code_lines.append(
            "reduced = asm.eliminate_internal_nodes(assembled, external_node_ids=ext_ids)"
        )
        generated_code_lines.append(
            "pretty_display_reduced_system(assembled, reduced)"
        )

        print("\n当前 reduced system 的节点顺序为：")
        for k, gid in enumerate(reduced["external_ids"], start=1):
            print(f"{k}: gid={gid}, node={assembled['global_nodes'][gid]}")

        pretty_display_reduced_system(assembled, reduced)

        # ===== 新增功能 1：是否调整显示顺序 =====
        reorder_ans = input(
            "\n是否要更改 reduced system 中节点显示顺序？(y/n): "
        ).strip().lower()

        if reorder_ans == "y":
            print("\n请输入新的顺序，必须是当前 reduced system 已包含的节点，只能重排，不能增删。")
            print("可以输入 global_nodes 中的真实名字，也可以输入 alias。")
            print("例如: netR1,netL2  或  netR.1,netL.2")

            reorder_input = input("新的节点顺序: ").strip()
            new_labels = [x.strip() for x in reorder_input.split(",") if x.strip()]

            new_external_ids = []
            for lbl in new_labels:
                if lbl in assembled["global_nodes"]:
                    gid = assembled["global_nodes"].index(lbl)
                else:
                    gid = assembled["node_to_gid"][_resolve_node_label(lbl, alias_map)]
                new_external_ids.append(gid)

            reduced = reorder_reduced_system(assembled, reduced, new_external_ids)

            generated_code_lines.append("")
            generated_code_lines.append("# -------- 重排 reduced system 节点顺序 --------")
            generated_code_lines.append(
                f"reduced = reorder_reduced_system(assembled, reduced, {new_external_ids})"
            )
            generated_code_lines.append(
                "pretty_display_reduced_system(assembled, reduced)"
            )

            print("\n重排后的 reduced system：")
            pretty_display_reduced_system(assembled, reduced)

        # ===== 新增功能 2：打印 Python 格式 =====
        print_reduced_python(reduced, name_prefix="reduced_final")

        generated_code_lines.append("")
        generated_code_lines.append("# -------- reduced system 的 Python 格式 --------")
        generated_code_lines.append(
            matrix_to_python_string(reduced["G_eq"], "G_reduced_final")
        )
        generated_code_lines.append(
            matrix_to_python_string(reduced["Ihis_eq"], "Ihis_reduced_final")
        )

    else:
        reduced = None

    # 打印自动生成代码
    print("\n===== 自动生成的可复制代码 =====\n")
    print("\n".join(generated_code_lines))

    return {
        "symbol_table": symbol_table,
        "subnets": subnets,
        "subnet_sizes": subnet_sizes,
        "alias_map": alias_map,
        "connections": connection_pairs,
        "grounded_nodes": grounded_nodes,
        "assembled": assembled,
        "reduced": reduced,
        "generated_code": "\n".join(generated_code_lines),
    }


```


```python
result = build_network_interactively_v2()
```

    ===== 第一步：输入子网络数量 =====
    

    请输入子网络个数:  2
    

    
    ===== 第二步：一次性输入每个子网络的节点大小 =====
    

    请输入 2 个子网络的节点大小，用逗号分隔，例如 2,2,3,2,2 :
     10,5
    

    
    你输入的子网络大小为：
    子网络 1: 10 节点
    子网络 2: 5 节点
    
    ===== 第三步：逐个输入子网络名字、G、Ihis =====
    
    --- 子网络 1 / 2 ---
    

    请输入子网络名字（该网络大小为 10）:  Trf
    

    
    G 矩阵输入模板：
    [[&, &, &, &, &, &, &, &, &, &], [&, &, &, &, &, &, &, &, &, &], [&, &, &, &, &, &, &, &, &, &], [&, &, &, &, &, &, &, &, &, &], [&, &, &, &, &, &, &, &, &, &], [&, &, &, &, &, &, &, &, &, &], [&, &, &, &, &, &, &, &, &, &], [&, &, &, &, &, &, &, &, &, &], [&, &, &, &, &, &, &, &, &, &], [&, &, &, &, &, &, &, &, &, &]]
    

    请输入 G 矩阵:
     [[2*G11 + 2*w1, -G11 - w1, -G11 - w1, G12, 0, -G12, 0, 0, 0, 0], [-G11 - w1, 2*G11 + 2*w1, -G11 - w1, -G12, G12, 0, 0, 0, 0, 0], [-G11 - w1, -G11 - w1, 2*G11 + 2*w1, 0, -G12, G12, 0, 0, 0, 0], [G12, -G12, 0, G22 + Grc + w2, 0, 0, -G22 - w2, -Grc, 0, 0], [0, G12, -G12, 0, G22 + Grc + w2, 0, -G22 - w2, 0, -Grc, 0], [-G12, 0, G12, 0, 0, G22 + Grc + w2, -G22 - w2, 0, 0, -Grc], [0, 0, 0, -G22 - w2, -G22 - w2, -G22 - w2, 3*G22 + 3*w2, 0, 0, 0], [0, 0, 0, -Grc, 0, 0, 0, Grc, 0, 0], [0, 0, 0, 0, -Grc, 0, 0, 0, Grc, 0], [0, 0, 0, 0, 0, -Grc, 0, 0, 0, Grc]]
    

    
    Ihis 向量输入模板：
    [[&], [&], [&], [&], [&], [&], [&], [&], [&], [&]]
    

    请输入 Ihis 向量:
     [[IhisA - IhisC], [-IhisA + IhisB], [-IhisB + IhisC], [Ihisa + Ihisa_rc], [Ihisb + Ihisb_rc], [Ihisc + Ihisc_rc], [-Ihisa - Ihisb - Ihisc], [-Ihisa_rc], [-Ihisb_rc], [-Ihisc_rc]]
    

    
    --- 子网络 2 / 2 ---
    

    请输入子网络名字（该网络大小为 5）:  UCM
    

    
    G 矩阵输入模板：
    [[&, &, &, &, &], [&, &, &, &, &], [&, &, &, &, &], [&, &, &, &, &], [&, &, &, &, &]]
    

    请输入 G 矩阵:
     [[AA, 0, 0, AP, AN], [0, BB, 0, BP, BN], [0, 0, CC, CP, CN], [AP, BP, CP, PP, PN], [AN, BN, CN, PN, NN]]
    

    
    Ihis 向量输入模板：
    [[&], [&], [&], [&], [&]]
    

    请输入 Ihis 向量:
     [[UCMhis1], [UCMhis2], [UCMhis3], [UCMhis4], [UCMhis5]]
    

    
    ===== 自动生成的节点别名 =====
    Trf: Trf1 <-> Trf.1, Trf2 <-> Trf.2, Trf3 <-> Trf.3, Trf4 <-> Trf.4, Trf5 <-> Trf.5, Trf6 <-> Trf.6, Trf7 <-> Trf.7, Trf8 <-> Trf.8, Trf9 <-> Trf.9, Trf10 <-> Trf.10
    UCM: UCM1 <-> UCM.1, UCM2 <-> UCM.2, UCM3 <-> UCM.3, UCM4 <-> UCM.4, UCM5 <-> UCM.5
    
    ===== 第四步：输入连接关系 =====
    

    请输入连接对的个数（每次连接两个节点）:  3
    

    
    --- 连接 1 / 3 ---
    

    请输入第一个节点（例如 netR2 或 netR.2）:  Trf4
    请输入第二个节点（例如 netL1 或 netL.1）:  UCM1
    

    
    --- 连接 2 / 3 ---
    

    请输入第一个节点（例如 netR2 或 netR.2）:  Trf5
    请输入第二个节点（例如 netL1 或 netL.1）:  UCM2
    

    
    --- 连接 3 / 3 ---
    

    请输入第一个节点（例如 netR2 或 netR.2）:  Trf6
    请输入第二个节点（例如 netL1 或 netL.1）:  UCM3
    

    
    ===== 第五步：接地（可选） =====
    

    请输入要接地的节点，多个用逗号分隔；若没有，直接回车:
     
    

    
    ===== 第六步：装配完整系统 =====
    ========== Global node classes ==========
    Trf.1           -> class [Trf.1] -> global node 11
    Trf.10          -> class [Trf.10] -> global node 3
    Trf.2           -> class [Trf.2] -> global node 6
    Trf.3           -> class [Trf.3] -> global node 10
    Trf.4           -> class [Trf.4] -> global node 4
    Trf.5           -> class [Trf.5] -> global node 0
    Trf.6           -> class [Trf.6] -> global node 1
    Trf.7           -> class [Trf.7] -> global node 2
    Trf.8           -> class [Trf.8] -> global node 7
    Trf.9           -> class [Trf.9] -> global node 9
    UCM.1           -> class [Trf.4] -> global node 4
    UCM.2           -> class [Trf.5] -> global node 0
    UCM.3           -> class [Trf.6] -> global node 1
    UCM.4           -> class [UCM.4] -> global node 8
    UCM.5           -> class [UCM.5] -> global node 5
    
    ========== Stamping log ==========
    Ihis: Trf[0] (Trf.1) -> global[11] += IhisA - IhisC
    G: Trf[0,0] (Trf.1,Trf.1) -> G_global[11,11] += 2*G11 + 2*w1
    G: Trf[0,1] (Trf.1,Trf.2) -> G_global[11,6] += -G11 - w1
    G: Trf[0,2] (Trf.1,Trf.3) -> G_global[11,10] += -G11 - w1
    G: Trf[0,3] (Trf.1,Trf.4) -> G_global[11,4] += G12
    G: Trf[0,4] (Trf.1,Trf.5) -> G_global[11,0] += 0
    G: Trf[0,5] (Trf.1,Trf.6) -> G_global[11,1] += -G12
    G: Trf[0,6] (Trf.1,Trf.7) -> G_global[11,2] += 0
    G: Trf[0,7] (Trf.1,Trf.8) -> G_global[11,7] += 0
    G: Trf[0,8] (Trf.1,Trf.9) -> G_global[11,9] += 0
    G: Trf[0,9] (Trf.1,Trf.10) -> G_global[11,3] += 0
    Ihis: Trf[1] (Trf.2) -> global[6] += -IhisA + IhisB
    G: Trf[1,0] (Trf.2,Trf.1) -> G_global[6,11] += -G11 - w1
    G: Trf[1,1] (Trf.2,Trf.2) -> G_global[6,6] += 2*G11 + 2*w1
    G: Trf[1,2] (Trf.2,Trf.3) -> G_global[6,10] += -G11 - w1
    G: Trf[1,3] (Trf.2,Trf.4) -> G_global[6,4] += -G12
    G: Trf[1,4] (Trf.2,Trf.5) -> G_global[6,0] += G12
    G: Trf[1,5] (Trf.2,Trf.6) -> G_global[6,1] += 0
    G: Trf[1,6] (Trf.2,Trf.7) -> G_global[6,2] += 0
    G: Trf[1,7] (Trf.2,Trf.8) -> G_global[6,7] += 0
    G: Trf[1,8] (Trf.2,Trf.9) -> G_global[6,9] += 0
    G: Trf[1,9] (Trf.2,Trf.10) -> G_global[6,3] += 0
    Ihis: Trf[2] (Trf.3) -> global[10] += -IhisB + IhisC
    G: Trf[2,0] (Trf.3,Trf.1) -> G_global[10,11] += -G11 - w1
    G: Trf[2,1] (Trf.3,Trf.2) -> G_global[10,6] += -G11 - w1
    G: Trf[2,2] (Trf.3,Trf.3) -> G_global[10,10] += 2*G11 + 2*w1
    G: Trf[2,3] (Trf.3,Trf.4) -> G_global[10,4] += 0
    G: Trf[2,4] (Trf.3,Trf.5) -> G_global[10,0] += -G12
    G: Trf[2,5] (Trf.3,Trf.6) -> G_global[10,1] += G12
    G: Trf[2,6] (Trf.3,Trf.7) -> G_global[10,2] += 0
    G: Trf[2,7] (Trf.3,Trf.8) -> G_global[10,7] += 0
    G: Trf[2,8] (Trf.3,Trf.9) -> G_global[10,9] += 0
    G: Trf[2,9] (Trf.3,Trf.10) -> G_global[10,3] += 0
    Ihis: Trf[3] (Trf.4) -> global[4] += Ihisa + Ihisa_rc
    G: Trf[3,0] (Trf.4,Trf.1) -> G_global[4,11] += G12
    G: Trf[3,1] (Trf.4,Trf.2) -> G_global[4,6] += -G12
    G: Trf[3,2] (Trf.4,Trf.3) -> G_global[4,10] += 0
    G: Trf[3,3] (Trf.4,Trf.4) -> G_global[4,4] += G22 + Grc + w2
    G: Trf[3,4] (Trf.4,Trf.5) -> G_global[4,0] += 0
    G: Trf[3,5] (Trf.4,Trf.6) -> G_global[4,1] += 0
    G: Trf[3,6] (Trf.4,Trf.7) -> G_global[4,2] += -G22 - w2
    G: Trf[3,7] (Trf.4,Trf.8) -> G_global[4,7] += -Grc
    G: Trf[3,8] (Trf.4,Trf.9) -> G_global[4,9] += 0
    G: Trf[3,9] (Trf.4,Trf.10) -> G_global[4,3] += 0
    Ihis: Trf[4] (Trf.5) -> global[0] += Ihisb + Ihisb_rc
    G: Trf[4,0] (Trf.5,Trf.1) -> G_global[0,11] += 0
    G: Trf[4,1] (Trf.5,Trf.2) -> G_global[0,6] += G12
    G: Trf[4,2] (Trf.5,Trf.3) -> G_global[0,10] += -G12
    G: Trf[4,3] (Trf.5,Trf.4) -> G_global[0,4] += 0
    G: Trf[4,4] (Trf.5,Trf.5) -> G_global[0,0] += G22 + Grc + w2
    G: Trf[4,5] (Trf.5,Trf.6) -> G_global[0,1] += 0
    G: Trf[4,6] (Trf.5,Trf.7) -> G_global[0,2] += -G22 - w2
    G: Trf[4,7] (Trf.5,Trf.8) -> G_global[0,7] += 0
    G: Trf[4,8] (Trf.5,Trf.9) -> G_global[0,9] += -Grc
    G: Trf[4,9] (Trf.5,Trf.10) -> G_global[0,3] += 0
    Ihis: Trf[5] (Trf.6) -> global[1] += Ihisc + Ihisc_rc
    G: Trf[5,0] (Trf.6,Trf.1) -> G_global[1,11] += -G12
    G: Trf[5,1] (Trf.6,Trf.2) -> G_global[1,6] += 0
    G: Trf[5,2] (Trf.6,Trf.3) -> G_global[1,10] += G12
    G: Trf[5,3] (Trf.6,Trf.4) -> G_global[1,4] += 0
    G: Trf[5,4] (Trf.6,Trf.5) -> G_global[1,0] += 0
    G: Trf[5,5] (Trf.6,Trf.6) -> G_global[1,1] += G22 + Grc + w2
    G: Trf[5,6] (Trf.6,Trf.7) -> G_global[1,2] += -G22 - w2
    G: Trf[5,7] (Trf.6,Trf.8) -> G_global[1,7] += 0
    G: Trf[5,8] (Trf.6,Trf.9) -> G_global[1,9] += 0
    G: Trf[5,9] (Trf.6,Trf.10) -> G_global[1,3] += -Grc
    Ihis: Trf[6] (Trf.7) -> global[2] += -Ihisa - Ihisb - Ihisc
    G: Trf[6,0] (Trf.7,Trf.1) -> G_global[2,11] += 0
    G: Trf[6,1] (Trf.7,Trf.2) -> G_global[2,6] += 0
    G: Trf[6,2] (Trf.7,Trf.3) -> G_global[2,10] += 0
    G: Trf[6,3] (Trf.7,Trf.4) -> G_global[2,4] += -G22 - w2
    G: Trf[6,4] (Trf.7,Trf.5) -> G_global[2,0] += -G22 - w2
    G: Trf[6,5] (Trf.7,Trf.6) -> G_global[2,1] += -G22 - w2
    G: Trf[6,6] (Trf.7,Trf.7) -> G_global[2,2] += 3*G22 + 3*w2
    G: Trf[6,7] (Trf.7,Trf.8) -> G_global[2,7] += 0
    G: Trf[6,8] (Trf.7,Trf.9) -> G_global[2,9] += 0
    G: Trf[6,9] (Trf.7,Trf.10) -> G_global[2,3] += 0
    Ihis: Trf[7] (Trf.8) -> global[7] += -Ihisa_rc
    G: Trf[7,0] (Trf.8,Trf.1) -> G_global[7,11] += 0
    G: Trf[7,1] (Trf.8,Trf.2) -> G_global[7,6] += 0
    G: Trf[7,2] (Trf.8,Trf.3) -> G_global[7,10] += 0
    G: Trf[7,3] (Trf.8,Trf.4) -> G_global[7,4] += -Grc
    G: Trf[7,4] (Trf.8,Trf.5) -> G_global[7,0] += 0
    G: Trf[7,5] (Trf.8,Trf.6) -> G_global[7,1] += 0
    G: Trf[7,6] (Trf.8,Trf.7) -> G_global[7,2] += 0
    G: Trf[7,7] (Trf.8,Trf.8) -> G_global[7,7] += Grc
    G: Trf[7,8] (Trf.8,Trf.9) -> G_global[7,9] += 0
    G: Trf[7,9] (Trf.8,Trf.10) -> G_global[7,3] += 0
    Ihis: Trf[8] (Trf.9) -> global[9] += -Ihisb_rc
    G: Trf[8,0] (Trf.9,Trf.1) -> G_global[9,11] += 0
    G: Trf[8,1] (Trf.9,Trf.2) -> G_global[9,6] += 0
    G: Trf[8,2] (Trf.9,Trf.3) -> G_global[9,10] += 0
    G: Trf[8,3] (Trf.9,Trf.4) -> G_global[9,4] += 0
    G: Trf[8,4] (Trf.9,Trf.5) -> G_global[9,0] += -Grc
    G: Trf[8,5] (Trf.9,Trf.6) -> G_global[9,1] += 0
    G: Trf[8,6] (Trf.9,Trf.7) -> G_global[9,2] += 0
    G: Trf[8,7] (Trf.9,Trf.8) -> G_global[9,7] += 0
    G: Trf[8,8] (Trf.9,Trf.9) -> G_global[9,9] += Grc
    G: Trf[8,9] (Trf.9,Trf.10) -> G_global[9,3] += 0
    Ihis: Trf[9] (Trf.10) -> global[3] += -Ihisc_rc
    G: Trf[9,0] (Trf.10,Trf.1) -> G_global[3,11] += 0
    G: Trf[9,1] (Trf.10,Trf.2) -> G_global[3,6] += 0
    G: Trf[9,2] (Trf.10,Trf.3) -> G_global[3,10] += 0
    G: Trf[9,3] (Trf.10,Trf.4) -> G_global[3,4] += 0
    G: Trf[9,4] (Trf.10,Trf.5) -> G_global[3,0] += 0
    G: Trf[9,5] (Trf.10,Trf.6) -> G_global[3,1] += -Grc
    G: Trf[9,6] (Trf.10,Trf.7) -> G_global[3,2] += 0
    G: Trf[9,7] (Trf.10,Trf.8) -> G_global[3,7] += 0
    G: Trf[9,8] (Trf.10,Trf.9) -> G_global[3,9] += 0
    G: Trf[9,9] (Trf.10,Trf.10) -> G_global[3,3] += Grc
    Ihis: UCM[0] (UCM.1) -> global[4] += UCMhis1
    G: UCM[0,0] (UCM.1,UCM.1) -> G_global[4,4] += AA
    G: UCM[0,1] (UCM.1,UCM.2) -> G_global[4,0] += 0
    G: UCM[0,2] (UCM.1,UCM.3) -> G_global[4,1] += 0
    G: UCM[0,3] (UCM.1,UCM.4) -> G_global[4,8] += AP
    G: UCM[0,4] (UCM.1,UCM.5) -> G_global[4,5] += AN
    Ihis: UCM[1] (UCM.2) -> global[0] += UCMhis2
    G: UCM[1,0] (UCM.2,UCM.1) -> G_global[0,4] += 0
    G: UCM[1,1] (UCM.2,UCM.2) -> G_global[0,0] += BB
    G: UCM[1,2] (UCM.2,UCM.3) -> G_global[0,1] += 0
    G: UCM[1,3] (UCM.2,UCM.4) -> G_global[0,8] += BP
    G: UCM[1,4] (UCM.2,UCM.5) -> G_global[0,5] += BN
    Ihis: UCM[2] (UCM.3) -> global[1] += UCMhis3
    G: UCM[2,0] (UCM.3,UCM.1) -> G_global[1,4] += 0
    G: UCM[2,1] (UCM.3,UCM.2) -> G_global[1,0] += 0
    G: UCM[2,2] (UCM.3,UCM.3) -> G_global[1,1] += CC
    G: UCM[2,3] (UCM.3,UCM.4) -> G_global[1,8] += CP
    G: UCM[2,4] (UCM.3,UCM.5) -> G_global[1,5] += CN
    Ihis: UCM[3] (UCM.4) -> global[8] += UCMhis4
    G: UCM[3,0] (UCM.4,UCM.1) -> G_global[8,4] += AP
    G: UCM[3,1] (UCM.4,UCM.2) -> G_global[8,0] += BP
    G: UCM[3,2] (UCM.4,UCM.3) -> G_global[8,1] += CP
    G: UCM[3,3] (UCM.4,UCM.4) -> G_global[8,8] += PP
    G: UCM[3,4] (UCM.4,UCM.5) -> G_global[8,5] += PN
    Ihis: UCM[4] (UCM.5) -> global[5] += UCMhis5
    G: UCM[4,0] (UCM.5,UCM.1) -> G_global[5,4] += AN
    G: UCM[4,1] (UCM.5,UCM.2) -> G_global[5,0] += BN
    G: UCM[4,2] (UCM.5,UCM.3) -> G_global[5,1] += CN
    G: UCM[4,3] (UCM.5,UCM.4) -> G_global[5,8] += PN
    G: UCM[4,4] (UCM.5,UCM.5) -> G_global[5,5] += NN
    
    ========== Global equation ==========
    I_global = G_global * V_global + Ihis_global
    
    ========== System Equation ==========
    
    


$\displaystyle \left[\begin{matrix}I_{Trf_5}\\I_{Trf_6}\\I_{Trf_7}\\I_{Trf_10}\\I_{Trf_4}\\I_{UCM_5}\\I_{Trf_2}\\I_{Trf_8}\\I_{UCM_4}\\I_{Trf_9}\\I_{Trf_3}\\I_{Trf_1}\end{matrix}\right] = \left[\begin{array}{cccccccccccc}BB + G_{22} + Grc + w_{2} & 0 & - G_{22} - w_{2} & 0 & 0 & BN & G_{12} & 0 & BP & - Grc & - G_{12} & 0\\0 & CC + G_{22} + Grc + w_{2} & - G_{22} - w_{2} & - Grc & 0 & CN & 0 & 0 & CP & 0 & G_{12} & - G_{12}\\- G_{22} - w_{2} & - G_{22} - w_{2} & 3 G_{22} + 3 w_{2} & 0 & - G_{22} - w_{2} & 0 & 0 & 0 & 0 & 0 & 0 & 0\\0 & - Grc & 0 & Grc & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0\\0 & 0 & - G_{22} - w_{2} & 0 & AA + G_{22} + Grc + w_{2} & AN & - G_{12} & - Grc & AP & 0 & 0 & G_{12}\\BN & CN & 0 & 0 & AN & NN & 0 & 0 & PN & 0 & 0 & 0\\G_{12} & 0 & 0 & 0 & - G_{12} & 0 & 2 G_{11} + 2 w_{1} & 0 & 0 & 0 & - G_{11} - w_{1} & - G_{11} - w_{1}\\0 & 0 & 0 & 0 & - Grc & 0 & 0 & Grc & 0 & 0 & 0 & 0\\BP & CP & 0 & 0 & AP & PN & 0 & 0 & PP & 0 & 0 & 0\\- Grc & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & Grc & 0 & 0\\- G_{12} & G_{12} & 0 & 0 & 0 & 0 & - G_{11} - w_{1} & 0 & 0 & 0 & 2 G_{11} + 2 w_{1} & - G_{11} - w_{1}\\0 & - G_{12} & 0 & 0 & G_{12} & 0 & - G_{11} - w_{1} & 0 & 0 & 0 & - G_{11} - w_{1} & 2 G_{11} + 2 w_{1}\end{array}\right] \left[\begin{matrix}V_{Trf_5}\\V_{Trf_6}\\V_{Trf_7}\\V_{Trf_10}\\V_{Trf_4}\\V_{UCM_5}\\V_{Trf_2}\\V_{Trf_8}\\V_{UCM_4}\\V_{Trf_9}\\V_{Trf_3}\\V_{Trf_1}\end{matrix}\right] + \left[\begin{matrix}Ihisb + Ihisb_{rc} + UCMhis_{2}\\Ihisc + Ihisc_{rc} + UCMhis_{3}\\- Ihisa - Ihisb - Ihisc\\- Ihisc_{rc}\\Ihisa + Ihisa_{rc} + UCMhis_{1}\\UCMhis_{5}\\IhisB - IhisA\\- Ihisa_{rc}\\UCMhis_{4}\\- Ihisb_{rc}\\IhisC - IhisB\\IhisA - IhisC\end{matrix}\right]$


    
    ===== 第七步：选择外部节点，生成 reduced system =====
    可用节点别名如下：
    Trf: ['Trf1', 'Trf2', 'Trf3', 'Trf4', 'Trf5', 'Trf6', 'Trf7', 'Trf8', 'Trf9', 'Trf10']
    UCM: ['UCM1', 'UCM2', 'UCM3', 'UCM4', 'UCM5']
    

    请输入要保留的外部节点，多个用逗号分隔（例如 netR1,netL2）；若保留所有节点，输入 all：
     Trf1,Trf2,Trf3,Trf7,Trf8,Trf9,Trf10,UCM4,UCM5
    

    
    当前 reduced system 的节点顺序为：
    1: gid=2, node=Trf.7
    2: gid=3, node=Trf.10
    3: gid=5, node=UCM.5
    4: gid=6, node=Trf.2
    5: gid=7, node=Trf.8
    6: gid=8, node=UCM.4
    7: gid=9, node=Trf.9
    8: gid=10, node=Trf.3
    9: gid=11, node=Trf.1
    
    ========== Reduced External System Equation ==========
    
    


$\displaystyle \left[\begin{matrix}I_{Trf_7}\\I_{Trf_10}\\I_{UCM_5}\\I_{Trf_2}\\I_{Trf_8}\\I_{UCM_4}\\I_{Trf_9}\\I_{Trf_3}\\I_{Trf_1}\end{matrix}\right] = \left[\begin{matrix}3 G_{22} + 3 w_{2} - \frac{\left(G_{22} + w_{2}\right)^{2}}{CC + G_{22} + Grc + w_{2}} - \frac{\left(G_{22} + w_{2}\right)^{2}}{BB + G_{22} + Grc + w_{2}} - \frac{\left(G_{22} + w_{2}\right)^{2}}{AA + G_{22} + Grc + w_{2}} & - \frac{Grc \left(G_{22} + w_{2}\right)}{CC + G_{22} + Grc + w_{2}} & \frac{AN \left(G_{22} + w_{2}\right)}{AA + G_{22} + Grc + w_{2}} + \frac{BN \left(G_{22} + w_{2}\right)}{BB + G_{22} + Grc + w_{2}} + \frac{CN \left(G_{22} + w_{2}\right)}{CC + G_{22} + Grc + w_{2}} & \frac{G_{12} \left(AA - BB\right) \left(G_{22} + w_{2}\right)}{\left(AA + G_{22} + Grc + w_{2}\right) \left(BB + G_{22} + Grc + w_{2}\right)} & - \frac{Grc \left(G_{22} + w_{2}\right)}{AA + G_{22} + Grc + w_{2}} & \frac{AP \left(G_{22} + w_{2}\right)}{AA + G_{22} + Grc + w_{2}} + \frac{BP \left(G_{22} + w_{2}\right)}{BB + G_{22} + Grc + w_{2}} + \frac{CP \left(G_{22} + w_{2}\right)}{CC + G_{22} + Grc + w_{2}} & - \frac{Grc \left(G_{22} + w_{2}\right)}{BB + G_{22} + Grc + w_{2}} & \frac{G_{12} \left(BB - CC\right) \left(G_{22} + w_{2}\right)}{\left(BB + G_{22} + Grc + w_{2}\right) \left(CC + G_{22} + Grc + w_{2}\right)} & \frac{G_{12} \left(- AA + CC\right) \left(G_{22} + w_{2}\right)}{\left(AA + G_{22} + Grc + w_{2}\right) \left(CC + G_{22} + Grc + w_{2}\right)}\\- \frac{Grc \left(G_{22} + w_{2}\right)}{CC + G_{22} + Grc + w_{2}} & \frac{Grc \left(CC + G_{22} + w_{2}\right)}{CC + G_{22} + Grc + w_{2}} & \frac{CN Grc}{CC + G_{22} + Grc + w_{2}} & 0 & 0 & \frac{CP Grc}{CC + G_{22} + Grc + w_{2}} & 0 & \frac{G_{12} Grc}{CC + G_{22} + Grc + w_{2}} & - \frac{G_{12} Grc}{CC + G_{22} + Grc + w_{2}}\\\frac{AN \left(G_{22} + w_{2}\right)}{AA + G_{22} + Grc + w_{2}} + \frac{BN \left(G_{22} + w_{2}\right)}{BB + G_{22} + Grc + w_{2}} + \frac{CN \left(G_{22} + w_{2}\right)}{CC + G_{22} + Grc + w_{2}} & \frac{CN Grc}{CC + G_{22} + Grc + w_{2}} & - \frac{AN^{2}}{AA + G_{22} + Grc + w_{2}} - \frac{BN^{2}}{BB + G_{22} + Grc + w_{2}} - \frac{CN^{2}}{CC + G_{22} + Grc + w_{2}} + NN & \frac{G_{12} \left(AN \left(BB + G_{22} + Grc + w_{2}\right) - BN \left(AA + G_{22} + Grc + w_{2}\right)\right)}{\left(AA + G_{22} + Grc + w_{2}\right) \left(BB + G_{22} + Grc + w_{2}\right)} & \frac{AN Grc}{AA + G_{22} + Grc + w_{2}} & - \frac{AN AP}{AA + G_{22} + Grc + w_{2}} - \frac{BN BP}{BB + G_{22} + Grc + w_{2}} - \frac{CN CP}{CC + G_{22} + Grc + w_{2}} + PN & \frac{BN Grc}{BB + G_{22} + Grc + w_{2}} & \frac{G_{12} \left(BN \left(CC + G_{22} + Grc + w_{2}\right) - CN \left(BB + G_{22} + Grc + w_{2}\right)\right)}{\left(BB + G_{22} + Grc + w_{2}\right) \left(CC + G_{22} + Grc + w_{2}\right)} & \frac{G_{12} \left(- AN \left(CC + G_{22} + Grc + w_{2}\right) + CN \left(AA + G_{22} + Grc + w_{2}\right)\right)}{\left(AA + G_{22} + Grc + w_{2}\right) \left(CC + G_{22} + Grc + w_{2}\right)}\\\frac{G_{12} \left(AA - BB\right) \left(G_{22} + w_{2}\right)}{\left(AA + G_{22} + Grc + w_{2}\right) \left(BB + G_{22} + Grc + w_{2}\right)} & 0 & \frac{G_{12} \left(AN \left(BB + G_{22} + Grc + w_{2}\right) - BN \left(AA + G_{22} + Grc + w_{2}\right)\right)}{\left(AA + G_{22} + Grc + w_{2}\right) \left(BB + G_{22} + Grc + w_{2}\right)} & 2 G_{11} - \frac{G_{12}^{2}}{BB + G_{22} + Grc + w_{2}} - \frac{G_{12}^{2}}{AA + G_{22} + Grc + w_{2}} + 2 w_{1} & - \frac{G_{12} Grc}{AA + G_{22} + Grc + w_{2}} & \frac{G_{12} \left(AP \left(BB + G_{22} + Grc + w_{2}\right) - BP \left(AA + G_{22} + Grc + w_{2}\right)\right)}{\left(AA + G_{22} + Grc + w_{2}\right) \left(BB + G_{22} + Grc + w_{2}\right)} & \frac{G_{12} Grc}{BB + G_{22} + Grc + w_{2}} & \frac{G_{12}^{2} - \left(G_{11} + w_{1}\right) \left(BB + G_{22} + Grc + w_{2}\right)}{BB + G_{22} + Grc + w_{2}} & \frac{G_{12}^{2} - \left(G_{11} + w_{1}\right) \left(AA + G_{22} + Grc + w_{2}\right)}{AA + G_{22} + Grc + w_{2}}\\- \frac{Grc \left(G_{22} + w_{2}\right)}{AA + G_{22} + Grc + w_{2}} & 0 & \frac{AN Grc}{AA + G_{22} + Grc + w_{2}} & - \frac{G_{12} Grc}{AA + G_{22} + Grc + w_{2}} & \frac{Grc \left(AA + G_{22} + w_{2}\right)}{AA + G_{22} + Grc + w_{2}} & \frac{AP Grc}{AA + G_{22} + Grc + w_{2}} & 0 & 0 & \frac{G_{12} Grc}{AA + G_{22} + Grc + w_{2}}\\\frac{AP \left(G_{22} + w_{2}\right)}{AA + G_{22} + Grc + w_{2}} + \frac{BP \left(G_{22} + w_{2}\right)}{BB + G_{22} + Grc + w_{2}} + \frac{CP \left(G_{22} + w_{2}\right)}{CC + G_{22} + Grc + w_{2}} & \frac{CP Grc}{CC + G_{22} + Grc + w_{2}} & - \frac{AN AP}{AA + G_{22} + Grc + w_{2}} - \frac{BN BP}{BB + G_{22} + Grc + w_{2}} - \frac{CN CP}{CC + G_{22} + Grc + w_{2}} + PN & \frac{G_{12} \left(AP \left(BB + G_{22} + Grc + w_{2}\right) - BP \left(AA + G_{22} + Grc + w_{2}\right)\right)}{\left(AA + G_{22} + Grc + w_{2}\right) \left(BB + G_{22} + Grc + w_{2}\right)} & \frac{AP Grc}{AA + G_{22} + Grc + w_{2}} & - \frac{AP^{2}}{AA + G_{22} + Grc + w_{2}} - \frac{BP^{2}}{BB + G_{22} + Grc + w_{2}} - \frac{CP^{2}}{CC + G_{22} + Grc + w_{2}} + PP & \frac{BP Grc}{BB + G_{22} + Grc + w_{2}} & \frac{G_{12} \left(BP \left(CC + G_{22} + Grc + w_{2}\right) - CP \left(BB + G_{22} + Grc + w_{2}\right)\right)}{\left(BB + G_{22} + Grc + w_{2}\right) \left(CC + G_{22} + Grc + w_{2}\right)} & \frac{G_{12} \left(- AP \left(CC + G_{22} + Grc + w_{2}\right) + CP \left(AA + G_{22} + Grc + w_{2}\right)\right)}{\left(AA + G_{22} + Grc + w_{2}\right) \left(CC + G_{22} + Grc + w_{2}\right)}\\- \frac{Grc \left(G_{22} + w_{2}\right)}{BB + G_{22} + Grc + w_{2}} & 0 & \frac{BN Grc}{BB + G_{22} + Grc + w_{2}} & \frac{G_{12} Grc}{BB + G_{22} + Grc + w_{2}} & 0 & \frac{BP Grc}{BB + G_{22} + Grc + w_{2}} & \frac{Grc \left(BB + G_{22} + w_{2}\right)}{BB + G_{22} + Grc + w_{2}} & - \frac{G_{12} Grc}{BB + G_{22} + Grc + w_{2}} & 0\\\frac{G_{12} \left(BB - CC\right) \left(G_{22} + w_{2}\right)}{\left(BB + G_{22} + Grc + w_{2}\right) \left(CC + G_{22} + Grc + w_{2}\right)} & \frac{G_{12} Grc}{CC + G_{22} + Grc + w_{2}} & \frac{G_{12} \left(BN \left(CC + G_{22} + Grc + w_{2}\right) - CN \left(BB + G_{22} + Grc + w_{2}\right)\right)}{\left(BB + G_{22} + Grc + w_{2}\right) \left(CC + G_{22} + Grc + w_{2}\right)} & \frac{G_{12}^{2} - \left(G_{11} + w_{1}\right) \left(BB + G_{22} + Grc + w_{2}\right)}{BB + G_{22} + Grc + w_{2}} & 0 & \frac{G_{12} \left(BP \left(CC + G_{22} + Grc + w_{2}\right) - CP \left(BB + G_{22} + Grc + w_{2}\right)\right)}{\left(BB + G_{22} + Grc + w_{2}\right) \left(CC + G_{22} + Grc + w_{2}\right)} & - \frac{G_{12} Grc}{BB + G_{22} + Grc + w_{2}} & 2 G_{11} - \frac{G_{12}^{2}}{CC + G_{22} + Grc + w_{2}} - \frac{G_{12}^{2}}{BB + G_{22} + Grc + w_{2}} + 2 w_{1} & \frac{G_{12}^{2} - \left(G_{11} + w_{1}\right) \left(CC + G_{22} + Grc + w_{2}\right)}{CC + G_{22} + Grc + w_{2}}\\\frac{G_{12} \left(- AA + CC\right) \left(G_{22} + w_{2}\right)}{\left(AA + G_{22} + Grc + w_{2}\right) \left(CC + G_{22} + Grc + w_{2}\right)} & - \frac{G_{12} Grc}{CC + G_{22} + Grc + w_{2}} & \frac{G_{12} \left(- AN \left(CC + G_{22} + Grc + w_{2}\right) + CN \left(AA + G_{22} + Grc + w_{2}\right)\right)}{\left(AA + G_{22} + Grc + w_{2}\right) \left(CC + G_{22} + Grc + w_{2}\right)} & \frac{G_{12}^{2} - \left(G_{11} + w_{1}\right) \left(AA + G_{22} + Grc + w_{2}\right)}{AA + G_{22} + Grc + w_{2}} & \frac{G_{12} Grc}{AA + G_{22} + Grc + w_{2}} & \frac{G_{12} \left(- AP \left(CC + G_{22} + Grc + w_{2}\right) + CP \left(AA + G_{22} + Grc + w_{2}\right)\right)}{\left(AA + G_{22} + Grc + w_{2}\right) \left(CC + G_{22} + Grc + w_{2}\right)} & 0 & \frac{G_{12}^{2} - \left(G_{11} + w_{1}\right) \left(CC + G_{22} + Grc + w_{2}\right)}{CC + G_{22} + Grc + w_{2}} & 2 G_{11} - \frac{G_{12}^{2}}{CC + G_{22} + Grc + w_{2}} - \frac{G_{12}^{2}}{AA + G_{22} + Grc + w_{2}} + 2 w_{1}\end{matrix}\right] \left[\begin{matrix}V_{Trf_7}\\V_{Trf_10}\\V_{UCM_5}\\V_{Trf_2}\\V_{Trf_8}\\V_{UCM_4}\\V_{Trf_9}\\V_{Trf_3}\\V_{Trf_1}\end{matrix}\right] + \left[\begin{matrix}- Ihisa - Ihisb - Ihisc + \frac{\left(G_{22} + w_{2}\right) \left(Ihisa + Ihisa_{rc} + UCMhis_{1}\right)}{AA + G_{22} + Grc + w_{2}} + \frac{\left(G_{22} + w_{2}\right) \left(Ihisb + Ihisb_{rc} + UCMhis_{2}\right)}{BB + G_{22} + Grc + w_{2}} + \frac{\left(G_{22} + w_{2}\right) \left(Ihisc + Ihisc_{rc} + UCMhis_{3}\right)}{CC + G_{22} + Grc + w_{2}}\\\frac{Grc \left(Ihisc + Ihisc_{rc} + UCMhis_{3}\right) - Ihisc_{rc} \left(CC + G_{22} + Grc + w_{2}\right)}{CC + G_{22} + Grc + w_{2}}\\- \frac{AN \left(Ihisa + Ihisa_{rc} + UCMhis_{1}\right)}{AA + G_{22} + Grc + w_{2}} - \frac{BN \left(Ihisb + Ihisb_{rc} + UCMhis_{2}\right)}{BB + G_{22} + Grc + w_{2}} - \frac{CN \left(Ihisc + Ihisc_{rc} + UCMhis_{3}\right)}{CC + G_{22} + Grc + w_{2}} + UCMhis_{5}\\\frac{G_{12} \left(Ihisa + Ihisa_{rc} + UCMhis_{1}\right)}{AA + G_{22} + Grc + w_{2}} - \frac{G_{12} \left(Ihisb + Ihisb_{rc} + UCMhis_{2}\right)}{BB + G_{22} + Grc + w_{2}} - IhisA + IhisB\\\frac{Grc \left(Ihisa + Ihisa_{rc} + UCMhis_{1}\right) - Ihisa_{rc} \left(AA + G_{22} + Grc + w_{2}\right)}{AA + G_{22} + Grc + w_{2}}\\- \frac{AP \left(Ihisa + Ihisa_{rc} + UCMhis_{1}\right)}{AA + G_{22} + Grc + w_{2}} - \frac{BP \left(Ihisb + Ihisb_{rc} + UCMhis_{2}\right)}{BB + G_{22} + Grc + w_{2}} - \frac{CP \left(Ihisc + Ihisc_{rc} + UCMhis_{3}\right)}{CC + G_{22} + Grc + w_{2}} + UCMhis_{4}\\\frac{Grc \left(Ihisb + Ihisb_{rc} + UCMhis_{2}\right) - Ihisb_{rc} \left(BB + G_{22} + Grc + w_{2}\right)}{BB + G_{22} + Grc + w_{2}}\\\frac{G_{12} \left(Ihisb + Ihisb_{rc} + UCMhis_{2}\right)}{BB + G_{22} + Grc + w_{2}} - \frac{G_{12} \left(Ihisc + Ihisc_{rc} + UCMhis_{3}\right)}{CC + G_{22} + Grc + w_{2}} - IhisB + IhisC\\- \frac{G_{12} \left(Ihisa + Ihisa_{rc} + UCMhis_{1}\right)}{AA + G_{22} + Grc + w_{2}} + \frac{G_{12} \left(Ihisc + Ihisc_{rc} + UCMhis_{3}\right)}{CC + G_{22} + Grc + w_{2}} + IhisA - IhisC\end{matrix}\right]$


    
    是否要更改 reduced system 中节点显示顺序？(y/n):  y
    

    
    请输入新的顺序，必须是当前 reduced system 已包含的节点，只能重排，不能增删。
    可以输入 global_nodes 中的真实名字，也可以输入 alias。
    例如: netR1,netL2  或  netR.1,netL.2
    

    新的节点顺序:  Trf1,Trf2,Trf3,Trf7,Trf8,Trf9,Trf10,UCM4,UCM5
    

    
    重排后的 reduced system：
    
    ========== Reduced External System Equation ==========
    
    


$\displaystyle \left[\begin{matrix}I_{Trf_1}\\I_{Trf_2}\\I_{Trf_3}\\I_{Trf_7}\\I_{Trf_8}\\I_{Trf_9}\\I_{Trf_10}\\I_{UCM_4}\\I_{UCM_5}\end{matrix}\right] = \left[\begin{matrix}2 G_{11} - \frac{G_{12}^{2}}{CC + G_{22} + Grc + w_{2}} - \frac{G_{12}^{2}}{AA + G_{22} + Grc + w_{2}} + 2 w_{1} & \frac{G_{12}^{2} - \left(G_{11} + w_{1}\right) \left(AA + G_{22} + Grc + w_{2}\right)}{AA + G_{22} + Grc + w_{2}} & \frac{G_{12}^{2} - \left(G_{11} + w_{1}\right) \left(CC + G_{22} + Grc + w_{2}\right)}{CC + G_{22} + Grc + w_{2}} & \frac{G_{12} \left(- AA + CC\right) \left(G_{22} + w_{2}\right)}{\left(AA + G_{22} + Grc + w_{2}\right) \left(CC + G_{22} + Grc + w_{2}\right)} & \frac{G_{12} Grc}{AA + G_{22} + Grc + w_{2}} & 0 & - \frac{G_{12} Grc}{CC + G_{22} + Grc + w_{2}} & \frac{G_{12} \left(- AP \left(CC + G_{22} + Grc + w_{2}\right) + CP \left(AA + G_{22} + Grc + w_{2}\right)\right)}{\left(AA + G_{22} + Grc + w_{2}\right) \left(CC + G_{22} + Grc + w_{2}\right)} & \frac{G_{12} \left(- AN \left(CC + G_{22} + Grc + w_{2}\right) + CN \left(AA + G_{22} + Grc + w_{2}\right)\right)}{\left(AA + G_{22} + Grc + w_{2}\right) \left(CC + G_{22} + Grc + w_{2}\right)}\\\frac{G_{12}^{2} - \left(G_{11} + w_{1}\right) \left(AA + G_{22} + Grc + w_{2}\right)}{AA + G_{22} + Grc + w_{2}} & 2 G_{11} - \frac{G_{12}^{2}}{BB + G_{22} + Grc + w_{2}} - \frac{G_{12}^{2}}{AA + G_{22} + Grc + w_{2}} + 2 w_{1} & \frac{G_{12}^{2} - \left(G_{11} + w_{1}\right) \left(BB + G_{22} + Grc + w_{2}\right)}{BB + G_{22} + Grc + w_{2}} & \frac{G_{12} \left(AA - BB\right) \left(G_{22} + w_{2}\right)}{\left(AA + G_{22} + Grc + w_{2}\right) \left(BB + G_{22} + Grc + w_{2}\right)} & - \frac{G_{12} Grc}{AA + G_{22} + Grc + w_{2}} & \frac{G_{12} Grc}{BB + G_{22} + Grc + w_{2}} & 0 & \frac{G_{12} \left(AP \left(BB + G_{22} + Grc + w_{2}\right) - BP \left(AA + G_{22} + Grc + w_{2}\right)\right)}{\left(AA + G_{22} + Grc + w_{2}\right) \left(BB + G_{22} + Grc + w_{2}\right)} & \frac{G_{12} \left(AN \left(BB + G_{22} + Grc + w_{2}\right) - BN \left(AA + G_{22} + Grc + w_{2}\right)\right)}{\left(AA + G_{22} + Grc + w_{2}\right) \left(BB + G_{22} + Grc + w_{2}\right)}\\\frac{G_{12}^{2} - \left(G_{11} + w_{1}\right) \left(CC + G_{22} + Grc + w_{2}\right)}{CC + G_{22} + Grc + w_{2}} & \frac{G_{12}^{2} - \left(G_{11} + w_{1}\right) \left(BB + G_{22} + Grc + w_{2}\right)}{BB + G_{22} + Grc + w_{2}} & 2 G_{11} - \frac{G_{12}^{2}}{CC + G_{22} + Grc + w_{2}} - \frac{G_{12}^{2}}{BB + G_{22} + Grc + w_{2}} + 2 w_{1} & \frac{G_{12} \left(BB - CC\right) \left(G_{22} + w_{2}\right)}{\left(BB + G_{22} + Grc + w_{2}\right) \left(CC + G_{22} + Grc + w_{2}\right)} & 0 & - \frac{G_{12} Grc}{BB + G_{22} + Grc + w_{2}} & \frac{G_{12} Grc}{CC + G_{22} + Grc + w_{2}} & \frac{G_{12} \left(BP \left(CC + G_{22} + Grc + w_{2}\right) - CP \left(BB + G_{22} + Grc + w_{2}\right)\right)}{\left(BB + G_{22} + Grc + w_{2}\right) \left(CC + G_{22} + Grc + w_{2}\right)} & \frac{G_{12} \left(BN \left(CC + G_{22} + Grc + w_{2}\right) - CN \left(BB + G_{22} + Grc + w_{2}\right)\right)}{\left(BB + G_{22} + Grc + w_{2}\right) \left(CC + G_{22} + Grc + w_{2}\right)}\\\frac{G_{12} \left(- AA + CC\right) \left(G_{22} + w_{2}\right)}{\left(AA + G_{22} + Grc + w_{2}\right) \left(CC + G_{22} + Grc + w_{2}\right)} & \frac{G_{12} \left(AA - BB\right) \left(G_{22} + w_{2}\right)}{\left(AA + G_{22} + Grc + w_{2}\right) \left(BB + G_{22} + Grc + w_{2}\right)} & \frac{G_{12} \left(BB - CC\right) \left(G_{22} + w_{2}\right)}{\left(BB + G_{22} + Grc + w_{2}\right) \left(CC + G_{22} + Grc + w_{2}\right)} & 3 G_{22} + 3 w_{2} - \frac{\left(G_{22} + w_{2}\right)^{2}}{CC + G_{22} + Grc + w_{2}} - \frac{\left(G_{22} + w_{2}\right)^{2}}{BB + G_{22} + Grc + w_{2}} - \frac{\left(G_{22} + w_{2}\right)^{2}}{AA + G_{22} + Grc + w_{2}} & - \frac{Grc \left(G_{22} + w_{2}\right)}{AA + G_{22} + Grc + w_{2}} & - \frac{Grc \left(G_{22} + w_{2}\right)}{BB + G_{22} + Grc + w_{2}} & - \frac{Grc \left(G_{22} + w_{2}\right)}{CC + G_{22} + Grc + w_{2}} & \frac{AP \left(G_{22} + w_{2}\right)}{AA + G_{22} + Grc + w_{2}} + \frac{BP \left(G_{22} + w_{2}\right)}{BB + G_{22} + Grc + w_{2}} + \frac{CP \left(G_{22} + w_{2}\right)}{CC + G_{22} + Grc + w_{2}} & \frac{AN \left(G_{22} + w_{2}\right)}{AA + G_{22} + Grc + w_{2}} + \frac{BN \left(G_{22} + w_{2}\right)}{BB + G_{22} + Grc + w_{2}} + \frac{CN \left(G_{22} + w_{2}\right)}{CC + G_{22} + Grc + w_{2}}\\\frac{G_{12} Grc}{AA + G_{22} + Grc + w_{2}} & - \frac{G_{12} Grc}{AA + G_{22} + Grc + w_{2}} & 0 & - \frac{Grc \left(G_{22} + w_{2}\right)}{AA + G_{22} + Grc + w_{2}} & \frac{Grc \left(AA + G_{22} + w_{2}\right)}{AA + G_{22} + Grc + w_{2}} & 0 & 0 & \frac{AP Grc}{AA + G_{22} + Grc + w_{2}} & \frac{AN Grc}{AA + G_{22} + Grc + w_{2}}\\0 & \frac{G_{12} Grc}{BB + G_{22} + Grc + w_{2}} & - \frac{G_{12} Grc}{BB + G_{22} + Grc + w_{2}} & - \frac{Grc \left(G_{22} + w_{2}\right)}{BB + G_{22} + Grc + w_{2}} & 0 & \frac{Grc \left(BB + G_{22} + w_{2}\right)}{BB + G_{22} + Grc + w_{2}} & 0 & \frac{BP Grc}{BB + G_{22} + Grc + w_{2}} & \frac{BN Grc}{BB + G_{22} + Grc + w_{2}}\\- \frac{G_{12} Grc}{CC + G_{22} + Grc + w_{2}} & 0 & \frac{G_{12} Grc}{CC + G_{22} + Grc + w_{2}} & - \frac{Grc \left(G_{22} + w_{2}\right)}{CC + G_{22} + Grc + w_{2}} & 0 & 0 & \frac{Grc \left(CC + G_{22} + w_{2}\right)}{CC + G_{22} + Grc + w_{2}} & \frac{CP Grc}{CC + G_{22} + Grc + w_{2}} & \frac{CN Grc}{CC + G_{22} + Grc + w_{2}}\\\frac{G_{12} \left(- AP \left(CC + G_{22} + Grc + w_{2}\right) + CP \left(AA + G_{22} + Grc + w_{2}\right)\right)}{\left(AA + G_{22} + Grc + w_{2}\right) \left(CC + G_{22} + Grc + w_{2}\right)} & \frac{G_{12} \left(AP \left(BB + G_{22} + Grc + w_{2}\right) - BP \left(AA + G_{22} + Grc + w_{2}\right)\right)}{\left(AA + G_{22} + Grc + w_{2}\right) \left(BB + G_{22} + Grc + w_{2}\right)} & \frac{G_{12} \left(BP \left(CC + G_{22} + Grc + w_{2}\right) - CP \left(BB + G_{22} + Grc + w_{2}\right)\right)}{\left(BB + G_{22} + Grc + w_{2}\right) \left(CC + G_{22} + Grc + w_{2}\right)} & \frac{AP \left(G_{22} + w_{2}\right)}{AA + G_{22} + Grc + w_{2}} + \frac{BP \left(G_{22} + w_{2}\right)}{BB + G_{22} + Grc + w_{2}} + \frac{CP \left(G_{22} + w_{2}\right)}{CC + G_{22} + Grc + w_{2}} & \frac{AP Grc}{AA + G_{22} + Grc + w_{2}} & \frac{BP Grc}{BB + G_{22} + Grc + w_{2}} & \frac{CP Grc}{CC + G_{22} + Grc + w_{2}} & - \frac{AP^{2}}{AA + G_{22} + Grc + w_{2}} - \frac{BP^{2}}{BB + G_{22} + Grc + w_{2}} - \frac{CP^{2}}{CC + G_{22} + Grc + w_{2}} + PP & - \frac{AN AP}{AA + G_{22} + Grc + w_{2}} - \frac{BN BP}{BB + G_{22} + Grc + w_{2}} - \frac{CN CP}{CC + G_{22} + Grc + w_{2}} + PN\\\frac{G_{12} \left(- AN \left(CC + G_{22} + Grc + w_{2}\right) + CN \left(AA + G_{22} + Grc + w_{2}\right)\right)}{\left(AA + G_{22} + Grc + w_{2}\right) \left(CC + G_{22} + Grc + w_{2}\right)} & \frac{G_{12} \left(AN \left(BB + G_{22} + Grc + w_{2}\right) - BN \left(AA + G_{22} + Grc + w_{2}\right)\right)}{\left(AA + G_{22} + Grc + w_{2}\right) \left(BB + G_{22} + Grc + w_{2}\right)} & \frac{G_{12} \left(BN \left(CC + G_{22} + Grc + w_{2}\right) - CN \left(BB + G_{22} + Grc + w_{2}\right)\right)}{\left(BB + G_{22} + Grc + w_{2}\right) \left(CC + G_{22} + Grc + w_{2}\right)} & \frac{AN \left(G_{22} + w_{2}\right)}{AA + G_{22} + Grc + w_{2}} + \frac{BN \left(G_{22} + w_{2}\right)}{BB + G_{22} + Grc + w_{2}} + \frac{CN \left(G_{22} + w_{2}\right)}{CC + G_{22} + Grc + w_{2}} & \frac{AN Grc}{AA + G_{22} + Grc + w_{2}} & \frac{BN Grc}{BB + G_{22} + Grc + w_{2}} & \frac{CN Grc}{CC + G_{22} + Grc + w_{2}} & - \frac{AN AP}{AA + G_{22} + Grc + w_{2}} - \frac{BN BP}{BB + G_{22} + Grc + w_{2}} - \frac{CN CP}{CC + G_{22} + Grc + w_{2}} + PN & - \frac{AN^{2}}{AA + G_{22} + Grc + w_{2}} - \frac{BN^{2}}{BB + G_{22} + Grc + w_{2}} - \frac{CN^{2}}{CC + G_{22} + Grc + w_{2}} + NN\end{matrix}\right] \left[\begin{matrix}V_{Trf_1}\\V_{Trf_2}\\V_{Trf_3}\\V_{Trf_7}\\V_{Trf_8}\\V_{Trf_9}\\V_{Trf_10}\\V_{UCM_4}\\V_{UCM_5}\end{matrix}\right] + \left[\begin{matrix}- \frac{G_{12} \left(Ihisa + Ihisa_{rc} + UCMhis_{1}\right)}{AA + G_{22} + Grc + w_{2}} + \frac{G_{12} \left(Ihisc + Ihisc_{rc} + UCMhis_{3}\right)}{CC + G_{22} + Grc + w_{2}} + IhisA - IhisC\\\frac{G_{12} \left(Ihisa + Ihisa_{rc} + UCMhis_{1}\right)}{AA + G_{22} + Grc + w_{2}} - \frac{G_{12} \left(Ihisb + Ihisb_{rc} + UCMhis_{2}\right)}{BB + G_{22} + Grc + w_{2}} - IhisA + IhisB\\\frac{G_{12} \left(Ihisb + Ihisb_{rc} + UCMhis_{2}\right)}{BB + G_{22} + Grc + w_{2}} - \frac{G_{12} \left(Ihisc + Ihisc_{rc} + UCMhis_{3}\right)}{CC + G_{22} + Grc + w_{2}} - IhisB + IhisC\\- Ihisa - Ihisb - Ihisc + \frac{\left(G_{22} + w_{2}\right) \left(Ihisa + Ihisa_{rc} + UCMhis_{1}\right)}{AA + G_{22} + Grc + w_{2}} + \frac{\left(G_{22} + w_{2}\right) \left(Ihisb + Ihisb_{rc} + UCMhis_{2}\right)}{BB + G_{22} + Grc + w_{2}} + \frac{\left(G_{22} + w_{2}\right) \left(Ihisc + Ihisc_{rc} + UCMhis_{3}\right)}{CC + G_{22} + Grc + w_{2}}\\\frac{Grc \left(Ihisa + Ihisa_{rc} + UCMhis_{1}\right) - Ihisa_{rc} \left(AA + G_{22} + Grc + w_{2}\right)}{AA + G_{22} + Grc + w_{2}}\\\frac{Grc \left(Ihisb + Ihisb_{rc} + UCMhis_{2}\right) - Ihisb_{rc} \left(BB + G_{22} + Grc + w_{2}\right)}{BB + G_{22} + Grc + w_{2}}\\\frac{Grc \left(Ihisc + Ihisc_{rc} + UCMhis_{3}\right) - Ihisc_{rc} \left(CC + G_{22} + Grc + w_{2}\right)}{CC + G_{22} + Grc + w_{2}}\\- \frac{AP \left(Ihisa + Ihisa_{rc} + UCMhis_{1}\right)}{AA + G_{22} + Grc + w_{2}} - \frac{BP \left(Ihisb + Ihisb_{rc} + UCMhis_{2}\right)}{BB + G_{22} + Grc + w_{2}} - \frac{CP \left(Ihisc + Ihisc_{rc} + UCMhis_{3}\right)}{CC + G_{22} + Grc + w_{2}} + UCMhis_{4}\\- \frac{AN \left(Ihisa + Ihisa_{rc} + UCMhis_{1}\right)}{AA + G_{22} + Grc + w_{2}} - \frac{BN \left(Ihisb + Ihisb_{rc} + UCMhis_{2}\right)}{BB + G_{22} + Grc + w_{2}} - \frac{CN \left(Ihisc + Ihisc_{rc} + UCMhis_{3}\right)}{CC + G_{22} + Grc + w_{2}} + UCMhis_{5}\end{matrix}\right]$


    
    ========== Reduced System Python Format ==========
    
    G_reduced_final = sp.Matrix([[2*G11 - G12**2/(CC + G22 + Grc + w2) - G12**2/(AA + G22 + Grc + w2) + 2*w1, (G12**2 - (G11 + w1)*(AA + G22 + Grc + w2))/(AA + G22 + Grc + w2), (G12**2 - (G11 + w1)*(CC + G22 + Grc + w2))/(CC + G22 + Grc + w2), G12*(-AA + CC)*(G22 + w2)/((AA + G22 + Grc + w2)*(CC + G22 + Grc + w2)), G12*Grc/(AA + G22 + Grc + w2), 0, -G12*Grc/(CC + G22 + Grc + w2), G12*(-AP*(CC + G22 + Grc + w2) + CP*(AA + G22 + Grc + w2))/((AA + G22 + Grc + w2)*(CC + G22 + Grc + w2)), G12*(-AN*(CC + G22 + Grc + w2) + CN*(AA + G22 + Grc + w2))/((AA + G22 + Grc + w2)*(CC + G22 + Grc + w2))], [(G12**2 - (G11 + w1)*(AA + G22 + Grc + w2))/(AA + G22 + Grc + w2), 2*G11 - G12**2/(BB + G22 + Grc + w2) - G12**2/(AA + G22 + Grc + w2) + 2*w1, (G12**2 - (G11 + w1)*(BB + G22 + Grc + w2))/(BB + G22 + Grc + w2), G12*(AA - BB)*(G22 + w2)/((AA + G22 + Grc + w2)*(BB + G22 + Grc + w2)), -G12*Grc/(AA + G22 + Grc + w2), G12*Grc/(BB + G22 + Grc + w2), 0, G12*(AP*(BB + G22 + Grc + w2) - BP*(AA + G22 + Grc + w2))/((AA + G22 + Grc + w2)*(BB + G22 + Grc + w2)), G12*(AN*(BB + G22 + Grc + w2) - BN*(AA + G22 + Grc + w2))/((AA + G22 + Grc + w2)*(BB + G22 + Grc + w2))], [(G12**2 - (G11 + w1)*(CC + G22 + Grc + w2))/(CC + G22 + Grc + w2), (G12**2 - (G11 + w1)*(BB + G22 + Grc + w2))/(BB + G22 + Grc + w2), 2*G11 - G12**2/(CC + G22 + Grc + w2) - G12**2/(BB + G22 + Grc + w2) + 2*w1, G12*(BB - CC)*(G22 + w2)/((BB + G22 + Grc + w2)*(CC + G22 + Grc + w2)), 0, -G12*Grc/(BB + G22 + Grc + w2), G12*Grc/(CC + G22 + Grc + w2), G12*(BP*(CC + G22 + Grc + w2) - CP*(BB + G22 + Grc + w2))/((BB + G22 + Grc + w2)*(CC + G22 + Grc + w2)), G12*(BN*(CC + G22 + Grc + w2) - CN*(BB + G22 + Grc + w2))/((BB + G22 + Grc + w2)*(CC + G22 + Grc + w2))], [G12*(-AA + CC)*(G22 + w2)/((AA + G22 + Grc + w2)*(CC + G22 + Grc + w2)), G12*(AA - BB)*(G22 + w2)/((AA + G22 + Grc + w2)*(BB + G22 + Grc + w2)), G12*(BB - CC)*(G22 + w2)/((BB + G22 + Grc + w2)*(CC + G22 + Grc + w2)), 3*G22 + 3*w2 - (G22 + w2)**2/(CC + G22 + Grc + w2) - (G22 + w2)**2/(BB + G22 + Grc + w2) - (G22 + w2)**2/(AA + G22 + Grc + w2), -Grc*(G22 + w2)/(AA + G22 + Grc + w2), -Grc*(G22 + w2)/(BB + G22 + Grc + w2), -Grc*(G22 + w2)/(CC + G22 + Grc + w2), AP*(G22 + w2)/(AA + G22 + Grc + w2) + BP*(G22 + w2)/(BB + G22 + Grc + w2) + CP*(G22 + w2)/(CC + G22 + Grc + w2), AN*(G22 + w2)/(AA + G22 + Grc + w2) + BN*(G22 + w2)/(BB + G22 + Grc + w2) + CN*(G22 + w2)/(CC + G22 + Grc + w2)], [G12*Grc/(AA + G22 + Grc + w2), -G12*Grc/(AA + G22 + Grc + w2), 0, -Grc*(G22 + w2)/(AA + G22 + Grc + w2), Grc*(AA + G22 + w2)/(AA + G22 + Grc + w2), 0, 0, AP*Grc/(AA + G22 + Grc + w2), AN*Grc/(AA + G22 + Grc + w2)], [0, G12*Grc/(BB + G22 + Grc + w2), -G12*Grc/(BB + G22 + Grc + w2), -Grc*(G22 + w2)/(BB + G22 + Grc + w2), 0, Grc*(BB + G22 + w2)/(BB + G22 + Grc + w2), 0, BP*Grc/(BB + G22 + Grc + w2), BN*Grc/(BB + G22 + Grc + w2)], [-G12*Grc/(CC + G22 + Grc + w2), 0, G12*Grc/(CC + G22 + Grc + w2), -Grc*(G22 + w2)/(CC + G22 + Grc + w2), 0, 0, Grc*(CC + G22 + w2)/(CC + G22 + Grc + w2), CP*Grc/(CC + G22 + Grc + w2), CN*Grc/(CC + G22 + Grc + w2)], [G12*(-AP*(CC + G22 + Grc + w2) + CP*(AA + G22 + Grc + w2))/((AA + G22 + Grc + w2)*(CC + G22 + Grc + w2)), G12*(AP*(BB + G22 + Grc + w2) - BP*(AA + G22 + Grc + w2))/((AA + G22 + Grc + w2)*(BB + G22 + Grc + w2)), G12*(BP*(CC + G22 + Grc + w2) - CP*(BB + G22 + Grc + w2))/((BB + G22 + Grc + w2)*(CC + G22 + Grc + w2)), AP*(G22 + w2)/(AA + G22 + Grc + w2) + BP*(G22 + w2)/(BB + G22 + Grc + w2) + CP*(G22 + w2)/(CC + G22 + Grc + w2), AP*Grc/(AA + G22 + Grc + w2), BP*Grc/(BB + G22 + Grc + w2), CP*Grc/(CC + G22 + Grc + w2), -AP**2/(AA + G22 + Grc + w2) - BP**2/(BB + G22 + Grc + w2) - CP**2/(CC + G22 + Grc + w2) + PP, -AN*AP/(AA + G22 + Grc + w2) - BN*BP/(BB + G22 + Grc + w2) - CN*CP/(CC + G22 + Grc + w2) + PN], [G12*(-AN*(CC + G22 + Grc + w2) + CN*(AA + G22 + Grc + w2))/((AA + G22 + Grc + w2)*(CC + G22 + Grc + w2)), G12*(AN*(BB + G22 + Grc + w2) - BN*(AA + G22 + Grc + w2))/((AA + G22 + Grc + w2)*(BB + G22 + Grc + w2)), G12*(BN*(CC + G22 + Grc + w2) - CN*(BB + G22 + Grc + w2))/((BB + G22 + Grc + w2)*(CC + G22 + Grc + w2)), AN*(G22 + w2)/(AA + G22 + Grc + w2) + BN*(G22 + w2)/(BB + G22 + Grc + w2) + CN*(G22 + w2)/(CC + G22 + Grc + w2), AN*Grc/(AA + G22 + Grc + w2), BN*Grc/(BB + G22 + Grc + w2), CN*Grc/(CC + G22 + Grc + w2), -AN*AP/(AA + G22 + Grc + w2) - BN*BP/(BB + G22 + Grc + w2) - CN*CP/(CC + G22 + Grc + w2) + PN, -AN**2/(AA + G22 + Grc + w2) - BN**2/(BB + G22 + Grc + w2) - CN**2/(CC + G22 + Grc + w2) + NN]])
    
    Ihis_reduced_final = sp.Matrix([[-G12*(Ihisa + Ihisa_rc + UCMhis1)/(AA + G22 + Grc + w2) + G12*(Ihisc + Ihisc_rc + UCMhis3)/(CC + G22 + Grc + w2) + IhisA - IhisC], [G12*(Ihisa + Ihisa_rc + UCMhis1)/(AA + G22 + Grc + w2) - G12*(Ihisb + Ihisb_rc + UCMhis2)/(BB + G22 + Grc + w2) - IhisA + IhisB], [G12*(Ihisb + Ihisb_rc + UCMhis2)/(BB + G22 + Grc + w2) - G12*(Ihisc + Ihisc_rc + UCMhis3)/(CC + G22 + Grc + w2) - IhisB + IhisC], [-Ihisa - Ihisb - Ihisc + (G22 + w2)*(Ihisa + Ihisa_rc + UCMhis1)/(AA + G22 + Grc + w2) + (G22 + w2)*(Ihisb + Ihisb_rc + UCMhis2)/(BB + G22 + Grc + w2) + (G22 + w2)*(Ihisc + Ihisc_rc + UCMhis3)/(CC + G22 + Grc + w2)], [(Grc*(Ihisa + Ihisa_rc + UCMhis1) - Ihisa_rc*(AA + G22 + Grc + w2))/(AA + G22 + Grc + w2)], [(Grc*(Ihisb + Ihisb_rc + UCMhis2) - Ihisb_rc*(BB + G22 + Grc + w2))/(BB + G22 + Grc + w2)], [(Grc*(Ihisc + Ihisc_rc + UCMhis3) - Ihisc_rc*(CC + G22 + Grc + w2))/(CC + G22 + Grc + w2)], [-AP*(Ihisa + Ihisa_rc + UCMhis1)/(AA + G22 + Grc + w2) - BP*(Ihisb + Ihisb_rc + UCMhis2)/(BB + G22 + Grc + w2) - CP*(Ihisc + Ihisc_rc + UCMhis3)/(CC + G22 + Grc + w2) + UCMhis4], [-AN*(Ihisa + Ihisa_rc + UCMhis1)/(AA + G22 + Grc + w2) - BN*(Ihisb + Ihisb_rc + UCMhis2)/(BB + G22 + Grc + w2) - CN*(Ihisc + Ihisc_rc + UCMhis3)/(CC + G22 + Grc + w2) + UCMhis5]])
    
    ===== 自动生成的可复制代码 =====
    
    # -------- 子网络：Trf --------
    G_Trf = sp.Matrix([[2*G11 + 2*w1, -G11 - w1, -G11 - w1, G12, 0, -G12, 0, 0, 0, 0], [-G11 - w1, 2*G11 + 2*w1, -G11 - w1, -G12, G12, 0, 0, 0, 0, 0], [-G11 - w1, -G11 - w1, 2*G11 + 2*w1, 0, -G12, G12, 0, 0, 0, 0], [G12, -G12, 0, G22 + Grc + w2, 0, 0, -G22 - w2, -Grc, 0, 0], [0, G12, -G12, 0, G22 + Grc + w2, 0, -G22 - w2, 0, -Grc, 0], [-G12, 0, G12, 0, 0, G22 + Grc + w2, -G22 - w2, 0, 0, -Grc], [0, 0, 0, -G22 - w2, -G22 - w2, -G22 - w2, 3*G22 + 3*w2, 0, 0, 0], [0, 0, 0, -Grc, 0, 0, 0, Grc, 0, 0], [0, 0, 0, 0, -Grc, 0, 0, 0, Grc, 0], [0, 0, 0, 0, 0, -Grc, 0, 0, 0, Grc]])
    Ihis_Trf = sp.Matrix([[IhisA - IhisC], [-IhisA + IhisB], [-IhisB + IhisC], [Ihisa + Ihisa_rc], [Ihisb + Ihisb_rc], [Ihisc + Ihisc_rc], [-Ihisa - Ihisb - Ihisc], [-Ihisa_rc], [-Ihisb_rc], [-Ihisc_rc]])
    Trf = SubNetwork(
        name="Trf",
        local_nodes=['1', '2', '3', '4', '5', '6', '7', '8', '9', '10'],
        G_local=G_Trf,
        Ihis_local=Ihis_Trf
    )
    
    # -------- 子网络：UCM --------
    G_UCM = sp.Matrix([[AA, 0, 0, AP, AN], [0, BB, 0, BP, BN], [0, 0, CC, CP, CN], [AP, BP, CP, PP, PN], [AN, BN, CN, PN, NN]])
    Ihis_UCM = sp.Matrix([[UCMhis1], [UCMhis2], [UCMhis3], [UCMhis4], [UCMhis5]])
    UCM = SubNetwork(
        name="UCM",
        local_nodes=['1', '2', '3', '4', '5'],
        G_local=G_UCM,
        Ihis_local=Ihis_UCM
    )
    
    # -------- 装配 --------
    asm = NetworkAssembler()
    asm.add_subnetwork(Trf)
    asm.add_subnetwork(UCM)
    asm.connect("Trf.4", "UCM.1")
    asm.connect("Trf.5", "UCM.2")
    asm.connect("Trf.6", "UCM.3")
    
    assembled = asm.assemble(verbose=True)
    pretty_display_system_split(assembled)
    
    # -------- 选择外部节点并消元 --------
    gid_Trf1 = assembled["node_to_gid"]["Trf.1"]
    gid_Trf2 = assembled["node_to_gid"]["Trf.2"]
    gid_Trf3 = assembled["node_to_gid"]["Trf.3"]
    gid_Trf7 = assembled["node_to_gid"]["Trf.7"]
    gid_Trf8 = assembled["node_to_gid"]["Trf.8"]
    gid_Trf9 = assembled["node_to_gid"]["Trf.9"]
    gid_Trf10 = assembled["node_to_gid"]["Trf.10"]
    gid_UCM4 = assembled["node_to_gid"]["UCM.4"]
    gid_UCM5 = assembled["node_to_gid"]["UCM.5"]
    ext_ids = sorted(list({2, 3, 5, 6, 7, 8, 9, 10, 11}))
    reduced = asm.eliminate_internal_nodes(assembled, external_node_ids=ext_ids)
    pretty_display_reduced_system(assembled, reduced)
    
    # -------- 重排 reduced system 节点顺序 --------
    reduced = reorder_reduced_system(assembled, reduced, [11, 6, 10, 2, 7, 9, 3, 8, 5])
    pretty_display_reduced_system(assembled, reduced)
    
    # -------- reduced system 的 Python 格式 --------
    G_reduced_final = sp.Matrix([[2*G11 - G12**2/(CC + G22 + Grc + w2) - G12**2/(AA + G22 + Grc + w2) + 2*w1, (G12**2 - (G11 + w1)*(AA + G22 + Grc + w2))/(AA + G22 + Grc + w2), (G12**2 - (G11 + w1)*(CC + G22 + Grc + w2))/(CC + G22 + Grc + w2), G12*(-AA + CC)*(G22 + w2)/((AA + G22 + Grc + w2)*(CC + G22 + Grc + w2)), G12*Grc/(AA + G22 + Grc + w2), 0, -G12*Grc/(CC + G22 + Grc + w2), G12*(-AP*(CC + G22 + Grc + w2) + CP*(AA + G22 + Grc + w2))/((AA + G22 + Grc + w2)*(CC + G22 + Grc + w2)), G12*(-AN*(CC + G22 + Grc + w2) + CN*(AA + G22 + Grc + w2))/((AA + G22 + Grc + w2)*(CC + G22 + Grc + w2))], [(G12**2 - (G11 + w1)*(AA + G22 + Grc + w2))/(AA + G22 + Grc + w2), 2*G11 - G12**2/(BB + G22 + Grc + w2) - G12**2/(AA + G22 + Grc + w2) + 2*w1, (G12**2 - (G11 + w1)*(BB + G22 + Grc + w2))/(BB + G22 + Grc + w2), G12*(AA - BB)*(G22 + w2)/((AA + G22 + Grc + w2)*(BB + G22 + Grc + w2)), -G12*Grc/(AA + G22 + Grc + w2), G12*Grc/(BB + G22 + Grc + w2), 0, G12*(AP*(BB + G22 + Grc + w2) - BP*(AA + G22 + Grc + w2))/((AA + G22 + Grc + w2)*(BB + G22 + Grc + w2)), G12*(AN*(BB + G22 + Grc + w2) - BN*(AA + G22 + Grc + w2))/((AA + G22 + Grc + w2)*(BB + G22 + Grc + w2))], [(G12**2 - (G11 + w1)*(CC + G22 + Grc + w2))/(CC + G22 + Grc + w2), (G12**2 - (G11 + w1)*(BB + G22 + Grc + w2))/(BB + G22 + Grc + w2), 2*G11 - G12**2/(CC + G22 + Grc + w2) - G12**2/(BB + G22 + Grc + w2) + 2*w1, G12*(BB - CC)*(G22 + w2)/((BB + G22 + Grc + w2)*(CC + G22 + Grc + w2)), 0, -G12*Grc/(BB + G22 + Grc + w2), G12*Grc/(CC + G22 + Grc + w2), G12*(BP*(CC + G22 + Grc + w2) - CP*(BB + G22 + Grc + w2))/((BB + G22 + Grc + w2)*(CC + G22 + Grc + w2)), G12*(BN*(CC + G22 + Grc + w2) - CN*(BB + G22 + Grc + w2))/((BB + G22 + Grc + w2)*(CC + G22 + Grc + w2))], [G12*(-AA + CC)*(G22 + w2)/((AA + G22 + Grc + w2)*(CC + G22 + Grc + w2)), G12*(AA - BB)*(G22 + w2)/((AA + G22 + Grc + w2)*(BB + G22 + Grc + w2)), G12*(BB - CC)*(G22 + w2)/((BB + G22 + Grc + w2)*(CC + G22 + Grc + w2)), 3*G22 + 3*w2 - (G22 + w2)**2/(CC + G22 + Grc + w2) - (G22 + w2)**2/(BB + G22 + Grc + w2) - (G22 + w2)**2/(AA + G22 + Grc + w2), -Grc*(G22 + w2)/(AA + G22 + Grc + w2), -Grc*(G22 + w2)/(BB + G22 + Grc + w2), -Grc*(G22 + w2)/(CC + G22 + Grc + w2), AP*(G22 + w2)/(AA + G22 + Grc + w2) + BP*(G22 + w2)/(BB + G22 + Grc + w2) + CP*(G22 + w2)/(CC + G22 + Grc + w2), AN*(G22 + w2)/(AA + G22 + Grc + w2) + BN*(G22 + w2)/(BB + G22 + Grc + w2) + CN*(G22 + w2)/(CC + G22 + Grc + w2)], [G12*Grc/(AA + G22 + Grc + w2), -G12*Grc/(AA + G22 + Grc + w2), 0, -Grc*(G22 + w2)/(AA + G22 + Grc + w2), Grc*(AA + G22 + w2)/(AA + G22 + Grc + w2), 0, 0, AP*Grc/(AA + G22 + Grc + w2), AN*Grc/(AA + G22 + Grc + w2)], [0, G12*Grc/(BB + G22 + Grc + w2), -G12*Grc/(BB + G22 + Grc + w2), -Grc*(G22 + w2)/(BB + G22 + Grc + w2), 0, Grc*(BB + G22 + w2)/(BB + G22 + Grc + w2), 0, BP*Grc/(BB + G22 + Grc + w2), BN*Grc/(BB + G22 + Grc + w2)], [-G12*Grc/(CC + G22 + Grc + w2), 0, G12*Grc/(CC + G22 + Grc + w2), -Grc*(G22 + w2)/(CC + G22 + Grc + w2), 0, 0, Grc*(CC + G22 + w2)/(CC + G22 + Grc + w2), CP*Grc/(CC + G22 + Grc + w2), CN*Grc/(CC + G22 + Grc + w2)], [G12*(-AP*(CC + G22 + Grc + w2) + CP*(AA + G22 + Grc + w2))/((AA + G22 + Grc + w2)*(CC + G22 + Grc + w2)), G12*(AP*(BB + G22 + Grc + w2) - BP*(AA + G22 + Grc + w2))/((AA + G22 + Grc + w2)*(BB + G22 + Grc + w2)), G12*(BP*(CC + G22 + Grc + w2) - CP*(BB + G22 + Grc + w2))/((BB + G22 + Grc + w2)*(CC + G22 + Grc + w2)), AP*(G22 + w2)/(AA + G22 + Grc + w2) + BP*(G22 + w2)/(BB + G22 + Grc + w2) + CP*(G22 + w2)/(CC + G22 + Grc + w2), AP*Grc/(AA + G22 + Grc + w2), BP*Grc/(BB + G22 + Grc + w2), CP*Grc/(CC + G22 + Grc + w2), -AP**2/(AA + G22 + Grc + w2) - BP**2/(BB + G22 + Grc + w2) - CP**2/(CC + G22 + Grc + w2) + PP, -AN*AP/(AA + G22 + Grc + w2) - BN*BP/(BB + G22 + Grc + w2) - CN*CP/(CC + G22 + Grc + w2) + PN], [G12*(-AN*(CC + G22 + Grc + w2) + CN*(AA + G22 + Grc + w2))/((AA + G22 + Grc + w2)*(CC + G22 + Grc + w2)), G12*(AN*(BB + G22 + Grc + w2) - BN*(AA + G22 + Grc + w2))/((AA + G22 + Grc + w2)*(BB + G22 + Grc + w2)), G12*(BN*(CC + G22 + Grc + w2) - CN*(BB + G22 + Grc + w2))/((BB + G22 + Grc + w2)*(CC + G22 + Grc + w2)), AN*(G22 + w2)/(AA + G22 + Grc + w2) + BN*(G22 + w2)/(BB + G22 + Grc + w2) + CN*(G22 + w2)/(CC + G22 + Grc + w2), AN*Grc/(AA + G22 + Grc + w2), BN*Grc/(BB + G22 + Grc + w2), CN*Grc/(CC + G22 + Grc + w2), -AN*AP/(AA + G22 + Grc + w2) - BN*BP/(BB + G22 + Grc + w2) - CN*CP/(CC + G22 + Grc + w2) + PN, -AN**2/(AA + G22 + Grc + w2) - BN**2/(BB + G22 + Grc + w2) - CN**2/(CC + G22 + Grc + w2) + NN]])
    Ihis_reduced_final = sp.Matrix([[-G12*(Ihisa + Ihisa_rc + UCMhis1)/(AA + G22 + Grc + w2) + G12*(Ihisc + Ihisc_rc + UCMhis3)/(CC + G22 + Grc + w2) + IhisA - IhisC], [G12*(Ihisa + Ihisa_rc + UCMhis1)/(AA + G22 + Grc + w2) - G12*(Ihisb + Ihisb_rc + UCMhis2)/(BB + G22 + Grc + w2) - IhisA + IhisB], [G12*(Ihisb + Ihisb_rc + UCMhis2)/(BB + G22 + Grc + w2) - G12*(Ihisc + Ihisc_rc + UCMhis3)/(CC + G22 + Grc + w2) - IhisB + IhisC], [-Ihisa - Ihisb - Ihisc + (G22 + w2)*(Ihisa + Ihisa_rc + UCMhis1)/(AA + G22 + Grc + w2) + (G22 + w2)*(Ihisb + Ihisb_rc + UCMhis2)/(BB + G22 + Grc + w2) + (G22 + w2)*(Ihisc + Ihisc_rc + UCMhis3)/(CC + G22 + Grc + w2)], [(Grc*(Ihisa + Ihisa_rc + UCMhis1) - Ihisa_rc*(AA + G22 + Grc + w2))/(AA + G22 + Grc + w2)], [(Grc*(Ihisb + Ihisb_rc + UCMhis2) - Ihisb_rc*(BB + G22 + Grc + w2))/(BB + G22 + Grc + w2)], [(Grc*(Ihisc + Ihisc_rc + UCMhis3) - Ihisc_rc*(CC + G22 + Grc + w2))/(CC + G22 + Grc + w2)], [-AP*(Ihisa + Ihisa_rc + UCMhis1)/(AA + G22 + Grc + w2) - BP*(Ihisb + Ihisb_rc + UCMhis2)/(BB + G22 + Grc + w2) - CP*(Ihisc + Ihisc_rc + UCMhis3)/(CC + G22 + Grc + w2) + UCMhis4], [-AN*(Ihisa + Ihisa_rc + UCMhis1)/(AA + G22 + Grc + w2) - BN*(Ihisb + Ihisb_rc + UCMhis2)/(BB + G22 + Grc + w2) - CN*(Ihisc + Ihisc_rc + UCMhis3)/(CC + G22 + Grc + w2) + UCMhis5]])
    


```python
result = build_network_interactively_v2()
```

    ===== 第一步：输入子网络数量 =====
    

    请输入子网络个数:  2
    

    
    ===== 第二步：一次性输入每个子网络的节点大小 =====
    

    请输入 2 个子网络的节点大小，用逗号分隔，例如 2,2,3,2,2 :
     10,5
    

    
    你输入的子网络大小为：
    子网络 1: 10 节点
    子网络 2: 5 节点
    
    ===== 第三步：逐个输入子网络名字、G、Ihis =====
    
    --- 子网络 1 / 2 ---
    

    请输入子网络名字（该网络大小为 10）:  Trf
    

    
    G 矩阵输入模板：
    [[&, &, &, &, &, &, &, &, &, &], [&, &, &, &, &, &, &, &, &, &], [&, &, &, &, &, &, &, &, &, &], [&, &, &, &, &, &, &, &, &, &], [&, &, &, &, &, &, &, &, &, &], [&, &, &, &, &, &, &, &, &, &], [&, &, &, &, &, &, &, &, &, &], [&, &, &, &, &, &, &, &, &, &], [&, &, &, &, &, &, &, &, &, &], [&, &, &, &, &, &, &, &, &, &]]
    

    请输入 G 矩阵:
     [[2*G11 + 2*w1, -G11 - w1, -G11 - w1, G12, 0, -G12, 0, 0, 0, 0], [-G11 - w1, 2*G11 + 2*w1, -G11 - w1, -G12, G12, 0, 0, 0, 0, 0], [-G11 - w1, -G11 - w1, 2*G11 + 2*w1, 0, -G12, G12, 0, 0, 0, 0], [G12, -G12, 0, G22 + Grc + w2, 0, 0, -G22 - w2, -Grc, 0, 0], [0, G12, -G12, 0, G22 + Grc + w2, 0, -G22 - w2, 0, -Grc, 0], [-G12, 0, G12, 0, 0, G22 + Grc + w2, -G22 - w2, 0, 0, -Grc], [0, 0, 0, -G22 - w2, -G22 - w2, -G22 - w2, 3*G22 + 3*w2, 0, 0, 0], [0, 0, 0, -Grc, 0, 0, 0, Grc, 0, 0], [0, 0, 0, 0, -Grc, 0, 0, 0, Grc, 0], [0, 0, 0, 0, 0, -Grc, 0, 0, 0, Grc]]
    

    
    Ihis 向量输入模板：
    [[&], [&], [&], [&], [&], [&], [&], [&], [&], [&]]
    

    请输入 Ihis 向量:
     [[IhisA - IhisC], [-IhisA + IhisB], [-IhisB + IhisC], [Ihisa + Ihisa_rc], [Ihisb + Ihisb_rc], [Ihisc + Ihisc_rc], [-Ihisa - Ihisb - Ihisc], [-Ihisa_rc], [-Ihisb_rc], [-Ihisc_rc]]
    

    
    --- 子网络 2 / 2 ---
    

    请输入子网络名字（该网络大小为 5）:  UCM
    

    
    G 矩阵输入模板：
    [[&, &, &, &, &], [&, &, &, &, &], [&, &, &, &, &], [&, &, &, &, &], [&, &, &, &, &]]
    

    请输入 G 矩阵:
     [[AA, 0, 0, AP, AN], [0, BB, 0, BP, BN], [0, 0, CC, CP, CN], [AP, BP, CP, PP, PN], [AN, BN, CN, PN, NN]]
    

    
    Ihis 向量输入模板：
    [[&], [&], [&], [&], [&]]
    

    请输入 Ihis 向量:
     [[UCMhis1], [UCMhis2], [UCMhis3], [UCMhis4], [UCMhis5]]
    

    
    ===== 自动生成的节点别名 =====
    Trf: Trf1 <-> Trf.1, Trf2 <-> Trf.2, Trf3 <-> Trf.3, Trf4 <-> Trf.4, Trf5 <-> Trf.5, Trf6 <-> Trf.6, Trf7 <-> Trf.7, Trf8 <-> Trf.8, Trf9 <-> Trf.9, Trf10 <-> Trf.10
    UCM: UCM1 <-> UCM.1, UCM2 <-> UCM.2, UCM3 <-> UCM.3, UCM4 <-> UCM.4, UCM5 <-> UCM.5
    
    ===== 第四步：输入连接关系 =====
    

    请输入连接对的个数（每次连接两个节点）:  3
    

    
    --- 连接 1 / 3 ---
    

    请输入第一个节点（例如 netR2 或 netR.2）:  Trf4
    请输入第二个节点（例如 netL1 或 netL.1）:  UCM1
    

    
    --- 连接 2 / 3 ---
    

    请输入第一个节点（例如 netR2 或 netR.2）:  Trf5
    请输入第二个节点（例如 netL1 或 netL.1）:  UCM2
    

    
    --- 连接 3 / 3 ---
    

    请输入第一个节点（例如 netR2 或 netR.2）:  Trf6
    请输入第二个节点（例如 netL1 或 netL.1）:  UCM3
    

    
    ===== 第五步：接地（可选） =====
    

    请输入要接地的节点，多个用逗号分隔；若没有，直接回车:
     
    

    
    ===== 第六步：装配完整系统 =====
    ========== Global node classes ==========
    Trf.1           -> class [Trf.1] -> global node 11
    Trf.10          -> class [Trf.10] -> global node 3
    Trf.2           -> class [Trf.2] -> global node 6
    Trf.3           -> class [Trf.3] -> global node 10
    Trf.4           -> class [Trf.4] -> global node 4
    Trf.5           -> class [Trf.5] -> global node 0
    Trf.6           -> class [Trf.6] -> global node 1
    Trf.7           -> class [Trf.7] -> global node 2
    Trf.8           -> class [Trf.8] -> global node 7
    Trf.9           -> class [Trf.9] -> global node 9
    UCM.1           -> class [Trf.4] -> global node 4
    UCM.2           -> class [Trf.5] -> global node 0
    UCM.3           -> class [Trf.6] -> global node 1
    UCM.4           -> class [UCM.4] -> global node 8
    UCM.5           -> class [UCM.5] -> global node 5
    
    ========== Stamping log ==========
    Ihis: Trf[0] (Trf.1) -> global[11] += IhisA - IhisC
    G: Trf[0,0] (Trf.1,Trf.1) -> G_global[11,11] += 2*G11 + 2*w1
    G: Trf[0,1] (Trf.1,Trf.2) -> G_global[11,6] += -G11 - w1
    G: Trf[0,2] (Trf.1,Trf.3) -> G_global[11,10] += -G11 - w1
    G: Trf[0,3] (Trf.1,Trf.4) -> G_global[11,4] += G12
    G: Trf[0,4] (Trf.1,Trf.5) -> G_global[11,0] += 0
    G: Trf[0,5] (Trf.1,Trf.6) -> G_global[11,1] += -G12
    G: Trf[0,6] (Trf.1,Trf.7) -> G_global[11,2] += 0
    G: Trf[0,7] (Trf.1,Trf.8) -> G_global[11,7] += 0
    G: Trf[0,8] (Trf.1,Trf.9) -> G_global[11,9] += 0
    G: Trf[0,9] (Trf.1,Trf.10) -> G_global[11,3] += 0
    Ihis: Trf[1] (Trf.2) -> global[6] += -IhisA + IhisB
    G: Trf[1,0] (Trf.2,Trf.1) -> G_global[6,11] += -G11 - w1
    G: Trf[1,1] (Trf.2,Trf.2) -> G_global[6,6] += 2*G11 + 2*w1
    G: Trf[1,2] (Trf.2,Trf.3) -> G_global[6,10] += -G11 - w1
    G: Trf[1,3] (Trf.2,Trf.4) -> G_global[6,4] += -G12
    G: Trf[1,4] (Trf.2,Trf.5) -> G_global[6,0] += G12
    G: Trf[1,5] (Trf.2,Trf.6) -> G_global[6,1] += 0
    G: Trf[1,6] (Trf.2,Trf.7) -> G_global[6,2] += 0
    G: Trf[1,7] (Trf.2,Trf.8) -> G_global[6,7] += 0
    G: Trf[1,8] (Trf.2,Trf.9) -> G_global[6,9] += 0
    G: Trf[1,9] (Trf.2,Trf.10) -> G_global[6,3] += 0
    Ihis: Trf[2] (Trf.3) -> global[10] += -IhisB + IhisC
    G: Trf[2,0] (Trf.3,Trf.1) -> G_global[10,11] += -G11 - w1
    G: Trf[2,1] (Trf.3,Trf.2) -> G_global[10,6] += -G11 - w1
    G: Trf[2,2] (Trf.3,Trf.3) -> G_global[10,10] += 2*G11 + 2*w1
    G: Trf[2,3] (Trf.3,Trf.4) -> G_global[10,4] += 0
    G: Trf[2,4] (Trf.3,Trf.5) -> G_global[10,0] += -G12
    G: Trf[2,5] (Trf.3,Trf.6) -> G_global[10,1] += G12
    G: Trf[2,6] (Trf.3,Trf.7) -> G_global[10,2] += 0
    G: Trf[2,7] (Trf.3,Trf.8) -> G_global[10,7] += 0
    G: Trf[2,8] (Trf.3,Trf.9) -> G_global[10,9] += 0
    G: Trf[2,9] (Trf.3,Trf.10) -> G_global[10,3] += 0
    Ihis: Trf[3] (Trf.4) -> global[4] += Ihisa + Ihisa_rc
    G: Trf[3,0] (Trf.4,Trf.1) -> G_global[4,11] += G12
    G: Trf[3,1] (Trf.4,Trf.2) -> G_global[4,6] += -G12
    G: Trf[3,2] (Trf.4,Trf.3) -> G_global[4,10] += 0
    G: Trf[3,3] (Trf.4,Trf.4) -> G_global[4,4] += G22 + Grc + w2
    G: Trf[3,4] (Trf.4,Trf.5) -> G_global[4,0] += 0
    G: Trf[3,5] (Trf.4,Trf.6) -> G_global[4,1] += 0
    G: Trf[3,6] (Trf.4,Trf.7) -> G_global[4,2] += -G22 - w2
    G: Trf[3,7] (Trf.4,Trf.8) -> G_global[4,7] += -Grc
    G: Trf[3,8] (Trf.4,Trf.9) -> G_global[4,9] += 0
    G: Trf[3,9] (Trf.4,Trf.10) -> G_global[4,3] += 0
    Ihis: Trf[4] (Trf.5) -> global[0] += Ihisb + Ihisb_rc
    G: Trf[4,0] (Trf.5,Trf.1) -> G_global[0,11] += 0
    G: Trf[4,1] (Trf.5,Trf.2) -> G_global[0,6] += G12
    G: Trf[4,2] (Trf.5,Trf.3) -> G_global[0,10] += -G12
    G: Trf[4,3] (Trf.5,Trf.4) -> G_global[0,4] += 0
    G: Trf[4,4] (Trf.5,Trf.5) -> G_global[0,0] += G22 + Grc + w2
    G: Trf[4,5] (Trf.5,Trf.6) -> G_global[0,1] += 0
    G: Trf[4,6] (Trf.5,Trf.7) -> G_global[0,2] += -G22 - w2
    G: Trf[4,7] (Trf.5,Trf.8) -> G_global[0,7] += 0
    G: Trf[4,8] (Trf.5,Trf.9) -> G_global[0,9] += -Grc
    G: Trf[4,9] (Trf.5,Trf.10) -> G_global[0,3] += 0
    Ihis: Trf[5] (Trf.6) -> global[1] += Ihisc + Ihisc_rc
    G: Trf[5,0] (Trf.6,Trf.1) -> G_global[1,11] += -G12
    G: Trf[5,1] (Trf.6,Trf.2) -> G_global[1,6] += 0
    G: Trf[5,2] (Trf.6,Trf.3) -> G_global[1,10] += G12
    G: Trf[5,3] (Trf.6,Trf.4) -> G_global[1,4] += 0
    G: Trf[5,4] (Trf.6,Trf.5) -> G_global[1,0] += 0
    G: Trf[5,5] (Trf.6,Trf.6) -> G_global[1,1] += G22 + Grc + w2
    G: Trf[5,6] (Trf.6,Trf.7) -> G_global[1,2] += -G22 - w2
    G: Trf[5,7] (Trf.6,Trf.8) -> G_global[1,7] += 0
    G: Trf[5,8] (Trf.6,Trf.9) -> G_global[1,9] += 0
    G: Trf[5,9] (Trf.6,Trf.10) -> G_global[1,3] += -Grc
    Ihis: Trf[6] (Trf.7) -> global[2] += -Ihisa - Ihisb - Ihisc
    G: Trf[6,0] (Trf.7,Trf.1) -> G_global[2,11] += 0
    G: Trf[6,1] (Trf.7,Trf.2) -> G_global[2,6] += 0
    G: Trf[6,2] (Trf.7,Trf.3) -> G_global[2,10] += 0
    G: Trf[6,3] (Trf.7,Trf.4) -> G_global[2,4] += -G22 - w2
    G: Trf[6,4] (Trf.7,Trf.5) -> G_global[2,0] += -G22 - w2
    G: Trf[6,5] (Trf.7,Trf.6) -> G_global[2,1] += -G22 - w2
    G: Trf[6,6] (Trf.7,Trf.7) -> G_global[2,2] += 3*G22 + 3*w2
    G: Trf[6,7] (Trf.7,Trf.8) -> G_global[2,7] += 0
    G: Trf[6,8] (Trf.7,Trf.9) -> G_global[2,9] += 0
    G: Trf[6,9] (Trf.7,Trf.10) -> G_global[2,3] += 0
    Ihis: Trf[7] (Trf.8) -> global[7] += -Ihisa_rc
    G: Trf[7,0] (Trf.8,Trf.1) -> G_global[7,11] += 0
    G: Trf[7,1] (Trf.8,Trf.2) -> G_global[7,6] += 0
    G: Trf[7,2] (Trf.8,Trf.3) -> G_global[7,10] += 0
    G: Trf[7,3] (Trf.8,Trf.4) -> G_global[7,4] += -Grc
    G: Trf[7,4] (Trf.8,Trf.5) -> G_global[7,0] += 0
    G: Trf[7,5] (Trf.8,Trf.6) -> G_global[7,1] += 0
    G: Trf[7,6] (Trf.8,Trf.7) -> G_global[7,2] += 0
    G: Trf[7,7] (Trf.8,Trf.8) -> G_global[7,7] += Grc
    G: Trf[7,8] (Trf.8,Trf.9) -> G_global[7,9] += 0
    G: Trf[7,9] (Trf.8,Trf.10) -> G_global[7,3] += 0
    Ihis: Trf[8] (Trf.9) -> global[9] += -Ihisb_rc
    G: Trf[8,0] (Trf.9,Trf.1) -> G_global[9,11] += 0
    G: Trf[8,1] (Trf.9,Trf.2) -> G_global[9,6] += 0
    G: Trf[8,2] (Trf.9,Trf.3) -> G_global[9,10] += 0
    G: Trf[8,3] (Trf.9,Trf.4) -> G_global[9,4] += 0
    G: Trf[8,4] (Trf.9,Trf.5) -> G_global[9,0] += -Grc
    G: Trf[8,5] (Trf.9,Trf.6) -> G_global[9,1] += 0
    G: Trf[8,6] (Trf.9,Trf.7) -> G_global[9,2] += 0
    G: Trf[8,7] (Trf.9,Trf.8) -> G_global[9,7] += 0
    G: Trf[8,8] (Trf.9,Trf.9) -> G_global[9,9] += Grc
    G: Trf[8,9] (Trf.9,Trf.10) -> G_global[9,3] += 0
    Ihis: Trf[9] (Trf.10) -> global[3] += -Ihisc_rc
    G: Trf[9,0] (Trf.10,Trf.1) -> G_global[3,11] += 0
    G: Trf[9,1] (Trf.10,Trf.2) -> G_global[3,6] += 0
    G: Trf[9,2] (Trf.10,Trf.3) -> G_global[3,10] += 0
    G: Trf[9,3] (Trf.10,Trf.4) -> G_global[3,4] += 0
    G: Trf[9,4] (Trf.10,Trf.5) -> G_global[3,0] += 0
    G: Trf[9,5] (Trf.10,Trf.6) -> G_global[3,1] += -Grc
    G: Trf[9,6] (Trf.10,Trf.7) -> G_global[3,2] += 0
    G: Trf[9,7] (Trf.10,Trf.8) -> G_global[3,7] += 0
    G: Trf[9,8] (Trf.10,Trf.9) -> G_global[3,9] += 0
    G: Trf[9,9] (Trf.10,Trf.10) -> G_global[3,3] += Grc
    Ihis: UCM[0] (UCM.1) -> global[4] += UCMhis1
    G: UCM[0,0] (UCM.1,UCM.1) -> G_global[4,4] += AA
    G: UCM[0,1] (UCM.1,UCM.2) -> G_global[4,0] += 0
    G: UCM[0,2] (UCM.1,UCM.3) -> G_global[4,1] += 0
    G: UCM[0,3] (UCM.1,UCM.4) -> G_global[4,8] += AP
    G: UCM[0,4] (UCM.1,UCM.5) -> G_global[4,5] += AN
    Ihis: UCM[1] (UCM.2) -> global[0] += UCMhis2
    G: UCM[1,0] (UCM.2,UCM.1) -> G_global[0,4] += 0
    G: UCM[1,1] (UCM.2,UCM.2) -> G_global[0,0] += BB
    G: UCM[1,2] (UCM.2,UCM.3) -> G_global[0,1] += 0
    G: UCM[1,3] (UCM.2,UCM.4) -> G_global[0,8] += BP
    G: UCM[1,4] (UCM.2,UCM.5) -> G_global[0,5] += BN
    Ihis: UCM[2] (UCM.3) -> global[1] += UCMhis3
    G: UCM[2,0] (UCM.3,UCM.1) -> G_global[1,4] += 0
    G: UCM[2,1] (UCM.3,UCM.2) -> G_global[1,0] += 0
    G: UCM[2,2] (UCM.3,UCM.3) -> G_global[1,1] += CC
    G: UCM[2,3] (UCM.3,UCM.4) -> G_global[1,8] += CP
    G: UCM[2,4] (UCM.3,UCM.5) -> G_global[1,5] += CN
    Ihis: UCM[3] (UCM.4) -> global[8] += UCMhis4
    G: UCM[3,0] (UCM.4,UCM.1) -> G_global[8,4] += AP
    G: UCM[3,1] (UCM.4,UCM.2) -> G_global[8,0] += BP
    G: UCM[3,2] (UCM.4,UCM.3) -> G_global[8,1] += CP
    G: UCM[3,3] (UCM.4,UCM.4) -> G_global[8,8] += PP
    G: UCM[3,4] (UCM.4,UCM.5) -> G_global[8,5] += PN
    Ihis: UCM[4] (UCM.5) -> global[5] += UCMhis5
    G: UCM[4,0] (UCM.5,UCM.1) -> G_global[5,4] += AN
    G: UCM[4,1] (UCM.5,UCM.2) -> G_global[5,0] += BN
    G: UCM[4,2] (UCM.5,UCM.3) -> G_global[5,1] += CN
    G: UCM[4,3] (UCM.5,UCM.4) -> G_global[5,8] += PN
    G: UCM[4,4] (UCM.5,UCM.5) -> G_global[5,5] += NN
    
    ========== Global equation ==========
    I_global = G_global * V_global + Ihis_global
    
    ========== System Equation ==========
    
    


$\displaystyle \left[\begin{matrix}I_{Trf_5}\\I_{Trf_6}\\I_{Trf_7}\\I_{Trf_10}\\I_{Trf_4}\\I_{UCM_5}\\I_{Trf_2}\\I_{Trf_8}\\I_{UCM_4}\\I_{Trf_9}\\I_{Trf_3}\\I_{Trf_1}\end{matrix}\right] = \left[\begin{array}{cccccccccccc}BB + G_{22} + Grc + w_{2} & 0 & - G_{22} - w_{2} & 0 & 0 & BN & G_{12} & 0 & BP & - Grc & - G_{12} & 0\\0 & CC + G_{22} + Grc + w_{2} & - G_{22} - w_{2} & - Grc & 0 & CN & 0 & 0 & CP & 0 & G_{12} & - G_{12}\\- G_{22} - w_{2} & - G_{22} - w_{2} & 3 G_{22} + 3 w_{2} & 0 & - G_{22} - w_{2} & 0 & 0 & 0 & 0 & 0 & 0 & 0\\0 & - Grc & 0 & Grc & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0\\0 & 0 & - G_{22} - w_{2} & 0 & AA + G_{22} + Grc + w_{2} & AN & - G_{12} & - Grc & AP & 0 & 0 & G_{12}\\BN & CN & 0 & 0 & AN & NN & 0 & 0 & PN & 0 & 0 & 0\\G_{12} & 0 & 0 & 0 & - G_{12} & 0 & 2 G_{11} + 2 w_{1} & 0 & 0 & 0 & - G_{11} - w_{1} & - G_{11} - w_{1}\\0 & 0 & 0 & 0 & - Grc & 0 & 0 & Grc & 0 & 0 & 0 & 0\\BP & CP & 0 & 0 & AP & PN & 0 & 0 & PP & 0 & 0 & 0\\- Grc & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & Grc & 0 & 0\\- G_{12} & G_{12} & 0 & 0 & 0 & 0 & - G_{11} - w_{1} & 0 & 0 & 0 & 2 G_{11} + 2 w_{1} & - G_{11} - w_{1}\\0 & - G_{12} & 0 & 0 & G_{12} & 0 & - G_{11} - w_{1} & 0 & 0 & 0 & - G_{11} - w_{1} & 2 G_{11} + 2 w_{1}\end{array}\right] \left[\begin{matrix}V_{Trf_5}\\V_{Trf_6}\\V_{Trf_7}\\V_{Trf_10}\\V_{Trf_4}\\V_{UCM_5}\\V_{Trf_2}\\V_{Trf_8}\\V_{UCM_4}\\V_{Trf_9}\\V_{Trf_3}\\V_{Trf_1}\end{matrix}\right] + \left[\begin{matrix}Ihisb + Ihisb_{rc} + UCMhis_{2}\\Ihisc + Ihisc_{rc} + UCMhis_{3}\\- Ihisa - Ihisb - Ihisc\\- Ihisc_{rc}\\Ihisa + Ihisa_{rc} + UCMhis_{1}\\UCMhis_{5}\\IhisB - IhisA\\- Ihisa_{rc}\\UCMhis_{4}\\- Ihisb_{rc}\\IhisC - IhisB\\IhisA - IhisC\end{matrix}\right]$


    
    ===== 第七步：选择外部节点，生成 reduced system =====
    可用节点别名如下：
    Trf: ['Trf1', 'Trf2', 'Trf3', 'Trf4', 'Trf5', 'Trf6', 'Trf7', 'Trf8', 'Trf9', 'Trf10']
    UCM: ['UCM1', 'UCM2', 'UCM3', 'UCM4', 'UCM5']
    

    请输入要保留的外部节点，多个用逗号分隔（例如 netR1,netL2）；若保留所有节点，输入 all：
     all
    

    
    当前 reduced system 的节点顺序为：
    1: gid=0, node=Trf.5
    2: gid=1, node=Trf.6
    3: gid=2, node=Trf.7
    4: gid=3, node=Trf.10
    5: gid=4, node=Trf.4
    6: gid=5, node=UCM.5
    7: gid=6, node=Trf.2
    8: gid=7, node=Trf.8
    9: gid=8, node=UCM.4
    10: gid=9, node=Trf.9
    11: gid=10, node=Trf.3
    12: gid=11, node=Trf.1
    
    ========== Reduced External System Equation ==========
    
    


$\displaystyle \left[\begin{matrix}I_{Trf_5}\\I_{Trf_6}\\I_{Trf_7}\\I_{Trf_10}\\I_{Trf_4}\\I_{UCM_5}\\I_{Trf_2}\\I_{Trf_8}\\I_{UCM_4}\\I_{Trf_9}\\I_{Trf_3}\\I_{Trf_1}\end{matrix}\right] = \left[\begin{array}{cccccccccccc}BB + G_{22} + Grc + w_{2} & 0 & - G_{22} - w_{2} & 0 & 0 & BN & G_{12} & 0 & BP & - Grc & - G_{12} & 0\\0 & CC + G_{22} + Grc + w_{2} & - G_{22} - w_{2} & - Grc & 0 & CN & 0 & 0 & CP & 0 & G_{12} & - G_{12}\\- G_{22} - w_{2} & - G_{22} - w_{2} & 3 G_{22} + 3 w_{2} & 0 & - G_{22} - w_{2} & 0 & 0 & 0 & 0 & 0 & 0 & 0\\0 & - Grc & 0 & Grc & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0\\0 & 0 & - G_{22} - w_{2} & 0 & AA + G_{22} + Grc + w_{2} & AN & - G_{12} & - Grc & AP & 0 & 0 & G_{12}\\BN & CN & 0 & 0 & AN & NN & 0 & 0 & PN & 0 & 0 & 0\\G_{12} & 0 & 0 & 0 & - G_{12} & 0 & 2 G_{11} + 2 w_{1} & 0 & 0 & 0 & - G_{11} - w_{1} & - G_{11} - w_{1}\\0 & 0 & 0 & 0 & - Grc & 0 & 0 & Grc & 0 & 0 & 0 & 0\\BP & CP & 0 & 0 & AP & PN & 0 & 0 & PP & 0 & 0 & 0\\- Grc & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & Grc & 0 & 0\\- G_{12} & G_{12} & 0 & 0 & 0 & 0 & - G_{11} - w_{1} & 0 & 0 & 0 & 2 G_{11} + 2 w_{1} & - G_{11} - w_{1}\\0 & - G_{12} & 0 & 0 & G_{12} & 0 & - G_{11} - w_{1} & 0 & 0 & 0 & - G_{11} - w_{1} & 2 G_{11} + 2 w_{1}\end{array}\right] \left[\begin{matrix}V_{Trf_5}\\V_{Trf_6}\\V_{Trf_7}\\V_{Trf_10}\\V_{Trf_4}\\V_{UCM_5}\\V_{Trf_2}\\V_{Trf_8}\\V_{UCM_4}\\V_{Trf_9}\\V_{Trf_3}\\V_{Trf_1}\end{matrix}\right] + \left[\begin{matrix}Ihisb + Ihisb_{rc} + UCMhis_{2}\\Ihisc + Ihisc_{rc} + UCMhis_{3}\\- Ihisa - Ihisb - Ihisc\\- Ihisc_{rc}\\Ihisa + Ihisa_{rc} + UCMhis_{1}\\UCMhis_{5}\\- IhisA + IhisB\\- Ihisa_{rc}\\UCMhis_{4}\\- Ihisb_{rc}\\- IhisB + IhisC\\IhisA - IhisC\end{matrix}\right]$


    
    是否要更改 reduced system 中节点显示顺序？(y/n):  y
    

    
    请输入新的顺序，必须是当前 reduced system 已包含的节点，只能重排，不能增删。
    可以输入 global_nodes 中的真实名字，也可以输入 alias。
    例如: netR1,netL2  或  netR.1,netL.2
    

    新的节点顺序:  Trf1,Trf2,Trf3,Trf4,Trf5,Trf6,Trf7,Trf8,Trf9,Trf10,UCM4,UCM5
    

    
    重排后的 reduced system：
    
    ========== Reduced External System Equation ==========
    
    


$\displaystyle \left[\begin{matrix}I_{Trf_1}\\I_{Trf_2}\\I_{Trf_3}\\I_{Trf_4}\\I_{Trf_5}\\I_{Trf_6}\\I_{Trf_7}\\I_{Trf_8}\\I_{Trf_9}\\I_{Trf_10}\\I_{UCM_4}\\I_{UCM_5}\end{matrix}\right] = \left[\begin{array}{cccccccccccc}2 G_{11} + 2 w_{1} & - G_{11} - w_{1} & - G_{11} - w_{1} & G_{12} & 0 & - G_{12} & 0 & 0 & 0 & 0 & 0 & 0\\- G_{11} - w_{1} & 2 G_{11} + 2 w_{1} & - G_{11} - w_{1} & - G_{12} & G_{12} & 0 & 0 & 0 & 0 & 0 & 0 & 0\\- G_{11} - w_{1} & - G_{11} - w_{1} & 2 G_{11} + 2 w_{1} & 0 & - G_{12} & G_{12} & 0 & 0 & 0 & 0 & 0 & 0\\G_{12} & - G_{12} & 0 & AA + G_{22} + Grc + w_{2} & 0 & 0 & - G_{22} - w_{2} & - Grc & 0 & 0 & AP & AN\\0 & G_{12} & - G_{12} & 0 & BB + G_{22} + Grc + w_{2} & 0 & - G_{22} - w_{2} & 0 & - Grc & 0 & BP & BN\\- G_{12} & 0 & G_{12} & 0 & 0 & CC + G_{22} + Grc + w_{2} & - G_{22} - w_{2} & 0 & 0 & - Grc & CP & CN\\0 & 0 & 0 & - G_{22} - w_{2} & - G_{22} - w_{2} & - G_{22} - w_{2} & 3 G_{22} + 3 w_{2} & 0 & 0 & 0 & 0 & 0\\0 & 0 & 0 & - Grc & 0 & 0 & 0 & Grc & 0 & 0 & 0 & 0\\0 & 0 & 0 & 0 & - Grc & 0 & 0 & 0 & Grc & 0 & 0 & 0\\0 & 0 & 0 & 0 & 0 & - Grc & 0 & 0 & 0 & Grc & 0 & 0\\0 & 0 & 0 & AP & BP & CP & 0 & 0 & 0 & 0 & PP & PN\\0 & 0 & 0 & AN & BN & CN & 0 & 0 & 0 & 0 & PN & NN\end{array}\right] \left[\begin{matrix}V_{Trf_1}\\V_{Trf_2}\\V_{Trf_3}\\V_{Trf_4}\\V_{Trf_5}\\V_{Trf_6}\\V_{Trf_7}\\V_{Trf_8}\\V_{Trf_9}\\V_{Trf_10}\\V_{UCM_4}\\V_{UCM_5}\end{matrix}\right] + \left[\begin{matrix}IhisA - IhisC\\- IhisA + IhisB\\- IhisB + IhisC\\Ihisa + Ihisa_{rc} + UCMhis_{1}\\Ihisb + Ihisb_{rc} + UCMhis_{2}\\Ihisc + Ihisc_{rc} + UCMhis_{3}\\- Ihisa - Ihisb - Ihisc\\- Ihisa_{rc}\\- Ihisb_{rc}\\- Ihisc_{rc}\\UCMhis_{4}\\UCMhis_{5}\end{matrix}\right]$


    
    ========== Reduced System Python Format ==========
    
    G_reduced_final = sp.Matrix([[2*G11 + 2*w1, -G11 - w1, -G11 - w1, G12, 0, -G12, 0, 0, 0, 0, 0, 0], [-G11 - w1, 2*G11 + 2*w1, -G11 - w1, -G12, G12, 0, 0, 0, 0, 0, 0, 0], [-G11 - w1, -G11 - w1, 2*G11 + 2*w1, 0, -G12, G12, 0, 0, 0, 0, 0, 0], [G12, -G12, 0, AA + G22 + Grc + w2, 0, 0, -G22 - w2, -Grc, 0, 0, AP, AN], [0, G12, -G12, 0, BB + G22 + Grc + w2, 0, -G22 - w2, 0, -Grc, 0, BP, BN], [-G12, 0, G12, 0, 0, CC + G22 + Grc + w2, -G22 - w2, 0, 0, -Grc, CP, CN], [0, 0, 0, -G22 - w2, -G22 - w2, -G22 - w2, 3*G22 + 3*w2, 0, 0, 0, 0, 0], [0, 0, 0, -Grc, 0, 0, 0, Grc, 0, 0, 0, 0], [0, 0, 0, 0, -Grc, 0, 0, 0, Grc, 0, 0, 0], [0, 0, 0, 0, 0, -Grc, 0, 0, 0, Grc, 0, 0], [0, 0, 0, AP, BP, CP, 0, 0, 0, 0, PP, PN], [0, 0, 0, AN, BN, CN, 0, 0, 0, 0, PN, NN]])
    
    Ihis_reduced_final = sp.Matrix([[IhisA - IhisC], [-IhisA + IhisB], [-IhisB + IhisC], [Ihisa + Ihisa_rc + UCMhis1], [Ihisb + Ihisb_rc + UCMhis2], [Ihisc + Ihisc_rc + UCMhis3], [-Ihisa - Ihisb - Ihisc], [-Ihisa_rc], [-Ihisb_rc], [-Ihisc_rc], [UCMhis4], [UCMhis5]])
    
    ===== 自动生成的可复制代码 =====
    
    # -------- 子网络：Trf --------
    G_Trf = sp.Matrix([[2*G11 + 2*w1, -G11 - w1, -G11 - w1, G12, 0, -G12, 0, 0, 0, 0], [-G11 - w1, 2*G11 + 2*w1, -G11 - w1, -G12, G12, 0, 0, 0, 0, 0], [-G11 - w1, -G11 - w1, 2*G11 + 2*w1, 0, -G12, G12, 0, 0, 0, 0], [G12, -G12, 0, G22 + Grc + w2, 0, 0, -G22 - w2, -Grc, 0, 0], [0, G12, -G12, 0, G22 + Grc + w2, 0, -G22 - w2, 0, -Grc, 0], [-G12, 0, G12, 0, 0, G22 + Grc + w2, -G22 - w2, 0, 0, -Grc], [0, 0, 0, -G22 - w2, -G22 - w2, -G22 - w2, 3*G22 + 3*w2, 0, 0, 0], [0, 0, 0, -Grc, 0, 0, 0, Grc, 0, 0], [0, 0, 0, 0, -Grc, 0, 0, 0, Grc, 0], [0, 0, 0, 0, 0, -Grc, 0, 0, 0, Grc]])
    Ihis_Trf = sp.Matrix([[IhisA - IhisC], [-IhisA + IhisB], [-IhisB + IhisC], [Ihisa + Ihisa_rc], [Ihisb + Ihisb_rc], [Ihisc + Ihisc_rc], [-Ihisa - Ihisb - Ihisc], [-Ihisa_rc], [-Ihisb_rc], [-Ihisc_rc]])
    Trf = SubNetwork(
        name="Trf",
        local_nodes=['1', '2', '3', '4', '5', '6', '7', '8', '9', '10'],
        G_local=G_Trf,
        Ihis_local=Ihis_Trf
    )
    
    # -------- 子网络：UCM --------
    G_UCM = sp.Matrix([[AA, 0, 0, AP, AN], [0, BB, 0, BP, BN], [0, 0, CC, CP, CN], [AP, BP, CP, PP, PN], [AN, BN, CN, PN, NN]])
    Ihis_UCM = sp.Matrix([[UCMhis1], [UCMhis2], [UCMhis3], [UCMhis4], [UCMhis5]])
    UCM = SubNetwork(
        name="UCM",
        local_nodes=['1', '2', '3', '4', '5'],
        G_local=G_UCM,
        Ihis_local=Ihis_UCM
    )
    
    # -------- 装配 --------
    asm = NetworkAssembler()
    asm.add_subnetwork(Trf)
    asm.add_subnetwork(UCM)
    asm.connect("Trf.4", "UCM.1")
    asm.connect("Trf.5", "UCM.2")
    asm.connect("Trf.6", "UCM.3")
    
    assembled = asm.assemble(verbose=True)
    pretty_display_system_split(assembled)
    
    # -------- 选择外部节点并消元 --------
    ext_ids = list(range(len(assembled["global_nodes"])))
    reduced = asm.eliminate_internal_nodes(assembled, external_node_ids=ext_ids)
    pretty_display_reduced_system(assembled, reduced)
    
    # -------- 重排 reduced system 节点顺序 --------
    reduced = reorder_reduced_system(assembled, reduced, [11, 6, 10, 4, 0, 1, 2, 7, 9, 3, 8, 5])
    pretty_display_reduced_system(assembled, reduced)
    
    # -------- reduced system 的 Python 格式 --------
    G_reduced_final = sp.Matrix([[2*G11 + 2*w1, -G11 - w1, -G11 - w1, G12, 0, -G12, 0, 0, 0, 0, 0, 0], [-G11 - w1, 2*G11 + 2*w1, -G11 - w1, -G12, G12, 0, 0, 0, 0, 0, 0, 0], [-G11 - w1, -G11 - w1, 2*G11 + 2*w1, 0, -G12, G12, 0, 0, 0, 0, 0, 0], [G12, -G12, 0, AA + G22 + Grc + w2, 0, 0, -G22 - w2, -Grc, 0, 0, AP, AN], [0, G12, -G12, 0, BB + G22 + Grc + w2, 0, -G22 - w2, 0, -Grc, 0, BP, BN], [-G12, 0, G12, 0, 0, CC + G22 + Grc + w2, -G22 - w2, 0, 0, -Grc, CP, CN], [0, 0, 0, -G22 - w2, -G22 - w2, -G22 - w2, 3*G22 + 3*w2, 0, 0, 0, 0, 0], [0, 0, 0, -Grc, 0, 0, 0, Grc, 0, 0, 0, 0], [0, 0, 0, 0, -Grc, 0, 0, 0, Grc, 0, 0, 0], [0, 0, 0, 0, 0, -Grc, 0, 0, 0, Grc, 0, 0], [0, 0, 0, AP, BP, CP, 0, 0, 0, 0, PP, PN], [0, 0, 0, AN, BN, CN, 0, 0, 0, 0, PN, NN]])
    Ihis_reduced_final = sp.Matrix([[IhisA - IhisC], [-IhisA + IhisB], [-IhisB + IhisC], [Ihisa + Ihisa_rc + UCMhis1], [Ihisb + Ihisb_rc + UCMhis2], [Ihisc + Ihisc_rc + UCMhis3], [-Ihisa - Ihisb - Ihisc], [-Ihisa_rc], [-Ihisb_rc], [-Ihisc_rc], [UCMhis4], [UCMhis5]])
    


```python

```

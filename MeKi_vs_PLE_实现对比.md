# MeKi vs PLE 实现级深度对比

## 0. 结论先行（1页）

### 0.1 一句话结论
- `PLE` 与 `MeKi` 都是“每层注入”的增强分支，但它们的核心差异不在“是否逐层”，而在：
  - 注入位置：`PLE` 为 layer 后挂分支，`MeKi` 为 MLP 内联分支。
  - 融合算子：`PLE` 为乘性门控（`*`），`MeKi` 为加性混合（`+`）。

### 0.2 实现层核心判断
- 在 `Qwen3+PLE`（`modeling_nebula_ple.py`）中，先执行父类完整层前向，再执行 PLE 分支并做 residual add。  
  证据：`super().forward(...)` 后才进入 `gate * per_layer_input` 分支。
- 在 `Qwen3+MeKi`（`modeling_qwen3.py`）中，MeKi 在 MLP 路径中插入：`mlp -> + meki_delta -> residual add`。  
  证据：`hidden_states = self.mlp(...)` 之后立刻执行 `hidden_states = hidden_states + self.meki_alpha * meki_output`。

### 0.3 工程选型直觉
- 追求“对外部层输入信号的强约束调制”：优先 `PLE`（乘性门控）。
- 追求“对 MLP 表达的补充偏移、改动更内聚”：优先 `MeKi`（加性偏移）。
- 若考虑二者共存：可行，但需重点防范残差叠加过强导致训练不稳（详见第 3.5 节）。

---

## 1. 比较范围与证据源

### 1.1 对比范围（锁定）
- 主对比对象：
  - `modeling_nebula_ple.py`（Qwen3+PLE）
  - `modeling_qwen3.py`（Qwen3+MeKi）
- 参考实现：
  - `gemma4/modeling_gemma4.py`（PLE 基线）
  - `configuration_qwen3.py`
  - `configuration_nebula_ple.py`
  - `gemma4/configuration_gemma4.py`

### 1.2 方法说明
- 所有“实现差异”均给出路径+行号。
- 无法被源码直接证实的结论，显式标注为“推断”。

### 1.3 术语约定
- `L`: `num_hidden_layers`
- `H`: `hidden_size`
- `P`: `hidden_size_per_layer_input`（PLE 维度）
- `M`: `meki_dim`（MeKi 维度）
- `B,S`: batch、seq length

---

## 2. 工程评审视角

### 2.1 共同点矩阵

| 维度 | PLE（Qwen3+PLE） | MeKi（Qwen3+MeKi） | 结论 |
|---|---|---|---|
| 每层输入生成范式 | `token_lookup + context_projection` | `token_lookup + context_projection` | 结构同构 |
| 打包嵌入（packed table） | `Embedding(vocab_ple, L*P)` | `Embedding(vocab, L*M)` | 都是单表打包多层 |
| 上下文投影 | `Linear(H -> L*P)` + Norm | `Linear(H -> L*M)` + Norm | 都有全局一次投影 |
| 按层切片 | `per_layer_inputs[:, :, i, :]` | `meki_layer_inputs[:, :, i, :]` | 都是 layer-wise slice |
| 混合缩放 | `* 2^-0.5` | `* 2^-0.5` | 同步使用缩放 |

代码证据：
- PLE 全局构造与切片：`modeling_nebula_ple.py:99,104,109-111,138-146,177`
- MeKi 全局构造与切片：`modeling_qwen3.py:364-376,449-463,471`

### 2.2 差异项矩阵（核心）

| 差异项 | PLE（Qwen3+PLE） | MeKi（Qwen3+MeKi） | 工程影响 |
|---|---|---|---|
| 注入位置 | layer 完整前向后再注入 | MLP 输出处内联注入 | MeKi 与 MLP 耦合更紧 |
| 层内融合算子 | `gate(h) * per_layer_input`（乘性） | `sigmoid(Wg h) + meki_embedding`（加性） | PLE 更“筛选”，MeKi 更“偏移” |
| 分支投影 | `P -> H` | `M -> H` | 两者均有回投影 GEMM |
| 分支归一化 | PLE 分支末端 `LayerNorm(H)` | `meki_input` 先 `LN(M)`，输出再 `LN(H)` | MeKi 归一化节点更多 |
| 系数语义 | PLE 分支无 `beta`（当前 Qwen3+PLE 实现） | `beta` 控制 context 路权重，`alpha` 控制注入强度 | MeKi 可调节度更高 |
| 输入约束 | 依赖 `input_ids` 获取 `embed_tokens_per_layer` | 依赖 `input_ids` 获取 `embed_tokens_meki` | 两者都不支持“无 token id”直通 |

代码证据：
- PLE 注入位置与算子：`modeling_nebula_ple.py:57-67,71-74`
- MeKi 注入位置与算子：`modeling_qwen3.py:284-291`
- 配置项（MeKi）：`configuration_qwen3.py:178-180,207-209`
- 配置项（PLE）：`configuration_nebula_ple.py:12-20`

### 2.3 关键路径伪代码（实现等价）

#### PLE（Qwen3+PLE）
```python
# 全局一次
ple_token   = EmbeddingPLE(input_ids).reshape(B,S,L,P)
ple_context = Norm(LinearH_to_LP(inputs_embeds).reshape(B,S,L,P))
ple_inputs  = (ple_token + ple_context) * 2**-0.5

# 每层
h = BaseDecoderLayerForward(h)   # attention + mlp + residual
gate  = GELU(Wg(h))              # H -> P
delta = Wp(gate * ple_inputs[:,:,i,:])  # P -> H
h = h + LN(delta)
```

#### MeKi（Qwen3+MeKi）
```python
# 全局一次
meki_token   = EmbeddingMeKi(input_ids).reshape(B,S,L,M)
meki_context = Norm(LinearH_to_LM(inputs_embeds).reshape(B,S,L,M))
meki_inputs  = (meki_token + beta * meki_context) * 2**-0.5

# 每层（在 MLP 内）
pre = post_attention_layernorm(h)
mlp_out = MLP(pre)
meki_fused = sigmoid(Wg(pre)) + LN(meki_inputs[:,:,i,:])  # H->M then add
delta = LN(Wo(meki_fused))  # M->H
h = residual_after_attn + (mlp_out + alpha * delta)
```

### 2.4 参数量与显存增量（估算）

只统计增量主项（忽略 bias/Norm 的小量）：

- PLE 增量约：
  - `Embedding`: `V_ple * (L*P)`
  - `Global projection`: `H * (L*P)`
  - `Per-layer`: `L * (H*P + P*H)`
- MeKi 增量约：
  - `Embedding`: `V * (L*M)`
  - `Global projection`: `H * (L*M)`
  - `Per-layer`: `L * (H*M + M*H)`

结论：
- 若 `V_ple≈V`，两者规模量级主要由 `P` vs `M` 决定。
- 额外访存主导项均来自“打包 embedding 表”。

### 2.5 推理开销（额外算子）
- 共同新增：
  - 1 次全局 `Linear(H -> L*D)`（D 为 P 或 M）
  - 每层 2 次小线性（`H->D`、`D->H`）和若干逐元素算子
- PLE 特有：
  - 每层乘性门控 `gate * per_layer_input`
- MeKi 特有：
  - 每层加性混合 `sigmoid(Wg(pre)) + norm(meki_input)`
  - 多一个可调 `beta` 作用于全局融合

### 2.6 Checkpoint 映射与兼容风险
- 由于模块名与参数名不同，映射必须按结构区分（例如 `per_layer_*` vs `meki_*`）。
- 若错误映射：
  - 轻则加载缺失/未使用参数。
  - 重则出现形状不匹配或 silent mismatch（模型可跑但行为漂移）。
- 建议：
  - 转换脚本强制断言 key 覆盖率、shape 一致率、数值 sanity（均值/方差）三项。

### 2.7 训练稳定性风险与监控项
- 共同风险：
  - 残差新增分支导致激活幅度变大，早期训练易出现梯度抖动。
- PLE 侧重点：
  - 乘性门控在 gate 接近 0 时可能导致分支信息熄灭。
- MeKi 侧重点：
  - 加性融合与 `alpha/beta` 联动不当，可能引起分支过强覆盖主干。
- 建议监控：
  - 分支输出范数占比 `||delta|| / ||main||`
  - 梯度 global norm、NaN/Inf 计数
  - `alpha/beta` 扫描下的 loss 曲线斜率与波动

---

## 3. 论文分析视角

### 3.1 统一形式化
- 主干层映射记为：
  - `h_l^base = F_l(h_{l-1})`
- 外部逐层输入记为：
  - PLE 输入 `e_l in R^P`
  - MeKi 输入 `m_l in R^M`

### 3.2 两种结构的形式表达

#### PLE（Qwen3+PLE）
- 全局输入构造：
  - `e_l = (E_l(token) + C_l(h_0)) / sqrt(2)`
- 层内注入：
  - `g_l = phi(W_l^g h_l^base)`
  - `Delta_l = W_l^p (g_l ⊙ e_l)`
  - `h_l = h_l^base + Norm(Delta_l)`

对应实现：`phi=GELU`，`⊙` 为逐元素乘法。

#### MeKi（Qwen3+MeKi）
- 全局输入构造：
  - `m_l = (M_l(token) + beta * C_l(h_0)) / sqrt(2)`
- 层内注入（MLP 内联）：
  - `u_l = sigmoid(W_l^g pre_l) + Norm(m_l)`
  - `Delta_l = Norm(W_l^o u_l)`
  - `h_l = h_l^attn_res + (MLP(pre_l) + alpha * Delta_l)`

### 3.3 信息流差异
- PLE：外部信号先被主干状态门控后再注入，属于“条件调制”。
- MeKi：外部信号与门控结果做加法，属于“偏移补偿”。
- 直观上：
  - PLE 对 token 外部锚点更“选择性通过”。
  - MeKi 对 MLP 表达更“平滑叠加”。

### 3.4 梯度通路对比
- PLE 梯度重点路径：
  - `e_l -> (g_l ⊙ e_l) -> Delta_l -> h_l`
  - 受 `g_l` 影响更强，门控会改变外部路径有效增益。
- MeKi 梯度重点路径：
  - `m_l -> Norm(m_l) -> u_l -> Delta_l -> h_l`
  - 与 `sigmoid(Wg pre_l)` 加性并联，路径通常更连续。

### 3.5 归类与可组合性
- 归类：
  - PLE 更接近“乘性条件 adapter”。
  - MeKi 更接近“加性 memory adapter”。
- 可组合性（推断）：
  - 两者可并存，前提是控制总分支强度，避免对主干 residual 形成过强覆盖。
  - 推荐采用分阶段启用或系数 warmup（`alpha/beta`、PLE 分支系数）以稳定训练。

---

## 4. 统一结论与选型建议

### 4.1 三类场景建议

#### 场景A：低时延 / 边缘推理优先
- 推荐：优先 `PLE`（`P` 取较小值）。
- 理由：乘性门控结构简单，参数路径直观。
- 风险：门控过抑制导致分支贡献低。

#### 场景B：结构化推理与稳定训练优先
- 推荐：优先 `MeKi`，并启用温和 `alpha/beta`。
- 理由：加性混合对 MLP 表达补偿更直接，可调节维度更多。
- 风险：`alpha/beta` 过大导致主干被覆盖。

#### 场景C：多模态对齐与可解释调制优先
- 推荐：`PLE` 或 `PLE + MeKi`（分阶段启用）。
- 理由：PLE 的“门控×外部输入”解释性更强；MeKi 可补偿 MLP 内表达。
- 风险：双分支叠加导致训练动态复杂化。

### 4.2 建议超参起点（工程起步）
- PLE 起点：
  - `hidden_size_per_layer_input`: 128 或 256
  - 若有分支系数：从 0.1~0.3 warmup 到目标值
- MeKi 起点：
  - `meki_dim`: 128 或 256
  - `meki_alpha`: 0.3~0.8
  - `meki_beta`: 0.5~1.0

> 说明：以上为工程经验起点，非源码固定配置。

---

## 附录A：代码证据索引（路径+行号）

### A.1 Qwen3+PLE（`modeling_nebula_ple.py`）
- PLE 层分支定义：
  - `NebulaPLEDecoderLayer`：`32`
  - `per_layer_input_gate`：`38`
  - `per_layer_projection`：`39`
  - `post_per_layer_input_norm`：`41`
- PLE 注入位置与算子：
  - `super().forward(...)` 后注入：`57-67`
  - `gate = ...`：`71`
  - `gate * per_layer_input`：`72`
  - `projection/norm`：`73-74`
- 全局 PLE 输入构造：
  - `embed_tokens_per_layer`：`99`
  - `per_layer_model_projection`：`104`
  - `per_layer_projection_norm`：`109`
  - 缩放项：`110-111`
  - `ple_token`：`138`
  - `ple_context`：`141-145`
  - 融合：`146`
  - 按层切片：`177`
- 输入约束：
  - 需要 `input_ids`：`137`

### A.2 Qwen3+MeKi（`modeling_qwen3.py`）
- MeKi 层分支定义：
  - `Qwen3DecoderLayer`：`233`
  - `meki_dim/use_meki`：`244-245`
  - `meki_alpha`：`247`
  - `meki_gate_proj`：`248`
  - `meki_out_proj`：`249`
  - `meki_mix_norm/meki_post_norm`：`250-251`
- MeKi 注入位置与算子：
  - `pre_mlp_hidden_states`：`284`
  - `hidden_states = self.mlp(...)`：`285`
  - `meki_fused = sigmoid(...) + meki_embedding`：`288`
  - `hidden_states += alpha * meki_output`：`291`
- 全局 MeKi 输入构造：
  - 模型级 `meki_dim/alpha/beta`：`357-360`
  - `embed_tokens_meki`：`364-366`
  - `meki_model_projection`：`369-371`
  - `meki_projection_norm`：`374`
  - 缩放项：`375-376`
  - `meki_token`：`449`
  - `meki_context`：`455-462`
  - 融合：`463`
  - 按层切片：`471`

### A.3 PLE 参考实现（`gemma4/modeling_gemma4.py`）
- 层内 PLE 分支定义与注入：
  - `hidden_size_per_layer_input`：`1350-1351`
  - `per_layer_input_gate/projection/norm`：`1353-1355`
  - 注入逻辑：`1412, 1414, 1417-1418`
- 全局 PLE 输入构造：
  - `embed_tokens_per_layer`：`1579-1581`
  - `per_layer_model_projection`：`1586-1588`
  - `per_layer_projection_norm`：`1592`
  - `per_layer_input_scale`：`1585`
  - `get_per_layer_inputs`：`1693`
  - `project_per_layer_inputs`：`1737`
  - 融合返回：`1770`
  - 按层切片：`1673`

### A.4 配置定义证据
- MeKi 配置：
  - `configuration_qwen3.py:178-180, 207-209`
- PLE 配置（Nebula）：
  - `configuration_nebula_ple.py:12-20`
- PLE 配置（Gemma4）：
  - `gemma4/configuration_gemma4.py:93-99,170-171`

---

## 附录B：符号表与术语映射

| 工程术语 | 数学符号 | 含义 |
|---|---|---|
| `hidden_states` | `h_l` | 第 `l` 层主干隐状态 |
| `pre_mlp_hidden_states` | `pre_l` | MLP 入口状态 |
| `per_layer_input` | `e_l` | PLE 每层外部输入 |
| `meki_input` | `m_l` | MeKi 每层外部输入 |
| `per_layer_input_gate` | `W_l^g` | PLE 门控投影 |
| `per_layer_projection` | `W_l^p` | PLE 回投影 |
| `meki_gate_proj` | `W_l^g` | MeKi 门控投影 |
| `meki_out_proj` | `W_l^o` | MeKi 回投影 |
| `per_layer_projection_norm` | `Norm_c` | PLE context 路归一化 |
| `meki_projection_norm` | `Norm_c` | MeKi context 路归一化 |
| `post_per_layer_input_norm` | `Norm_o` | PLE 输出归一化 |
| `meki_post_norm` | `Norm_o` | MeKi 输出归一化 |
| `per_layer_input_scale` | `s` | PLE 全局融合缩放（通常 `2^-0.5`） |
| `meki_input_scale` | `s` | MeKi 全局融合缩放（通常 `2^-0.5`） |
| `meki_alpha` | `alpha` | MeKi 注入强度 |
| `meki_beta` | `beta` | MeKi context 路权重 |

---

## 附录C：图示化流程（结构数据流）

### C.1 PLE（Qwen3+PLE）流程图

```mermaid
flowchart TD
    A[input_ids] --> B[PLE Embedding: Embedding(vocab_ple, L*P)]
    B --> C[reshape -> B,S,L,P]
    D[inputs_embeds] --> E[Linear H->L*P]
    E --> F[scale 1/sqrt(H)]
    F --> G[reshape -> B,S,L,P]
    G --> H[LayerNorm(P)]
    C --> I[ple_inputs = (token + context) * 1/sqrt(2)]
    H --> I

    J[Base Decoder Layer Forward<br/>attn + mlp + residual] --> K[hidden_states]
    K --> L[per_layer_input_gate: H->P + GELU]
    I --> M[slice layer i: B,S,P]
    L --> N[elementwise mul: gate * per_layer_input_i]
    M --> N
    N --> O[per_layer_projection: P->H]
    O --> P[post_per_layer_input_norm: LN(H)]
    K --> Q[residual add]
    P --> Q
    Q --> R[output h_l]
```

### C.2 MeKi（Qwen3+MeKi）流程图

```mermaid
flowchart TD
    A[input_ids] --> B[MeKi Embedding: Embedding(vocab, L*M)]
    B --> C[reshape -> B,S,L,M]
    D[inputs_embeds] --> E[Linear H->L*M]
    E --> F[scale 1/sqrt(H)]
    F --> G[reshape -> B,S,L,M]
    G --> H[LayerNorm(M)]
    C --> I[meki_inputs = (token + beta*context) * 1/sqrt(2)]
    H --> I

    J[post_attention_layernorm output: pre_mlp] --> K[MLP(pre_mlp)]
    J --> L[meki_gate_proj: H->M]
    L --> M[sigmoid]
    I --> N[slice layer i: B,S,M]
    N --> O[meki_mix_norm: LN(M)]
    M --> P[add: sigmoid(gate) + norm(meki_input_i)]
    O --> P
    P --> Q[meki_out_proj: M->H]
    Q --> R[meki_post_norm: LN(H)]
    R --> S[mlp_out + alpha*meki_out]
    K --> S
    S --> T[residual add]
    T --> U[output h_l]
```

### C.3 并排差异图（注入点与融合算子）

```mermaid
flowchart LR
    subgraph PLE_Path[PLE Path]
        P1[base layer output h_base] --> P2[gate(H->P)+GELU]
        P3[per_layer_input_i] --> P4[mul: gate * input]
        P2 --> P4
        P4 --> P5[P->H proj]
        P5 --> P6[LN(H)]
        P6 --> P7[h_base + delta]
    end

    subgraph MeKi_Path[MeKi Path]
        M1[pre_mlp] --> M2[MLP]
        M1 --> M3[gate(H->M)+sigmoid]
        M4[meki_input_i] --> M5[LN(M)]
        M3 --> M6[add: gate + input]
        M5 --> M6
        M6 --> M7[M->H proj]
        M7 --> M8[LN(H)]
        M2 --> M9[mlp_out + alpha*delta]
        M8 --> M9
    end
```

### C.4 图示解读（工程简版）
- PLE：注入点在“整层输出之后”，外部输入通过乘性门控进入主干。
- MeKi：注入点在“MLP 分支内部”，外部输入与门控信号先加性融合，再回投影到主干维度。
- 二者共同拥有“token lookup + context projection + layer-wise slicing”的全局构造阶段。

---

## 备注
- 本文全部结论以当前仓库实现为准，不外推至所有 MeKi/PLE 变体。
- 关于“二者可组合”的部分属于工程推断，已在正文显式标注。

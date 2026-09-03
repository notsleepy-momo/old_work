### **关于 LLM 随机性对论文合理性影响的分析**

针对你提到的“多次运行代码寻找期望结果”的情况，从学术研究和系统设计的角度来看，**“结果不一致”本身是 LLM 的固有特性，但“靠手动筛选最佳结果”在论文中是不合理的。**

如果你的项目 `ymj_TS` 要发论文（特别是像 UbiComp 这种顶级会议），你需要将这种“不确定性”转化为“鲁棒性证明”。以下是详细的分析和改进建议：

### **1. 为什么“多次运行寻找期望结果”不合理？**
在学术论文中，实验必须是**可重复的（Reproducible）**。
*   **选择性偏差（Cherry-picking）：** 如果你只展示多次运行中最好的一次结果，而没有说明平均表现，评审人（Reviewer）会认为你的系统不可靠。
*   **缺乏科学性：** 论文需要证明的是你的**算法逻辑（如 5 步管线、约束协议）**能有效提升表现，而不是 LLM 本身的运气。

### **2. 这种情况能发论文吗？**
**答案是：可以发，但必须改变实验和描述方式。**
AI 领域的论文通常不要求 100% 的确定性，但要求**统计学上的意义**。你需要证明即使 LLM 有随机性，你的系统依然比基线（Baselines）更强。

---

### **3. 如何在论文中处理“结果不一致”？（改进方案）**

为了让你的项目符合论文发表标准，建议从以下四个维度进行优化：

#### **A. 引入“多次运行的统计评估”**
不要只跑一次。在 `evaluation.py` 中增加批量评估逻辑：
*   **运行 N 次（如 5-10 次）：** 对同一个户型图运行多次。
*   **报告平均值和标准差：** 在结果表（Table）中显示 `F1: 0.85 ± 0.03`。这证明了你的系统在波动中依然保持高水平。

#### **B. 强调“约束协议（Constrained Action Protocol）”的作用**
这正是你 README 中提到的核心亮点。你应该在论文中强调：
*   **减少幻觉：** 即使 LLM 结果不一，但因为你限制它只能执行 `merge/delete/adjust`（Step 2），它无法“发明”不存在的家具。
*   **硬规则兜底：** 强调你的代码硬规则（如 `_auto_merge_strong_signal_pairs`）锁定了确定性部分，LLM 只负责处理模糊边界。

#### **C. 增加“消融实验”证明稳定性**
利用你 README 里的消融实验：
*   对比 **Free-form LLM**（无约束，随机性极大且易出错） vs **Constrained LLM**（你的方法）。
*   证明你的方法显著降低了**结果的方差（Variance）**，让输出变得更可预测。

#### **D. 降低 Temperature 参数**
在调用 API 时（`agents/` 目录下），将 LLM 的 `temperature` 设置为 `0` 或接近 `0`（如 `0.1`）。
*   这会强制 LLM 选择概率最高的 Token，极大提高结果的**一致性（Determinism）**。

---

### **4. 论文写作建议：如何描述这个现象？**

在论文的 **Experimental Setup** 或 **Discussion** 部分，你可以这样写：

> "Recognizing the inherent stochastic nature of Large Language Models (LLMs), we implemented a **Constrained Action Protocol** to bound the LLM's reasoning within a predefined geometric space. To ensure the reliability of our findings, all experiments were conducted over 5 independent runs. Our system exhibited high stability, with a standard deviation of less than 3% across key metrics (RTA and F1 score), significantly outperforming unconstrained baselines."

### **总结建议**
1.  **不要手动选结果**，而是**降低 Temperature** 提高一致性。
2.  **跑 5 次取平均值**，作为你论文里的最终分数。
3.  **强调你的 5 步管线**（尤其是 CV 引导和硬规则）是如何通过“系统工程”的方法来驯服 LLM 的随机性的。

这种“用确定性的系统架构处理不确定性的 AI 模型”的设计思路，反而是一个非常好的**学术贡献点**。

*内容由 AI 生成仅供参考*
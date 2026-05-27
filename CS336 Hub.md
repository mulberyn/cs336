---
tags: [course, cs336, deep-learning]
aliases: [CS336学习计划]
---
# CS336 学习中心
> 🎯 **目标**：2 个月内（2026-05-26 → 2026-07-25）完成 Stanford CS336 春季全部讲座与编程作业  
> ⚠️ **休息日**：5月31日、6月2日 不安排学习  
> 📚 **课程**：Language Models from Scratch
---
## 📋 课程核心模块
| 阶段 | 讲座 | 作业 | 主题 |
|------|------|------|------|
| 1. 基础与架构 | L1–L6 | Assignment 1 | 分词、资源核算、架构、注意力变体、GPU/TPU、Kernel/Triton |
| 2. 并行与规模化 | L7–L12 | Assignment 2 & 3 | 并行策略、Scaling Laws、推理、评估 |
| 3. 数据与训练 | L13–L16 | Assignment 4 | 数据收集/清洗/混合、SFT/RLHF、RLVR |
| 4. 对齐与前沿 | L17–L19 | Assignment 5 | RL 对齐系统、嘉宾讲座（Daniel Selsam, Dan Fu） |
| 🎓 收官 | — | — | 综合回顾、成果总结 |

---
## 🗓️ 每日学习计划（59天）
### 🔹 阶段一：基础与架构 (Day 1–12)
- [x] **Day 1 · 5月26日 周二**  
	- **L1**: Overview, tokenization: [[lecture_01 - Overview, tokenization]]
	- 📂 `lecture_01.py`  
- [ ] **Day 2 · 5月27日 周三**  
	- **L2**: PyTorch (einops), resource accounting (FLOPs, memory, arithmetic intensity) [Percy]  
	- 📂 `lecture_02.py`  
- [ ] **Day 3 · 5月28日 周四**  
	- **L3**: Architectures, hyperparameters [Tatsu]  
	- 📄 `lecture 3.pdf`  
- [ ] **Day 4 · 5月29日 周五**  
	- **L4**: Attention alternatives and mixture of experts [Tatsu]  
	- 📄 `lecture 4.pdf`  
- [ ] **Day 5 · 5月30日 周六**  
	- **L5**: GPUs, TPUs [Tatsu]  
	- 📄 `lecture 5.pdf`  
- [ ] **5月31日 周日 · 休息** 🌴
- [ ] **Day 6 · 6月1日 周一**  
	- **L6**: Kernels, Triton, XLA [Percy]  
	- 📂 `lecture_06.py`  
- [ ] **6月2日 周二 · 休息** 🌴
- [ ] **Day 7 · 6月3日 周三**  
	- **Assignment 1 第1部分**：阅读要求，搭建环境，实现 Tokenization  
- [ ] **Day 8 · 6月4日 周四**  
	- **Assignment 1 第2部分**：完成核心代码（模型前向、资源统计）  
- [ ] **Day 9 · 6月5日 周五**  
	- **Assignment 1 第3部分**：调试、优化，撰写实验分析  
- [ ] **Day 10 · 6月6日 周六**  
	- **Assignment 1 第4部分**：完善文档，通过全部测试  
- [ ] **Day 11 · 6月7日 周日**  
	- **Assignment 1 第5部分**：最终复核，提交准备  
- [ ] **Day 12 · 6月8日 周一**  
	- **Assignment 1 第6部分 (收尾)**：提交作业 1 ✅
### 🔹 阶段二：并行与规模化 (Day 13–33)
- [ ] **Day 13 · 6月9日 周二**  
	- **L7**: Parallelism [Percy]  
	- 📂 `lecture_07.py`  
- [ ] **Day 14 · 6月10日 周三**  
	- **L8**: Parallelism [Tatsu]  
	- 📄 `lecture_08.pdf`  
- [ ] **Day 15 · 6月11日 周四**  
	- **Assignment 2 Day1**：数据并行与模型并行基础实现  
- [ ] **Day 16 · 6月12日 周五**  
	- **Assignment 2 Day2**：通信原语与性能分析  
- [ ] **Day 17 · 6月13日 周六**  
	- **Assignment 2 Day3**：流水线并行策略  
- [ ] **Day 18 · 6月14日 周日**  
	- **Assignment 2 Day4**：混合并行与自动并行  
- [ ] **Day 19 · 6月15日 周一**  
	- **Assignment 2 Day5**：性能调优与对比实验  
- [ ] **Day 20 · 6月16日 周二**  
	- **Assignment 2 Day6**：完善分析与可视化  
- [ ] **Day 21 · 6月17日 周三**  
	- **Assignment 2 Day7**：代码整理与文档  
- [ ] **Day 22 · 6月18日 周四**  
	- **Assignment 2 Day8 (提交)**：最终测试，提交作业 2 ✅  
- [ ] **Day 23 · 6月19日 周五**  
	- **L9**: Scaling laws [Tatsu]  
	- 📄 `lecture_09.pdf`  
- [ ] **Day 24 · 6月20日 周六**  
	- **L10**: Inference [Percy]  
	- 📂 `lecture_10.py`  
- [ ] **Day 25 · 6月21日 周日**  
	- **L11**: Scaling laws (深入) [Tatsu]  
	- 📄 `lecture_11.pdf`  
- [ ] **Day 26 · 6月22日 周一**  
	- **L12**: Evaluation [Percy]  
	- 📂 `lecture_12.py`  
- [ ] **Day 27 · 6月23日 周二**  
	- **Assignment 3 Day1**：Scaling law 拟合实验  
- [ ] **Day 28 · 6月24日 周三**  
	- **Assignment 3 Day2**：推理优化与基准测试  
- [ ] **Day 29 · 6月25日 周四**  
	- **Assignment 3 Day3**：评估指标与任务实现  
- [ ] **Day 30 · 6月26日 周五**  
	- **Assignment 3 Day4**：完成核心代码  
- [ ] **Day 31 · 6月27日 周六**  
	- **Assignment 3 Day5**：消融实验与误差分析  
- [ ] **Day 32 · 6月28日 周日**  
	- **Assignment 3 Day6**：图表与报告  
- [ ] **Day 33 · 6月29日 周一**  
	- **Assignment 3 Day7 (提交)**：最终检查，提交作业 3 ✅
### 🔹 阶段三：数据与训练 (Day 34–46)
- [ ] **Day 34 · 6月30日 周二**  
	- **L13**: Data (sources, datasets) [Percy]  
	- 📂 `lecture_13.py`  
- [ ] **Day 35 · 7月1日 周三**  
	- **L14**: Data (filtering, deduplication, mixing, synthetic data) [Percy]  
	- 📂 `lecture_14.py`  
- [ ] **Day 36 · 7月2日 周四**  
	- **Assignment 4 Day1**：数据采集与预处理流水线  
- [ ] **Day 37 · 7月3日 周五**  
	- **Assignment 4 Day2**：质量过滤与去重算法  
- [ ] **Day 38 · 7月4日 周六**  
	- **Assignment 4 Day3**：数据混合与加权采样  
- [ ] **Day 39 · 7月5日 周日**  
	- **Assignment 4 Day4**：合成数据生成与验证  
- [ ] **Day 40 · 7月6日 周一**  
	- **Assignment 4 Day5**：训练一个小模型验证数据质量  
- [ ] **Day 41 · 7月7日 周二**  
	- **Assignment 4 Day6**：分析数据配比的影响  
- [ ] **Day 42 · 7月8日 周三**  
	- **Assignment 4 Day7**：迭代优化数据策略  
- [ ] **Day 43 · 7月9日 周四**  
	- **Assignment 4 Day8**：撰写数据报告  
- [ ] **Day 44 · 7月10日 周五**  
	- **Assignment 4 Day9 (提交)**：检查所有实验，提交作业 4 ✅  
- [ ] **Day 45 · 7月11日 周六**  
	- **L15**: Mid/post-training (SFT/RLHF) [Tatsu]  
	- 📄 `lecture_15.pdf`  
- [ ] **Day 46 · 7月12日 周日**  
	- **L16**: Post-training - RLVR [Tatsu]  
	- 📄 `lecture_16.pdf`
### 🔹 阶段四：对齐与前沿 (Day 47–58)
- [ ] **Day 47 · 7月13日 周一**  
	- **L17**: Alignment - RL (systems) [Percy]  
- [ ] **Day 48 · 7月14日 周二**  
	- **L18**: Guest lecture: Daniel Selsam  
- [ ] **Day 49 · 7月15日 周三**  
	- **L19**: Guest lecture: Dan Fu  
- [ ] **Day 50 · 7月16日 周四**  
	- **Assignment 5 Day1**：RLHF 系统搭建与奖励模型  
- [ ] **Day 51 · 7月17日 周五**  
	- **Assignment 5 Day2**：PPO/RLVR 算法实现  
- [ ] **Day 52 · 7月18日 周六**  
	- **Assignment 5 Day3**：对齐训练循环与稳定性  
- [ ] **Day 53 · 7月19日 周日**  
	- **Assignment 5 Day4**：评估对齐效果（有用性/安全性）  
- [ ] **Day 54 · 7月20日 周一**  
	- **Assignment 5 Day5**：对比不同 RL 策略  
- [ ] **Day 55 · 7月21日 周二**  
	- **Assignment 5 Day6**：总结系统设计选择  
- [ ] **Day 56 · 7月22日 周三**  
	- **Assignment 5 Day7**：完善实验报告  
- [ ] **Day 57 · 7月23日 周四**  
	- **Assignment 5 Day8**：代码清理与文档  
- [ ] **Day 58 · 7月24日 周五**  
	- **Assignment 5 Day9 (提交)**：最终复核，提交作业 5 ✅
### 🎓 收官：Day 59
- [ ] **Day 59 · 7月25日 周六**  
	- **综合回顾与课程总结**  
	- 串联全部讲座知识图谱  
	- 复盘5个作业的关键收获  
	- 输出一份自己的 CS336 学习笔记/博客大纲  
	- 🎉 恭喜完成 CS336 全部内容！
---
## 🔗 快捷链接
- [[CS336 讲座笔记 MOC]]
- [[Assignment 1 排错记录]]
- [[Assignment 2 并行实验]]
- [[Scaling Laws 文献阅读]]
- [[RLHF 对齐笔记]]
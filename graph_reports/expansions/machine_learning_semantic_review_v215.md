# V2.15《机器学习》关系候选人工语义复核清单

**注意：以下 54 条只通过原文行、教材章节、唯一 Chroma Chunk ID 的机械证据绑定检查。它们尚未独立语义复核，也未被纳入线上 GraphRAG。**

复核重点：关系方向与谓词是否准确、是否存在条件或限定语、是否把「可看作」误升格为无条件断言、同义概念是否真实等价。

## 第4章

| 复核 | 三元组 | 原文行 | Chunk 前缀 | 原文证据摘录 |
|---|---|---:|---|---|
| ☐ | ID3决策树学习算法 —USES→ 信息增益 | 2408 | `b3c6419e71b2` | 著名的ID3决策树学习算法 [Quinlan, 1986] 就是以信息增益为准则来选择划分属性. |
| ☐ | C4.5决策树算法 —USES→ 增益率 | 2490 | `07f86f657fe1` | 著名的 C4.5 决策树算法 [Quinlan, 1993] 不直接使用信息增益, 而是使用 “增益率” (gain ratio) 来选择最优划分属性. |
| ☐ | CART决策树 —USES→ 基尼指数 | 2510 | `a50dceb8e72c` | CART 决策树 [Breiman et al., 1984] 使用 “基尼指数” (Gini index) 来选择划分属性. |
| ☐ | 预剪枝 —IS_A→ 决策树剪枝 | 2532 | `70474bf1f1f4` | 决策树剪枝的基本策略有“预剪枝”(prepruning)和“后剪枝”(post-pruning)[Quinlan, 1993]. |
| ☐ | 后剪枝 —IS_A→ 决策树剪枝 | 2532 | `70474bf1f1f4` | 决策树剪枝的基本策略有“预剪枝”(prepruning)和“后剪枝”(post-pruning)[Quinlan, 1993]. |

## 第7章

| 复核 | 三元组 | 原文行 | Chunk 前缀 | 原文证据摘录 |
|---|---|---:|---|---|
| ☐ | 朴素贝叶斯分类器 —USES→ 属性条件独立性假设 | 4878 | `6d8e18e34350` | 朴素贝叶斯分类器(naïve Bayes classifier)采用了“属性条件独立性假设”(attribute conditional independence assumption): 对已知类别, 假设所有属性… |
| ☐ | 贝叶斯网 —USES→ 有向无环图 | 5159 | `25b32b334f4b` | 贝叶斯网(Bayesian network)亦称“信念网”(belief network), 它借助有向无环图 (Directed Acyclic Graph, 简称 DAG)来刻画属性之间的依赖关系 |
| ☐ | 贝叶斯网 —ALSO_CALLED→ 信念网 | 5159 | `25b32b334f4b` | 贝叶斯网(Bayesian network)亦称“信念网”(belief network) |
| ☐ | 贝叶斯网 —HAS_PART→ 网络参数 | 5165 | `25b32b334f4b` | 一个贝叶斯网 B 由结构 G 和参数 $\Theta$ 两部分构成 |
| ☐ | 贝叶斯网 —HAS_PART→ 网络结构 | 5165 | `25b32b334f4b` | 一个贝叶斯网 B 由结构 G 和参数 $\Theta$ 两部分构成 |
| ☐ | 贝叶斯网近似推断 —USES→ 吉布斯采样 | 5320 | `0f93176f8c80` | 在现实应用中，贝叶斯网的近似推断常使用吉布斯采样(Gibbs sampling)来完成，这是一种随机采样方法 |

## 第8章

| 复核 | 三元组 | 原文行 | Chunk 前缀 | 原文证据摘录 |
|---|---|---:|---|---|
| ☐ | 随机森林 —IS_A→ 并行化集成学习方法 | 5715 | `13fabbf281e5` | 根据个体学习器的生成方式, 目前的集成学习方法大致可分为两大类, 即个体学习器间存在强依赖关系、必须串行生成的序列化方法, 以及个体学习器间不存在强依赖关系、可同时生成的并行化方法; 前者的代表是 Boosting,… |
| ☐ | Bagging —IS_A→ 并行化集成学习方法 | 5715 | `13fabbf281e5` | 根据个体学习器的生成方式, 目前的集成学习方法大致可分为两大类, 即个体学习器间存在强依赖关系、必须串行生成的序列化方法, 以及个体学习器间不存在强依赖关系、可同时生成的并行化方法; 前者的代表是 Boosting,… |
| ☐ | Boosting —IS_A→ 序列化集成学习方法 | 5715 | `13fabbf281e5` | 根据个体学习器的生成方式, 目前的集成学习方法大致可分为两大类, 即个体学习器间存在强依赖关系、必须串行生成的序列化方法, 以及个体学习器间不存在强依赖关系、可同时生成的并行化方法; 前者的代表是 Boosting,… |
| ☐ | Boosting —USES→ 弱学习器 | 5719 | `d0548373c1a0` | Boosting 是一族可将弱学习器提升为强学习器的算法. |
| ☐ | Bagging —BASED_ON→ 自助采样法 | 5896 | `8080082be9e5` | Bagging [Breiman, 1996a] 是并行式集成学习方法最著名的代表. 从名字即可看出, 它直接基于我们在 2.2.3 节介绍过的自助采样法 (bootstrap sampling). |
| ☐ | Bagging —IS_A→ 并行式集成学习方法 | 5896 | `8080082be9e5` | Bagging [Breiman, 1996a] 是并行式集成学习方法最著名的代表. |
| ☐ | 随机森林 —VARIANT_OF→ Bagging | 5939 | `df6401ac494d` | 随机森林(Random Forest, 简称 RF) [Breiman, 2001a] 是 Bagging 的一个扩展变体. |
| ☐ | 随机森林 —USES→ 决策树 | 5939 | `df6401ac494d` | RF 在以决策树为基学习器构建 Bagging 集成的基础上, 进一步在决策树的训练过程中引入了随机属性选择. |
| ☐ | 随机森林 —USES→ 随机属性选择 | 5939 | `df6401ac494d` | RF 在以决策树为基学习器构建 Bagging 集成的基础上, 进一步在决策树的训练过程中引入了随机属性选择. |
| ☐ | 随机森林 —ALSO_CALLED→ RF | 5939 | `df6401ac494d` | 随机森林(Random Forest, 简称 RF) |

## 第9章

| 复核 | 三元组 | 原文行 | Chunk 前缀 | 原文证据摘录 |
|---|---|---:|---|---|
| ☐ | DBSCAN —IS_A→ 密度聚类算法 | 7235 | `3f6b56ab2991` | DBSCAN 是一种著名的密度聚类算法 |
| ☐ | DBSCAN —USES→ 邻域参数 | 7235 | `3f6b56ab2991` | DBSCAN 是一种著名的密度聚类算法, 它基于一组 “邻域” (neighborhood) 参数 $(\epsilon, MinPts)$ 来刻画样本分布的紧密程度. |
| ☐ | AGNES —IS_A→ 层次聚类算法 | 7361 | `c9bdc3b78e1d` | AGNES 是一种采用自底向上聚合策略的层次聚类算法. |

## 第10章

| 复核 | 三元组 | 原文行 | Chunk 前缀 | 原文证据摘录 |
|---|---|---:|---|---|
| ☐ | k近邻学习 —IS_A→ 监督学习方法 | 7813 | `4af7f6d66a14` | k 近邻(k-Nearest Neighbor, 简称 kNN)学习是一种常用的监督学习方法 |
| ☐ | k近邻学习 —ALSO_CALLED→ kNN | 7813 | `4af7f6d66a14` | k 近邻(k-Nearest Neighbor, 简称 kNN) |
| ☐ | 主成分分析 —IS_A→ 降维方法 | 7981 | `6f3996614263` | 主成分分析(Principal Component Analysis, 简称 PCA)是最常用的一种降维方法. |
| ☐ | 主成分分析 —ALSO_CALLED→ PCA | 7981 | `6f3996614263` | 主成分分析(Principal Component Analysis, 简称 PCA) |

## 第14章

| 复核 | 三元组 | 原文行 | Chunk 前缀 | 原文证据摘录 |
|---|---|---:|---|---|
| ☐ | 隐马尔可夫模型 —ALSO_CALLED→ HMM | 10905 | `f921a181a791` | 隐马尔可夫模型(Hidden Markov Model, 简称 HMM) |
| ☐ | 隐马尔可夫模型 —IS_A→ 动态贝叶斯网 | 10905 | `f921a181a791` | 隐马尔可夫模型(Hidden Markov Model, 简称 HMM)是结构最简单的动态贝叶斯网(dynamic Bayesian network) |
| ☐ | 隐马尔可夫模型 —HAS_PART→ 观测变量 | 10907 | `f921a181a791` | 隐马尔可夫模型中的变量可分为两组. 第一组是状态变量 $\{y_{1}, y_{2}, \ldots, y_{n}\}$ , 其中 $y_{i} \in Y$ 表示第 i 时刻的系统状态. 通常假定状态变量是隐藏的、… |
| ☐ | 隐马尔可夫模型 —HAS_PART→ 状态变量 | 10907 | `f921a181a791` | 隐马尔可夫模型中的变量可分为两组. 第一组是状态变量 |
| ☐ | 马尔可夫随机场 —ALSO_CALLED→ 马尔可夫网 | 10974 | `333e0b3a90bd` | 马尔可夫随机场(Markov Random Field, 简称 MRF)是典型的马尔可夫网 |
| ☐ | 马尔可夫随机场 —IS_A→ 无向图模型 | 10974 | `333e0b3a90bd` | 马尔可夫随机场(Markov Random Field, 简称 MRF)是典型的马尔可夫网, 这是一种著名的无向图模型. |
| ☐ | 马尔可夫随机场 —USES→ 势函数 | 10974 | `333e0b3a90bd` | 马尔可夫随机场有一组势函数(potential functions) |
| ☐ | 马尔可夫随机场 —ALSO_CALLED→ MRF | 10974 | `333e0b3a90bd` | 马尔可夫随机场(Markov Random Field, 简称 MRF) |
| ☐ | 势函数 —ALSO_CALLED→ 因子 | 10974 | `333e0b3a90bd` | 势函数(potential functions), 亦称 “因子” (factor) |
| ☐ | 条件随机场 —CONDITIONAL_FORM_OF→ 马尔可夫随机场（可看作） | 11093 | `5329a7bd50a8` | 条件随机场可看作给定观测值的马尔可夫随机场 |
| ☐ | 马尔可夫随机场 —IS_A→ 生成式模型 | 11095 | `5329a7bd50a8` | 前面介绍的隐马尔可夫模型和马尔可夫随机场都是生成式模型 |
| ☐ | 条件随机场 —ALSO_CALLED→ CRF | 11095 | `5329a7bd50a8` | 条件随机场(Conditional Random Field, 简称 CRF) |
| ☐ | 条件随机场 —IS_A→ 判别式模型 | 11095 | `5329a7bd50a8` | 而条件随机场则是判别式模型. |
| ☐ | 隐马尔可夫模型 —IS_A→ 生成式模型 | 11095 | `5329a7bd50a8` | 前面介绍的隐马尔可夫模型和马尔可夫随机场都是生成式模型 |
| ☐ | 条件随机场 —IS_A→ 判别式无向图模型 | 11095 | `5329a7bd50a8` | 条件随机场(Conditional Random Field, 简称 CRF) 是一种判别式无向图模型. |
| ☐ | 变分推断 —IS_A→ 近似推断方法 | 11267 | `77a798102747` | 第二类是使用确定性近似完成近似推断, 典型代表为变分推断(variational inference). |
| ☐ | 采样 —IS_A→ 近似推断方法 | 11267 | `77a798102747` | 近似推断方法大致可分为两大类: 第一类是采样(sampling), 通过使用随机化方法完成近似 |
| ☐ | MCMC —IS_A→ 采样技术 | 11291 | `39d0b37daccb` | 概率图模型中最常用的采样技术是马尔可夫链蒙特卡罗(Markov Chain Monte Carlo, 简称 MCMC)方法. |
| ☐ | MCMC —ALSO_CALLED→ 马尔可夫链蒙特卡罗 | 11291 | `39d0b37daccb` | 马尔可夫链蒙特卡罗(Markov Chain Monte Carlo, 简称 MCMC) |
| ☐ | MCMC —BASED_ON→ 马尔可夫链 | 11317 | `2212c7273bbc` | MCMC 方法先设法构造一条马尔可夫链 |
| ☐ | Metropolis-Hastings算法 —BASED_ON→ 拒绝采样 | 11321 | `2212c7273bbc` | Metropolis-Hastings (简称 MH) 算法是 MCMC 的重要代表. 它基于 “拒绝采样” (reject sampling) |
| ☐ | Metropolis-Hastings算法 —ALSO_CALLED→ MH算法 | 11321 | `2212c7273bbc` | Metropolis-Hastings (简称 MH) 算法 |
| ☐ | Metropolis-Hastings算法 —IS_A→ MCMC | 11321 | `2212c7273bbc` | Metropolis-Hastings (简称 MH) 算法是 MCMC 的重要代表. |
| ☐ | 吉布斯采样 —SPECIAL_CASE_OF→ MH算法（有时被视为） | 11353 | `5022d01071ae` | 吉布斯采样(Gibbs sampling)有时被视为 MH 算法的特例 |
| ☐ | 吉布斯采样 —USES→ 马尔可夫链 | 11353 | `5022d01071ae` | 它也使用马尔可夫链获取样本 |
| ☐ | 变分推断 —USES→ 简单分布 | 11363 | `b2cce1277ae6` | 变分推断通过使用已知简单分布来逼近需推断的复杂分布 |

## 明确拒绝的候选

- 第 4131 行 `高斯核 —ALSO_CALLED→ RBF核`：原文断言存在，但严格相同来源/章节/证据匹配到两个重叠 Chunk，因此未晋升。

## 审查建议

- 人工复核不应直接修改已冻结的 V2.15 JSON；如需更正，记录差异，另行生成下一版带 SHA 门禁的关系包。
- 多教材同名标签只是索引候选，不是自动产生的新三元组。
- 查看完整断言和绑定证据请使用 `machine_learning_claim_links_v215.json`。

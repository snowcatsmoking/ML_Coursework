# 音乐识别实验方法论

## 核心逻辑线：问题定义 → 数据分析 → 特征设计（基于数据） → 模型训练 → 结果分析

---

## Phase 1: 问题定义与评估指标

### 1.1 问题陈述
**任务定义**
- 输入：Hum/Whistle音频片段
- 输出：8首歌曲之一的分类结果（8分类问题）
- 挑战：个体差异、音频质量、Hum vs Whistle的特性差异

### 1.2 评估指标设计

**为什么选择这些指标？**

这是一个**8分类问题**，数据集**完全平衡**（每类100样本），因此需要综合考虑多个指标：

#### **主指标：Accuracy（准确率）**
```
定义：正确预测的样本数 / 总样本数
为什么选择：数据平衡时，accuracy直观反映整体性能
随机baseline：12.5% (1/8)
合格baseline：> 40%
```

#### **辅助指标1：Per-class Precision & Recall**
```
为什么需要：
- Accuracy只能看整体，看不出各类的表现
- 某些歌可能特别难识别，导致该类recall低
- 某些歌可能经常被误判为其他歌，导致precision低

计算方式：
对每首歌分别计算：
- Precision = 预测为该歌且正确 / 所有预测为该歌
- Recall = 预测为该歌且正确 / 该歌的真实样本数
```

#### **辅助指标2：Macro F1-score**
```
定义：各类F1-score的算术平均
为什么选择：
- F1是precision和recall的调和平均，平衡两者
- Macro平均对每个类一视同仁，不受样本数影响
- 在平衡数据集上，Macro F1 ≈ Weighted F1

F1 = 2 * (Precision * Recall) / (Precision + Recall)
```

#### **辅助指标3：Confusion Matrix（混淆矩阵）**
```
为什么重要：
- 揭示哪些歌容易被混淆
- 理解模型的错误模式
- 指导后续改进方向

不是一个数字，而是一个8x8矩阵
```

**指标优先级**：
1. **Test Accuracy**（最终性能）
2. **Per-class F1**（平衡性）
3. **Confusion Matrix**（错误分析）

---

## Phase 2: 数据探索与分析（关键！）

### 2.1 基础统计
```python
- 总样本数：800
- 每首歌样本数：100 (完全平衡 ✓)
- Hum vs Whistle：487 vs 313
- 参与者数量：120人
```

### 2.2 音频特性可视化分析

**目的：通过可视化理解数据，为特征设计提供依据**

#### **2.2.1 样本分布可视化**
- 每首歌的Hum/Whistle堆叠柱状图
- 音频长度分布直方图

#### **2.2.2 波形与频谱分析（核心）**

**随机抽样对比**：
1. **同一首歌，不同参与者的波形对比**
   - 观察：个体差异有多大？
   - 问题：如果差异很大，需要特征能够鲁棒地捕捉共性

2. **同一首歌，Hum vs Whistle的频谱图对比**
   - 观察：Hum和Whistle在频谱上的差异
   - 问题：需要什么特征来区分它们？

3. **不同歌曲的波形/频谱对比**
   - 观察：哪些歌看起来就很相似？
   - 观察：8首歌在波形/频谱上有什么明显差异？

**从可视化中回答的关键问题**：
- 不同歌曲的**音高（频率）**是否有明显差异？
  → 如果有，pitch特征可能有用
- 不同歌曲的**节奏（时域波形）**是否不同？
  → 如果有，时域特征可能有用
- Hum和Whistle的**频谱形状**是否不同？
  → 如果有，频谱特征可能有用

### 2.3 初步音频特征探索

**在设计最终特征前，先做探索性特征分析**

#### **Step 1：提取候选特征池**
```
基于音频信号处理的常见特征：
A. 时域特征：
   - power（能量）
   - zero_crossing_rate（过零率）

B. 频域特征：
   - pitch_mean, pitch_std, pitch_range（音高统计）
   - voiced_fraction（有声比例）
   - spectral_centroid（频谱质心）
   - spectral_rolloff（谱滚降）

C. 音色特征：
   - mfcc_1-5（前5个MFCC系数）
```

#### **Step 2：可视化特征分布**
```
对每个候选特征，绘制：
1. 箱线图（按8首歌分组）
   → 观察：该特征在8首歌上的区分度如何？

2. 小提琴图（按Hum/Whistle分组）
   → 观察：该特征能否区分Hum和Whistle？
```

**关键分析**：
- 如果某特征的8个箱子**高度重叠** → 区分度差，可能不有用
- 如果某特征的8个箱子**明显分离** → 区分度好，保留
- 如果某特征的Hum/Whistle**明显不同** → 有助于处理类型差异

#### **Step 3：特征相关性分析**
```
绘制候选特征的相关性热图
- 识别高度相关的特征（r > 0.9）
- 如果两个特征高度相关，只保留一个（避免冗余）
```

### 2.4 数据分割策略

**核心问题：如何避免数据泄漏？**

**分割原则**：按参与者分组
```
错误做法：随机划分 → 同一人的样本可能同时在train和test
正确做法：按participant分组 → 训练集和测试集的人完全不重叠

理由：
- 同一人的hum/whistle有独特的"声纹"
- 随机划分会让模型学会"识别人"而非"识别歌"
- 导致测试准确率虚高
```

**分割方案**：
```python
# 120个参与者
train_participants: 96人 (80%)
test_participants: 24人 (20%)

# 从训练参与者再分出验证集
train: 77人 (64% 总样本)
validation: 19人 (16% 总样本)
test: 24人 (20% 总样本)
```

**验证分割质量**：
- 检查train/val/test中每首歌的样本数量是否均衡
- 检查Hum/Whistle比例是否相似

---

## Phase 3: 特征工程（基于Phase 2的分析）

### 3.1 最终特征选择

**原则：只选择在Phase 2中被证明有区分度的特征**

#### **基于箱线图分析的结果**
```
假设Phase 2发现：
- pitch_mean在8首歌上有明显差异 → 保留
- pitch_std在8首歌上有明显差异 → 保留
- pitch_range如果8首歌重叠严重 → 删除
- mfcc_1-3有区分度 → 保留
- mfcc_4-5区分度弱 → 删除
...

最终选择：10-15个有效特征
```

**每个保留特征的理由（事后补充）**：
```
例如：
- pitch_mean：箱线图显示8首歌的中位数分离明显
  → 说明不同歌的音域中心确实不同

- spectral_centroid：箱线图显示Hum和Whistle有差异
  → 说明能帮助区分演唱类型
```

### 3.2 特征重要性验证

**用Random Forest快速验证特征选择是否合理**
```python
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
importances = rf.feature_importances_

# 检查：
# 1. 我们认为重要的特征，RF也认为重要吗？
# 2. 如果某特征importance < 0.01，考虑删除
```

### 3.3 特征空间可视化

**PCA降维到2D**
```python
from sklearn.decomposition import PCA
pca = PCA(n_components=2)
X_2d = pca.fit_transform(X_train)
# 绘制8首歌的散点图，不同颜色
```

**分析**：
- 8首歌是否形成明显的聚类？
- 哪些歌在特征空间中靠得很近？（可能容易混淆）
- 整体是线性可分还是非线性可分？（指导模型选择）

---

## Phase 4: 模型训练与对比

### 4.1 模型选择与理由

**选择4个经典模型（回归基础）**

1. **SVM (RBF kernel)**
   - 理由：小规模数据的经典选择，处理非线性边界
   - 默认参数：C=1.0, gamma='scale'

2. **SVM (Linear kernel)**
   - 理由：判断问题的线性可分性
   - 对比：如果Linear接近RBF，说明问题相对简单

3. **Random Forest**
   - 理由：鲁棒、可解释（特征重要性）
   - 默认参数：n_estimators=100

4. **k-NN**
   - 理由：最简单的baseline，基于相似度
   - 默认参数：n_neighbors=5

**不选择复杂模型的原因**：
- Gradient Boosting：过于复杂，容易过拟合小数据集
- MLP：样本量不足，黑盒性强
- 回归基础机器学习，不追求SOTA

### 4.2 训练流程

**Step 1：特征标准化**
```python
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)
```

**Step 2：在训练集上训练4个模型**
```python
for model in [SVM_RBF, SVM_Linear, RF, KNN]:
    model.fit(X_train_scaled, y_train)
```

**Step 3：在验证集上评估**
```python
y_val_pred = model.predict(X_val_scaled)
val_acc = accuracy_score(y_val, y_val_pred)
val_f1 = f1_score(y_val, y_val_pred, average='macro')
```

**Step 4：选择最佳模型在测试集上最终评估**

### 4.3 交叉验证（可选）

**目的**：评估模型稳定性
```python
# 5-fold CV on training set
# 注意：fold按participant分组
from sklearn.model_selection import GroupKFold
cv = GroupKFold(n_splits=5)
cv_scores = cross_val_score(model, X_train, y_train, cv=cv, groups=participants_train)
print(f"CV Accuracy: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")
```

**价值**：
- 如果std大，说明模型不稳定
- 帮助选择鲁棒的模型

---

## Phase 5: 结果分析（核心）

### 5.1 模型性能对比

**对比表**
```
| Model       | Train Acc | Val Acc | Val F1 | Test Acc | Test F1 |
|-------------|-----------|---------|--------|----------|---------|
| SVM-Linear  |           |         |        |          |         |
| SVM-RBF     |           |         |        |          |         |
| RF          |           |         |        |          |         |
| KNN         |           |         |        |          |         |
```

**可视化**：
- 柱状图：4个模型的Test Acc对比
- 折线图：Train vs Val vs Test性能（检测过拟合）

**分析**：
- 哪个模型最好？为什么？
- 是否有过拟合现象？（Train >> Test）
- Linear SVM vs RBF SVM对比说明什么？
  - 如果Linear接近RBF：问题相对线性可分
  - 如果RBF明显更好：问题有非线性特征

### 5.2 混淆矩阵分析

**可视化最佳模型的8x8混淆矩阵**

**深入分析**：
1. **对角线**：哪首歌最容易被正确识别？
2. **非对角线**：哪对歌最容易混淆？
3. **原因分析**：
   - 回到Phase 2的箱线图
   - 检查混淆歌曲在关键特征上的值
   - 例如："Song A和Song B混淆20次，检查发现它们的pitch_mean分别为220Hz和225Hz，接近是混淆原因"

### 5.3 Per-class性能分析

**可视化**：每首歌的Precision、Recall、F1条形图

**分析**：
- 哪首歌F1最高/最低？
- 低F1的歌是precision低还是recall低？
  - Precision低：经常被误判为这首歌
  - Recall低：这首歌经常识别不出
- 结合Phase 2的特征分析解释原因

### 5.4 Hum vs Whistle对比

**问题**：模型在两种类型上表现是否一致？

**分析**：
```python
hum_acc = accuracy_score(y_test[hum_mask], y_pred[hum_mask])
whistle_acc = accuracy_score(y_test[whistle_mask], y_pred[whistle_mask])
```

**可能结论**：
- Whistle更准确：音高更清晰（回到Phase 2的频谱图验证）
- Hum更准确：训练样本更多（487 vs 313）

### 5.5 Bad Case分析

**选择10-20个错误样本**

**分析步骤**：
1. 列出真实标签和预测标签
2. 查看这些样本的特征值
3. 寻找共性：
   - 某个参与者的样本特别容易错？（个体质量问题）
   - 某些特征异常（如pitch_std特别小，说明旋律单调）

**价值**：理解模型局限性

---

## Phase 6: 特征重要性解释

**使用Random Forest的feature importances**

**可视化**：特征重要性条形图（降序排列）

**分析与验证**：
1. 最重要的特征是什么？
2. 这个结果与Phase 2的箱线图分析一致吗？
   - 如果一致：说明我们的特征选择逻辑正确
   - 如果不一致：需要反思
3. 重要性低（<0.01）的特征是否应该删除？

**回到音乐直觉**：
- 例如："pitch_mean是最重要特征，验证了旋律是歌曲识别的核心"

---

## Phase 7: 可视化总结

### 7.1 决策边界可视化（可选）

**使用PCA降到2D**
```python
pca = PCA(n_components=2)
X_test_2d = pca.fit_transform(X_test_scaled)
# 绘制8首歌的散点图 + 最佳模型的决策边界
```

**价值**：
- 直观看到歌曲在特征空间的分布
- 理解为什么某些歌容易混淆（距离近）

### 7.2 最终总结图表

1. **模型对比图**：4个模型的test acc柱状图
2. **混淆矩阵热图**：最佳模型
3. **特征重要性图**：所有特征
4. **Per-class F1图**：8首歌的性能对比

---

## 实验流程总结（强调数据驱动）

```
Phase 1: 问题定义 + 评估指标设计
         ↓
    明确任务、选择合适的评估指标

Phase 2: 数据探索（核心！）
         ↓
    波形/频谱可视化 → 提取候选特征 → 箱线图分析区分度
    → 基于数据决定保留哪些特征

Phase 3: 特征工程（基于Phase 2）
         ↓
    只选择被证明有区分度的特征（10-15维）
    特征重要性验证 + PCA可视化

Phase 4: 模型训练
         ↓
    4个基础模型：SVM-Linear, SVM-RBF, RF, KNN
    训练 → 验证 → 测试

Phase 5: 结果分析（核心）
         ↓
    性能对比 + 混淆矩阵 + Per-class + Hum/Whistle + Bad case
    所有分析都要回到Phase 2的数据观察进行解释

Phase 6: 特征重要性解释
         ↓
    验证特征选择是否合理

Phase 7: 可视化总结
         ↓
    汇总所有结果
```

---

## 关键原则

1. **数据驱动**：特征不是YY的，而是基于数据分析得出
2. **回归基础**：使用经典ML方法，不追求复杂模型
3. **严谨性**：按参与者分组，避免数据泄漏
4. **闭环验证**：Phase 2发现 → Phase 3设计 → Phase 6验证
5. **深度分析**：不只看数字，要理解为什么

---

## 预期结果

**随机Baseline**：12.5% (1/8)
**合格Baseline**：40-50%
**良好性能**：60-70%
**优秀性能**：70%+

**成功标准**：
- 不追求最高分，追求严谨的方法
- 每个决策都有数据支撑
- 能够解释模型的成功和失败
- 形成完整的分析闭环

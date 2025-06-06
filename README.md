# rl-learning-notes
intro：强化学习小白一周学习路径，涵盖强化学习基础、DPO、GRPO、Reward Model的训练流程，以及它们与Qwen2-Instruct/Qwen2-Audio的结合方式
| 日期 | 学习主题 | 具体内容 | 产出 | 
| ---- | ---- | ---- | ---- |
| DAY1-2 6月6日 | 基础概念与Reward Model​ | 理论部分：<br>​​强化学习基础​​：理解马尔可夫决策过程（MDP）、策略梯度、PPO算法。<br>​​Reward Model训练流程​​：<br>数据标注：偏好排序（如Bradley-Terry模型）；损失函数；<br>实践：<br>用Hugging Face的transformers库加载预训练模型（如Qwen2），添加线性层作为Reward头部。 <br>代码实践：<br>任务​​：训练一个简单的Reward Model，对文本对（如问答）进行偏好打分。|111| 
| DAY3-4 6月8日 | DPO（Direct Preference Optimization） | 内容6 | 内容3 | 
| DAY5-6 6月10日 | GRPO（Group Relative Policy Optimization）​ | 内容6 | 内容3 | 
| DAY7-8 6月12日 | 结合Qwen2-Instruct/Qwen2-Audio​| 内容6 | 内容3 |  

第一次实践示例
```python
from transformers import AutoModelForSequenceClassification
model = AutoModel.from_pretrained("Qwen/Qwen2-7B")
reward_model = AutoModelForSequenceClassification.from_pretrained("Qwen/Qwen2-7B", num_labels=1)
# 输入文本对 (x, y1, y2)，计算偏好损失

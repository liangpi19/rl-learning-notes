# rl-learning-notes
intro：强化学习小白一周学习路径，涵盖强化学习基础、DPO、GRPO、Reward Model的训练流程，以及它们与Qwen2-Instruct/Qwen2-Audio的结合方式
| 日期 | 学习主题 | 具体内容 | 产出 | 
| ---- | ---- | ---- | ---- |
| DAY1-2 6月6日 | 基础概念与Reward Model​ | 理论部分：<br>​​强化学习基础​​：理解马尔可夫决策过程（MDP）、策略梯度、PPO算法。<br>​​Reward Model训练流程​​：<br>数据标注：偏好排序（如Bradley-Terry模型）；损失函数；<br>实践：<br>用Hugging Face的transformers库加载预训练模型（如Qwen2），添加线性层作为Reward头部。 <br>代码实践：<br>任务​​：训练一个简单的Reward Model，对文本对（如问答）进行偏好打分。|   | 
| DAY3-4 6月8日 | DPO（Direct Preference Optimization） |理论部分：<br>DPO原理​​：通过偏好数据直接优化策略，避免强化学习循环<br>与Qwen2结合​​：用Qwen2作为初始策略模型π ref,微调π θ<br>代码实践：<br>​​任务​​：使用DPO微调Qwen2-instruct，生成更符合人类偏好的回答。<br>​​库推荐​​：trl（Transformer Reinforcement Learning）库。|   | 
| DAY5-6 6月10日 | GRPO（Group Relative Policy Optimization）​ |理论部分：<br> GRPO核心思想：通过组内样本的相对优势（如归一化奖励）优化策略。<br>与Qwen2-Audio结合​​：参考小米团队用GRPO微调Qwen2-Audio的案例提升音频推理能力。<br>https://github.com/xiaomi-research/r1-aqa<br>代码实践：<br>任务​​：复现GRPO的组内优势计算，优化文本生成策略。<br>​​扩展​​：尝试在音频任务（如MMAU数据集）中应用GRPO|   | 
| DAY7-8 6月12日 | 结合Qwen2-Instruct/Qwen2-Audio​| 内容6 | 内容3 |  

第一次实践示例
```python
from transformers import AutoModelForSequenceClassification
model = AutoModel.from_pretrained("Qwen/Qwen2-7B")
reward_model = AutoModelForSequenceClassification.from_pretrained("Qwen/Qwen2-7B", num_labels=1)
# 输入文本对 (x, y1, y2)，计算偏好损失```

第二次实践示例
```python
from trl import DPOTrainer
dpo_trainer = DPOTrainer(
    model=Qwen2_model,
    ref_model=Qwen2_ref,
    args=training_args,
    train_dataset=preference_data,
)
dpo_trainer.train()```

第三次实践关键步骤
def compute_advantages(rewards):
    mean_reward = np.mean(rewards)
    std_reward = np.std(rewards)
    return (rewards - mean_reward) / (std_reward + 1e-8)

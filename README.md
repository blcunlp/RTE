# RTE
自然语言推断（Natural Language Inference，NLI），又称文本蕴含识别（Recognizing Textual Entailment，RTE），是一项经典的自然语言理解任务。给定前提句与假设句，该任务要求判断两者之间蕴含、矛盾、中立的推理关系。

这里列举了我们在NLI任务上工作的代码，包括：ESIM（复现），KAIN，MIMN等。

## 部分模型的论文如下：
ESIM（ "Enhanced LSTM for Natural Language Inference" by Chen et al. in 2016）：https://arxiv.org/pdf/1609.06038.pdf

KAIN（"Knowledge Augmented Inference Network for Natural Language Inference" by Jiang et al. in 2018）：https://link.springer.com/chapter/10.1007/978-981-13-3146-6_11
该工作被ccks 2018接收，使用Wordnet作为外部知识，用TransE将关系三元组转化为稠密连续的向量形式，加入模型。

MIMN（"Multi-turn Inference Matching Network for Natural Language Inference"by Chun et al. in 2018）：https://arxiv.org/pdf/1901.02222.pdf
该工作为NLPCC 2018 outstanding paper

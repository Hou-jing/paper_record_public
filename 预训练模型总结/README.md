## 特定领域
Sci, Bio, Fin, K分别代表科学、生物医学、金融和知识。Tok, Sent, Doc分别表示token、句子和文档。Region, Frame分别表示图像和视频的基本单位。Standard language model（SLM）的目标正是如此，训练模型以优化训练语料库中文本的概率P（x）（Radford等人，2019）。在这些情况下，文本通常以自回归的方式进行预测，每次预测序列中的token。这通常是从左到右进行的（详见下文），但也可以按其他顺序进行。
## 训练任务
损坏的文本重建Corrupted Text Reconstruction（CTR） 这些目标通过仅对输入句子的噪声部分计算损失，将处理后的文本恢复到未损坏的状态。
全文重构 Full Text Reconstruction（FTR） 这些目标通过计算整个输入文本的损失来重构文本，无论其是否经过噪声处理（刘易斯等人，2020a）。
从左到右的LM（L2R LM）是自回归式LM的一种
LM : Mask; L2R (Left to Right);前缀语言模型;编码器和解码器(Encoder-Decoder)
## 预训练模型辅助目标：
Next Sentence Prediction (NSP) (Devlin et al., 2019)：一种二元分类损失，用于预测两个片段是否连续出现在较大的文档中，或者是随机不相关的句子。
Sentence Order Prediction (SOP)（Lan 等人，2020 年）：用于预测两个句子是自然顺序还是交换顺序的二元分类损失。
Capital Word Prediction (CWP)（Liu 等人，2020b）：对每个词计算的二进制分类目标，预测每个词是否大写。
Sentence Deshuffling (SDS) (Liu et al., 2020b)：一种用于重组置换段的多类分类任务。
Sentence distance prediction句子距离预测 (SDP) (Liu et al., 2020b)：三类分类任务，预测两个句子之间的位置关系（在同一个文档中相邻，在同一个文档中不相邻，在不同文档中）。 
Masked Column Prediction (MCP) (屏蔽列预测 (MCP)（Yin 等人，2020 年）：给定一个表，恢复屏蔽列的名称和数据类型。 Linguistic-Visual Alignment (LVA) 语言视觉对齐 (LVA)（Lu 等人，2019 年）：预测文本内容是否可以与视觉内容对齐的二元分类。
Image Region prediction (IRP)图像区域预测 (IRP) (Su et al., 2020)：给定一个部分特征被屏蔽（归零）的图像，预测被屏蔽的区域。 
Replaced Token Detection (RTD)替换令牌检测 (RTD) (Xiao et al., 2021)：预测损坏输入中的每个令牌是否被生成样本替换的二进制分类损失。
Discourse Relation Prediction (DRP)话语关系预测 (DRP) (Sun et al., 2020)：预测两个句子之间的语义或修辞关系。
Translation Language Modeling (TLM)翻译语言建模 (TLM)（Lample 和 Conneau，2019 年）：在源句和目标句中随机考虑平行句和掩码词。
Information Retrieval Relevance信息检索相关性 (IRR) (Sun et al., 2020)：预测两个句子的信息检索相关性。 
Token-Passage Prediction (TPP)（Liu 等人，2020b）：识别出现在片段中的段落的关键字。
Universal Knowledge-Text Prediction通用知识文本预测 (UKTP)（Sun 等人，2021 年）：将知识整合到一个预训练的语言模型中。
Machine Translation (机器翻译 (MT) (Chi et al., 2021a)：将句子从源语言翻译成目标语言。
Translation Pair Span Corruption翻译对跨度损坏 (TPSC) (Chi et al., 2021a)：预测翻译对的掩码跨度。
Translation Span Corruption翻译跨度损坏 (TSC) (Chi et al., 2021a)：与 TPSC 不同，TSC 仅掩盖和预测一种语言的跨度
Multilingual Replaced Token Detection多语言替换标记检测 (MRTD) (Chi et al., 2021b)：通过生成对抗网络区分真实输入标记和损坏的多语言句子，其中生成器和鉴别器在语言之间共享。 
Translation Replaced Token Detection翻译替换标记检测 (TRTD) (Chi et al., 2021b)：通过生成对抗网络区分翻译对中的真实标记和掩码标记。 
Knowledge Embedding知识嵌入 (KE)（Wang 等人，2021 年）：将知识图 (KG) 中的实体和关系编码为分布式表示
Image-to-text transfer图像到文本传输 (ITT)（Wang 等人，2021 年）：类似于为输入图像生成相应描述的图像标题。 
Multimodality-to-text transfer (MTT) (Wang et al., 2021)：根据视觉信息和噪声语言信息生成目标文本
## 应用：
NLG——自然语言生成
NLU：自然语言理解 (Natural Language Understanding)

![总结图](https://img-blog.csdnimg.cn/e1e1bfb8e07441ce93ad0175002cb526.png)


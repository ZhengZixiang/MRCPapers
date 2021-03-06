# MRCPapers
Worth-reading paper list and other awesome resources on Machine Reading Comprehension (MRC) and Question Answering (QA). Suggestions about adding papers, repositories and other resource are welcomed!

机器阅读理解领域值得一读的论文列表和其他相关资源集合。欢迎新增论文、代码仓库与其他资源等建议！

## Paper
- **Bidirectional Attention Flow for Machine Comprehension**. *Min Joon Seo, Aniruddha Kembhavi, Ali Farhadi, Hannaneh Hajishirzi*. (ICLR 2017) [[paper]](https://openreview.net/forum?id=HJ0UKP9ge)[[code]](https://allenai.github.io/bi-att-flow/) - ***BiDAF***
- **Read + Verify: Machine Reading Comprehension with Unanswerable Questions**. *Minghao Hu, Furu Wei, Yuxing Peng, Zhen Huang, Nan Yang, Dongsheng Li*. (AAAI-2019) [[paper]](https://arxiv.org/pdf/1808.05759.pdf)[[unofficial code]](https://github.com/woshiyyya/Answer-Verifier-pytorch) - ***Verfier***
- **Cognitive Graph for Multi-Hop Reading Comprehension at Scale**. *Ming Ding, Chang Zhou, Qibin Chen, Hongxia Yang, Jie Tang*. (ACL 2019) [[paper]](https://arxiv.org/abs/1905.05460) - ***CogQA***
- **Dual Co-Matching Network for Multi-choice Reading Comprehension**. *Shuailiang Zhang, Hai Zhao, Yuwei Wu, Zhuosheng Zhang, Xi Zhou, Xiang Zhou*. (CoRR 2019) [[paper]](https://arxiv.org/abs/1901.09381) - ***DCMN***
- **A Multi-Type Multi-Span Network for Reading Comprehension that Requires Discrete Reasoning**. *Minghao Hu, Yuxing Peng, Zhen Huang, Dongsheng Li*. (EMNLP 2019) [[paper]](https://arxiv.org/abs/1908.05514) - ***MTMSN***
- **Tag-based Multi-Span Extraction in Reading Comprehension**. *Avia Efrat, Elad Segal, Mor Shoham*. (CoRR 2019) [[paper]](https://arxiv.org/abs/1909.13375)
- **SG-Net: Syntax-Guided Machine Reading Comprehension**. *Zhuosheng Zhang, Yuwei Wu, Junru Zhou, Sufeng Duan, Hai Zhao, Rui Wang*. (AAAI 2020) [[paper]](https://arxiv.org/abs/1908.05147)[[code]](https://github.com/cooelf/SG-Net) - ***SG-Net***
- **DCMN+: Dual Co-Matching Network for Multi-choice Reading Comprehension**. *Shuailiang Zhang, Hai Zhao, Yuwei Wu, Zhuosheng Zhang, Xi Zhou, Xiang Zhou*. (AAAI 2020) [[paper]](https://arxiv.org/abs/1908.11511.pdf)
- **MMM: Multi-stage Multi-task Learning for Multi-choice Reading Comprehension**. *Di Jin, Shuyang Gao, Jiun-Yu Kao, Tagyoung Chung, Dilek Hakkani-tur*. (AAAI 2020) [[paper]](https://arxiv.org/abs/1910.00458)[[code]](https://github.com/jind11/MMM-MCQA)
- **Retrospective Reader for Machine Reading Comprehension**. *Zhuosheng Zhang, Junjie Yang, Hai Zhao*. (CoRR 2020) [[paper]](https://arxiv.org/abs/2001.09694)[[Chinese blog]](https://mp.weixin.qq.com/s?__biz=MzIwMTc4ODE0Mw==&mid=2247502891&idx=1&sn=8f3d552ee384544d0b9a868e01b91ed9&key=5fa67e91c99877c949e72c80560ca0bb5dc99de132236c4b530784a3c3c2cc93a94dcd482b4968b128c7bc7553888c5df30cc4f734abb1a63a2bd02402645a9b966bd4291e333ef13e861eb06c80822a&ascene=1&uin=Mjg1NTM0NDcyMw%3D%3D&devicetype=Windows+10&version=6208006f&lang=zh_CN&exportkey=A%2BCwv01k%2FtyMn%2Ft38iF3KbY%3D&pass_ticket=nkIz09BYlgtIrHo7XkM4ahTkS8sck64jbLwU0LotdcTnxt2f%2FIuSGmn33Pc7gW1f) - ***Retro-Reader***
- **DeFormer: Decomposing Pre-trained Transformers for Faster Question Answering**. *Qingqing Cao, Harsh Trivedi, Aruna Balasubramanian, Niranjan Balasubramanian*. (ACL 2020) [[paper]](https://arxiv.org/abs/2005.00697)[[code]](https://github.com/StonyBrookNLP/deformer) - ***DeFormer***
- **Recurrent Chunking Mechanisms for Long-Text Machine Reading Comprehension**. *Hongyu Gong, Yelong Shen, Dian Yu, Jianshu Chen, Dong Yu*. (ACL 2020) [[paper]](https://arxiv.org/abs/2005.08056)[[code]](https://github.com/HongyuGong/RCM-Question-Answering)
- **Is Graph Structure Necessary for Multi-hop Question Answering?**. *Nan Shao, Yiming Cui, Ting Liu, Shijin Wang, Guoping Hu*. (EMNLP 2020) [[paper]](https://arxiv.org/abs/2004.03096)
- **DUMA: Reading Comprehension with Transposition Thinking**. *Pengfei Zhu, Hai Zhao, Xiaoguang Li*. (CoRR 2020) [[paper]](https://arxiv.org/abs/2001.09415)
- **No Answer is Better Than Wrong Answer: A Reflection Model for Document Level Machine Reading Comprehension**. *Xuguang Wang, Linjun Shou, Ming Gong, Nan Duan, Daxin Jiang*. (Findings of EMNLP 2020) [[paper]](https://arxiv.org/abs/2009.12056)
- **CogLTX: Applying BERT to Long Texts**. *Ming Ding, Chang Zhou, Hongxia Yang, Jie Tang*. (NeurIPS 2020) [[paper]](https://proceedings.neurips.cc/paper/2020/hash/96671501524948bc3937b4b30d0e57b9-Abstract.html)[[code]](https://github.com/Sleepychord/CogLTX)
- **HopRetriever: Retrieve Hops over Wikipedia to Answer Complex Questions**. *Shaobo Li, Xiaoguang Li, Lifeng Shang, Xin Jiang, Qun Liu, Chengjie Sun, Zhenzhou Ji, Bingquan Liu*. (AAAI 2021) [[paper]](https://arxiv.org/abs/2012.15534)[[Chinese blog]](https://mp.weixin.qq.com/s/vgx4JBJHxigySffIXXymtg)

## Survey & Review & Tutorial
- **Neural Machine Reading Comprehension: Methods and Trends**. *Shanshan Liu, Xin Zhang, Sheng Zhang, Hui Wang, Weiming Zhang* (Applied Sciences 2019) [[paper]](https://arxiv.org/abs/1907.01118)
- **Research on Machine Reading Comprehension and Textual Question Answering**. *Minghao Hu*. (PhD thesis 2019 in Chinese) [[paper]](https://github.com/huminghao16/thesis)
- **Neural Reading Comprehension and Beyond**. *Danqi Chen*. (PhD thesis 2019) [[paper]](https://purl.stanford.edu/gd576xb1833)
- **Open-Domain Question Answering**. *Danqi Chen*. (ACL 2020) [[slides]](https://github.com/danqi/acl2020-openqa-tutorial)
- **English Machine Reading Comprehension Datasets: A Survey**. *Daria Dzendzik, Carl Vogel, Jennifer Foster*. (CoRR 2021) [[paper]](https://arxiv.org/abs/2101.10421)

## Dataset
- **ChID: A Large-scale Chinese IDiom Dataset for Cloze Test**. *Chujie Zheng, Minlie Huang, Aixin Sun*. (ACL 2019) [[paper]](https://arxiv.org/abs/1906.01265)
- **Investigating Prior Knowledge for Challenging Chinese Machine Reading Comprehension**. *Kai Sun, Dian Yu, Dong Yu, Claire Cardie*. (TACL 2020) [[paper]](https://arxiv.org/abs/1904.09679)[[data]](https://github.com/nlpdata/c3) - ***C3***
- **CoQA: A Conversational Question Answering Challenge**. *Siva Reddy, Danqi Chen, Christopher D. Manning*. (NAACL 2019 & TACL 2019) [[paper]](https://arxiv.org/pdf/1808.07042.pdf) [[data]](https://stanfordnlp.github.io/coqa/)
- **DREAM: A Challenge Dataset and Models for Dialogue-Based Reading Comprehension**. *Kai Sun, Dian Yu, Jianshu Chen, Dong Yu, Yejin Choi, Claire Cardie*. (TACL Vol.7 2019) [[paper]](https://arxiv.org/abs/1902.00164)[[data]](https://dataset.org/dream/)
- **DROP: A Reading Comprehension Benchmark Requiring Discrete Reasoning Over Paragraphs**. *Dheeru Dua, Yizhong Wang, Pradeep Dasigi, Gabriel Stanovsky, Sameer Singh, Matt Gardner*. (NAACL-HLT 2019) [[paper]](https://arxiv.org/abs/1903.00161)
- **DuReader: a Chinese Machine Reading Comprehension Dataset from Real-world Applications**. *Wei He, Kai Liu, Jing Liu, Yajuan Lyu, Shiqi Zhao, Xinyan Xiao, Yuan Liu, Yizhong Wang, Hua Wu, Qiaoqiao She, Xuan Liu, Tian Wu, Haifeng Wang*. (ACL 2018 QA workshop) [[paper]](https://www.aclweb.org/anthology/W18-2605/)[[data]](https://github.com/baidu/DuReader)
- **HotpotQA: A Dataset for Diverse, Explainable Multi-hop Question Answering**. *Zhilin Yang, Peng Qi, Saizheng Zhang, Yoshua Bengio, William W. Cohen, Ruslan Salakhutdinov, Christopher D. Manning*. (EMNLP 2018) [[paper]](https://arxiv.org/abs/1809.09600)[[data]](https://hotpotqa.github.io/)
- **A Question-Entailment Approach to Question Answering**. *Asma Ben Abacha, Dina Demner-Fushman*. (BMC Bioinformatics 20, 511 (2019)) [[paper]](https://arxiv.org/abs/1901.08079)[[data]](https://github.com/abachaa/MedQuAD) - ***MedQuAD***
- **MIMICS: A Large-Scale Data Collection for Search Clarification**. *Hamed Zamani, Gord Lueck, Everest Chen, Rodolfo Quispe, Flint Luu, Nick Craswell*. (CIKM 2020) [[paper]](https://arxiv.org/abs/2006.10174)[[data]](https://github.com/microsoft/MIMICS)
- **Natural Questions: a Benchmark for Question Answering Research**. *Tom Kwiatkowski, Jennimaria Palomaki, Olivia Redfield, Michael Collins, Ankur Parikh, Chris Alberti, Danielle Epstein, Illia Polosukhin, Jacob Devlin, Kenton Lee, Kristina Toutanova, Llion Jones, Matthew Kelcey, Ming-Wei Chang, Andrew Dai, Jakob Uszkoreit, Quoc Le, Slav Petrov*. (TACL Vol.7 2019) [[paper]](https://transacl.org/ojs/index.php/tacl/article/view/1455)[[data]](https://ai.google.com/research/NaturalQuestions)
- **ProtoQA: A Question Answering Dataset for Prototypical Common-Sense Reasoning**. *Michael Boratko, Xiang Lorraine Li, Rajarshi Das, Tim O'Gorman, Dan Le, Andrew McCallum*. (EMNLP 2020) [[paper]](https://arxiv.org/abs/2005.00771)[[data]](https://github.com/iesl/protoqa-data)
- **QuAC : Question Answering in Context**. *Eunsol Choi, He He, Mohit Iyyer, Mark Yatskar, Wen-tau Yih, Yejin Choi, Percy Liang, Luke Zettlemoyer*. (EMNLP 2018) [[paper]](https://arxiv.org/abs/1808.07036)[[data]](https://quac.ai/)
- **RACE: Large-scale ReAding Comprehension Dataset From Examinations**. *Guokun Lai, Qizhe Xie, Hanxiao Liu, Yiming Yang, Eduard Hovy*. (EMNLP 2017) [[paper]](https://arxiv.org/abs/1704.04683)[[data]](https://www.cs.cmu.edu/~glai1/data/race/)
- **SQuAD: 100,000+ Questions for Machine Comprehension of Text**. *Pranav Rajpurkar, Jian Zhang, Konstantin Lopyrev, Percy Liang*. (EMNLP 2016) [[paper]](https://www.aclweb.org/anthology/D16-1264/)[[data]](https://github.com/rajpurkar/SQuAD-explorer/tree/master/dataset)
- **Know What You Don't Know: Unanswerable Questions for SQuAD**. *Pranav Rajpurkar, Robin Jia, Percy Liang*. (ACL 2018) [[paper]](https://www.aclweb.org/anthology/P18-2124/)[[data]](https://github.com/rajpurkar/SQuAD-explorer/tree/master/dataset) - ***SQuAD 2.0***
- **SWAG: A Large-Scale Adversarial Dataset for Grounded Commonsense Inference**. *Rowan Zellers, Yonatan Bisk, Roy Schwartz, Yejin Choi*. (EMNLP 2018) [[paper]](https://arxiv.org/abs/1808.05326)[[data]](https://github.com/rowanz/swagaf)
- **TYDI QA: A Benchmark for Information-Seeking Question Answering in Typologically Diverse Languages**. *Jonathan H. Clark, Jennimaria Palomaki, Vitaly Nikolaev, Eunsol Choi, Dan Garrette, Michael Collins, Tom Kwiatkowski*. (TACL 2020) [[paper]](https://storage.cloud.google.com/tydiqa/tydiqa.pdf)[[data]](https://github.com/google-research-datasets/tydiqa)

## Repository & Toolkit
- [AllenAI / allennlp-reading-comprehension](https://github.com/allenai/allennlp-reading-comprehension) - Reading comprehension tookit by the AllenNLP team
- [AMontgomerie / question_generator](https://github.com/AMontgomerie/question_generator) - An NLP system for generating reading comprehension questions
- [ankit-ai / BertQA-Attention-on-Steroids](https://github.com/ankit-ai/BertQA-Attention-on-Steroids)
- [alsang / BiDAF-pytorch](https://github.com/galsang/BiDAF-pytorch)
- [BshoterJ / awesome-kgqa](https://github.com/BshoterJ/awesome-kgqa) - A collection of some materials of knowledge graph question answering
- [cdqa-suite / cdQA](https://github.com/cdqa-suite/cdQA)
- [cdqa-suite / cdQA-annotator](https://github.com/cdqa-suite/cdQA-annotator)
- [deepset-ai / Haystack](https://github.com/deepset-ai/haystack)
- [Facebook Research / ParlAI](https://github.com/facebookresearch/ParlAI) - A framework for training and evaluating AI models on a variety of openly available dialogue datasets
- [efficientqa/retrieval-based-baselines](https://github.com/efficientqa/retrieval-based-baselines) - Tutorials on training and testing retrieval-based models
- [IndexFziQ / KMRC-Papers](https://github.com/IndexFziQ/KMRC-Papers)
- [krystalan / Multi-hopRC](https://github.com/krystalan/Multi-hopRC) - 多跳阅读理解相关论文
- [lcdevelop / ChatBotCourse](https://github.com/lcdevelop/ChatBotCourse) - 自己动手做聊天机器人教程
- [lixinsu / RCZoo](https://github.com/lixinsu/RCZoo)
- [seriousran / awesome-qa](https://github.com/seriousran/awesome-qa)
- [Sogou / SMRCToolkit](https://github.com/sogou/SMRCToolkit) - Sogou Machine Reading Comprehension (SMRC) toolkit based on TensorFlow
- [songyingxin / BERT-MRC-RACE](https://github.com/songyingxin/BERT-MRC-RACE)
- [tangbinh / question-answering](https://github.com/tangbinh/question-answering)
- [THUNLP / RCPapers](https://github.com/thunlp/RCPapers)
- [wavewangyue / kbqa](https://github.com/wavewangyue/kbqa) - 基于知识库的问答：seq2seq模型实践
- [xanhho / Reading-Comprehension-Question-Answering-Papers](https://github.com/xanhho/Reading-Comprehension-Question-Answering-Papers) - Survey on Machine Reading Comprehension
- [YingZiqiang / PyTorch-MRCToolkit](https://github.com/YingZiqiang/PyTorch-MRCToolkit)
- [yizhen20133868 / NLP-Conferences-Code](https://github.com/yizhen20133868/NLP-Conferences-Code)
- [ymcui / Chinese-RC-Datasets](https://github.com/ymcui/Chinese-RC-Datasets)
- [zcgzcgzcg1 / MRC_book](https://github.com/zcgzcgzcg1/MRC_book) - 《机器阅读理解：算法与实践》代码

## Blog Post
### English
- [Google / Progress and Challenges in Long-Form Open-Domain Question Answering](https://ai.googleblog.com/2021/03/progress-and-challenges-in-long-form.html)
- [Stanford / Beyond Local Pattern Matching: Recent Advances in Machine Reading](https://ai.stanford.edu/blog/beyond-local-pattern-matching/)
### Chinese
- [changreal / 【总结向】MRC 经典模型与技术](https://blog.csdn.net/changreal/article/details/105074629)
- [changreal / 【总结向】从CMRC2019头部排名看中文MRC](https://blog.csdn.net/changreal/article/details/105363937)
- [humdingers / 2020法研杯阅读理解赛道第一名方案](https://github.com/humdingers/2020CAIL_LDLJ)
- [Liberty / 文本太长，Transformer用不了怎么办](https://zhuanlan.zhihu.com/p/259835450)
- [Luke / 机器阅读理解之多答案抽取](https://zhuanlan.zhihu.com/p/101248202)
- [xhsun1997 / 详细介绍有关RACE数据集上经典的机器阅读理解模型](https://blog.csdn.net/m0_45478865/article/details/106869172?spm=1001.2014.3001.5501)
- [多多笔记 / 开放领域问答梳理系列(1)](https://mp.weixin.qq.com/s/iLE0zwhzd3ffri8VH24Z9g)
- [多多笔记 / 开放领域问答梳理系列(2)](https://mp.weixin.qq.com/s/6DpC1uJqrsyl0a62uLl4zw)
- [科学空间 / 学会提问的BERT：端到端地从篇章中构建问答对](https://kexue.fm/archives/7630)
- [科学空间 / “非自回归”也不差：基于MLM的阅读理解问答](https://kexue.fm/archives/7148)
- [科学空间 / 万能的seq2seq：基于seq2seq的阅读理解问答](https://kexue.fm/archives/7115)
- [平安寿险PAI / 机器阅读理解探索与实践](https://zhuanlan.zhihu.com/p/109309164)
- [知乎问答 / 机器阅读理解方向有什么值得follow的大佬，网站等等](https://www.zhihu.com/question/358469127/answer/1028144909)
- [知乎问答 / Bert 如何解决长文本问题？](https://www.zhihu.com/question/327450789)

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
- **Retrospective Reader for Machine Reading Comprehension**. *Zhuosheng Zhang, Junjie Yang, Hai Zhao*. (CoRR 2020) [[paper]](https://arxiv.org/abs/2001.09694)[[Chinese blog]](https://mp.weixin.qq.com/s?__biz=MzIwMTc4ODE0Mw==&mid=2247502891&idx=1&sn=8f3d552ee384544d0b9a868e01b91ed9&key=5fa67e91c99877c949e72c80560ca0bb5dc99de132236c4b530784a3c3c2cc93a94dcd482b4968b128c7bc7553888c5df30cc4f734abb1a63a2bd02402645a9b966bd4291e333ef13e861eb06c80822a&ascene=1&uin=Mjg1NTM0NDcyMw%3D%3D&devicetype=Windows+10&version=6208006f&lang=zh_CN&exportkey=A%2BCwv01k%2FtyMn%2Ft38iF3KbY%3D&pass_ticket=nkIz09BYlgtIrHo7XkM4ahTkS8sck64jbLwU0LotdcTnxt2f%2FIuSGmn33Pc7gW1f) - ***Retro-Reader***
- **DeFormer: Decomposing Pre-trained Transformers for Faster Question Answering**. *Qingqing Cao, Harsh Trivedi, Aruna Balasubramanian, Niranjan Balasubramanian*. (ACL 2020) [[paper]](https://arxiv.org/abs/2005.00697)[[code]](https://github.com/StonyBrookNLP/deformer) - ***DeFormer***
- **Is Graph Structure Necessary for Multi-hop Question Answering?**. *Nan Shao, Yiming Cui, Ting Liu, Shijin Wang, Guoping Hu*. (EMNLP 2020) [[paper]](https://arxiv.org/abs/2004.03096)

## Survey & Review & Tutorial
- **Neural Machine Reading Comprehension: Methods and Trends**. *Shanshan Liu, Xin Zhang, Sheng Zhang, Hui Wang, Weiming Zhang* (Applied Sciences 2019) [[paper]](https://arxiv.org/abs/1907.01118)
- **Research on Machine Reading Comprehension and Textual Question Answering**. *Minghao Hu*. (PhD thesis 2019 in Chinese) [[paper]](https://github.com/huminghao16/thesis)
- **Neural Reading Comprehension and Beyond**. *Danqi Chen*. (PhD thesis 2019) [[paper]](https://purl.stanford.edu/gd576xb1833)
- **Open-Domain Question Answering**. *Danqi Chen*. (ACL 2020) [[slides]](https://github.com/danqi/acl2020-openqa-tutorial)

## Dataset
- **SQuAD: 100,000+ Questions for Machine Comprehension of Text**. *Pranav Rajpurkar, Jian Zhang, Konstantin Lopyrev, Percy Liang*. (EMNLP 2016) [[paper]](https://www.aclweb.org/anthology/D16-1264/)[[data]](https://github.com/rajpurkar/SQuAD-explorer/tree/master/dataset)
- **Know What You Don't Know: Unanswerable Questions for SQuAD**. *Pranav Rajpurkar, Robin Jia, Percy Liang*. (ACL 2018) [[paper]](https://www.aclweb.org/anthology/P18-2124/)[[data]](https://github.com/rajpurkar/SQuAD-explorer/tree/master/dataset) - ***SQuAD 2.0***
- **DuReader: a Chinese Machine Reading Comprehension Dataset from Real-world Applications**. *Wei He, Kai Liu, Jing Liu, Yajuan Lyu, Shiqi Zhao, Xinyan Xiao, Yuan Liu, Yizhong Wang, Hua Wu, Qiaoqiao She, Xuan Liu, Tian Wu, Haifeng Wang*. (ACL 2018 QA workshop) [[paper]](https://www.aclweb.org/anthology/W18-2605/)[[data]](https://github.com/baidu/DuReader)
- **HotpotQA: A Dataset for Diverse, Explainable Multi-hop Question Answering**. *Zhilin Yang, Peng Qi, Saizheng Zhang, Yoshua Bengio, William W. Cohen, Ruslan Salakhutdinov, Christopher D. Manning*. (EMNLP 2018) [[paper]](https://arxiv.org/abs/1809.09600)[[data]](https://hotpotqa.github.io/)
- **QuAC : Question Answering in Context**. *Eunsol Choi, He He, Mohit Iyyer, Mark Yatskar, Wen-tau Yih, Yejin Choi, Percy Liang, Luke Zettlemoyer*. (EMNLP 2018) [[paper]](https://arxiv.org/abs/1808.07036)[[data]](https://quac.ai/)
- **A Question-Entailment Approach to Question Answering**. *Asma Ben Abacha, Dina Demner-Fushman*. (BMC Bioinformatics 20, 511 (2019)) [[paper]](https://arxiv.org/abs/1901.08079)[[data]](https://github.com/abachaa/MedQuAD) - ***MedQuAD***
- **CoQA: A Conversational Question Answering Challenge**. *Siva Reddy, Danqi Chen, Christopher D. Manning*. (NAACL 2019 & TACL 2019) [[paper]](https://arxiv.org/pdf/1808.07042.pdf) [[data]](https://stanfordnlp.github.io/coqa/)
- **DROP: A Reading Comprehension Benchmark Requiring Discrete Reasoning Over Paragraphs**. *Dheeru Dua, Yizhong Wang, Pradeep Dasigi, Gabriel Stanovsky, Sameer Singh, Matt Gardner*. (NAACL-HLT 2019) [[paper]](https://arxiv.org/abs/1903.00161)
- **ChID: A Large-scale Chinese IDiom Dataset for Cloze Test**. *Chujie Zheng, Minlie Huang, Aixin Sun*. (ACL 2019) [[paper]](https://arxiv.org/abs/1906.01265)
- **Investigating Prior Knowledge for Challenging Chinese Machine Reading Comprehension**. *Kai Sun, Dian Yu, Dong Yu, Claire Cardie*. (TACL 2020) [[paper]](https://arxiv.org/abs/1904.09679)[[data]](https://github.com/nlpdata/c3) - ***C3***
- **TYDI QA: A Benchmark for Information-Seeking Question Answering in Typologically Diverse Languages**. *Jonathan H. Clark, Jennimaria Palomaki, Vitaly Nikolaev, Eunsol Choi, Dan Garrette, Michael Collins, Tom Kwiatkowski*. (TACL 2020) [[paper]](https://storage.cloud.google.com/tydiqa/tydiqa.pdf)[[data]](https://github.com/google-research-datasets/tydiqa)
- **MIMICS: A Large-Scale Data Collection for Search Clarification**. *Hamed Zamani, Gord Lueck, Everest Chen, Rodolfo Quispe, Flint Luu, Nick Craswell*. (CIKM 2020) [[paper]](https://arxiv.org/abs/2006.10174)[[data]](https://github.com/microsoft/MIMICS)

## Repository & Toolkit
- [AllenAI / allennlp-reading-comprehension](https://github.com/allenai/allennlp-reading-comprehension) - Reading comprehension tookit by the AllenNLP team
- [ankit-ai / BertQA-Attention-on-Steroids](https://github.com/ankit-ai/BertQA-Attention-on-Steroids)
- [alsang / BiDAF-pytorch](https://github.com/galsang/BiDAF-pytorch)
- [BshoterJ / awesome-kgqa](https://github.com/BshoterJ/awesome-kgqa) - A collection of some materials of knowledge graph question answering
- [cdqa-suite / cdQA](https://github.com/cdqa-suite/cdQA)
- [cdqa-suite / cdQA-annotator](https://github.com/cdqa-suite/cdQA-annotator)
- [deepset-ai / Haystack](https://github.com/deepset-ai/haystack)
- [Facebook Research / ParlAI](https://github.com/facebookresearch/ParlAI) - A framework for training and evaluating AI models on a variety of openly available dialogue datasets
- [efficientqa/retrieval-based-baselines](https://github.com/efficientqa/retrieval-based-baselines) - Tutorials on training and testing retrieval-based models
- [IndexFziQ / KMRC-Papers](https://github.com/IndexFziQ/KMRC-Papers)
- [lixinsu / RCZoo](https://github.com/lixinsu/RCZoo)
- [seriousran / awesome-qa](https://github.com/seriousran/awesome-qa)
- [Sogou / SMRCToolkit](https://github.com/sogou/SMRCToolkit) - Sogou Machine Reading Comprehension (SMRC) toolkit based on TensorFlow
- [songyingxin / BERT-MRC-RACE](https://github.com/songyingxin/BERT-MRC-RACE)
- [tangbinh / question-answering](https://github.com/tangbinh/question-answering)
- [THUNLP / RCPapers](https://github.com/thunlp/RCPapers)
- [xanhho / Reading-Comprehension-Question-Answering-Papers](https://github.com/xanhho/Reading-Comprehension-Question-Answering-Papers) - Survey on Machine Reading Comprehension
- [YingZiqiang / PyTorch-MRCToolkit](https://github.com/YingZiqiang/PyTorch-MRCToolkit)
- [yizhen20133868 / NLP-Conferences-Code](https://github.com/yizhen20133868/NLP-Conferences-Code)
- [ymcui / Chinese-RC-Datasets](https://github.com/ymcui/Chinese-RC-Datasets)
- [zcgzcgzcg1 / MRC_book](https://github.com/zcgzcgzcg1/MRC_book) - 《机器阅读理解：算法与实践》代码

## Blog Post
### English
- [Beyond Local Pattern Matching: Recent Advances in Machine Reading](https://ai.stanford.edu/blog/beyond-local-pattern-matching/)
### Chinese
- [changreal / 【总结向】MRC 经典模型与技术](https://blog.csdn.net/changreal/article/details/105074629)
- [changreal / 【总结向】从CMRC2019头部排名看中文MRC](https://blog.csdn.net/changreal/article/details/105363937)
- [humdingers / 2020法研杯阅读理解赛道第一名方案](https://github.com/humdingers/2020CAIL_LDLJ)
- [Luke / 机器阅读理解之多答案抽取](https://zhuanlan.zhihu.com/p/101248202)
- [科学空间 / 学会提问的BERT：端到端地从篇章中构建问答对](https://kexue.fm/archives/7630)
- [科学空间 / “非自回归”也不差：基于MLM的阅读理解问答](https://kexue.fm/archives/7148)
- [科学空间 / 万能的seq2seq：基于seq2seq的阅读理解问答](https://kexue.fm/archives/7115)
- [平安寿险PAI / 机器阅读理解探索与实践](https://zhuanlan.zhihu.com/p/109309164)
- [知乎问答 / 机器阅读理解方向有什么值得follow的大佬，网站等等](https://www.zhihu.com/question/358469127/answer/1028144909)

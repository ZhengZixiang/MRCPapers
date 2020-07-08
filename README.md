# MRCPapers
Worth-reading paper list and other awesome resources on Machine Reading Comprehension (MRC) and Question Answering (QA). Suggestions about adding papers, repositories and other resource are welcomed!

机器阅读理解领域值得一读的论文列表和其他相关资源集合。欢迎新增论文、代码仓库与其他资源等建议！

## Paper
- Bidirectional Attention Flow for Machine Comprehension (ICLR 2017) [[paper]](https://openreview.net/forum?id=HJ0UKP9ge)[[code]](https://allenai.github.io/bi-att-flow/) - ***BiDAF***
- Read + Verify: Machine Reading Comprehension with Unanswerable Questions (AAAI-2019) [[paper]](https://arxiv.org/pdf/1808.05759.pdf)[[unofficial code]](https://github.com/woshiyyya/Answer-Verifier-pytorch) - ***Verfier***
- Cognitive Graph for Multi-Hop Reading Comprehension at Scale (ACL 2019) [[paper]](https://arxiv.org/abs/1905.05460) - ***CogQA***
- Dual Co-Matching Network for Multi-choice Reading Comprehension (CoRR 2019) [[paper]](https://arxiv.org/abs/1901.09381) - ***DCMN***
- A Multi-Type Multi-Span Network for Reading Comprehension that Requires Discrete Reasoning (EMNLP 2019) [[paper]](https://arxiv.org/abs/1908.05514) - ***MTMSN***
- Tag-based Multi-Span Extraction in Reading Comprehension (CoRR 2019) [[paper]](https://arxiv.org/abs/1909.13375)
- SG-Net: Syntax-Guided Machine Reading Comprehension (AAAI 2020) [[paper]](https://arxiv.org/abs/1908.05147)[[code]](https://github.com/cooelf/SG-Net) - ***SG-Net***
- DCMN+: Dual Co-Matching Network for Multi-choice Reading Comprehension (AAAI 2020) [[paper]](https://arxiv.org/abs/1908.11511.pdf) - ***DCMN+***
- Retrospective Reader for Machine Reading Comprehension [[paper]](https://arxiv.org/abs/2001.09694)[[Chinese blog]](https://mp.weixin.qq.com/s?__biz=MzIwMTc4ODE0Mw==&mid=2247502891&idx=1&sn=8f3d552ee384544d0b9a868e01b91ed9&key=5fa67e91c99877c949e72c80560ca0bb5dc99de132236c4b530784a3c3c2cc93a94dcd482b4968b128c7bc7553888c5df30cc4f734abb1a63a2bd02402645a9b966bd4291e333ef13e861eb06c80822a&ascene=1&uin=Mjg1NTM0NDcyMw%3D%3D&devicetype=Windows+10&version=6208006f&lang=zh_CN&exportkey=A%2BCwv01k%2FtyMn%2Ft38iF3KbY%3D&pass_ticket=nkIz09BYlgtIrHo7XkM4ahTkS8sck64jbLwU0LotdcTnxt2f%2FIuSGmn33Pc7gW1f) - ***Retro-Reader***
- DeFormer: Decomposing Pre-trained Transformers for Faster Question Answering (ACL 2020) [[paper]](https://arxiv.org/abs/2005.00697)[[code]](https://github.com/StonyBrookNLP/deformer) - ***DeFormer***

## Survey & Review & Tutorial
- Neural Machine Reading Comprehension: Methods and Trends (Applied Sciences 2019) [[paper]](https://arxiv.org/abs/1907.01118)
- Research on Machine Reading Comprehension and Textual Question Answering (Minghao Hu's PhD thesis 2019 in Chinese) [[paper]](https://github.com/huminghao16/thesis)
- Neural Reading Comprehension and Beyond (Danqi Chen's PhD thesis 2019) [[paper]](https://purl.stanford.edu/gd576xb1833)
- Open-Domain Question Answering (ACL 2020) [[slides]](https://github.com/danqi/acl2020-openqa-tutorial)

## Dataset
- SQuAD: 100,000+ Questions for Machine Comprehension of Text (EMNLP 2016) [[paper]](https://www.aclweb.org/anthology/D16-1264/)[[data]](https://github.com/rajpurkar/SQuAD-explorer/tree/master/dataset) - ***SQuAD 1.1***
- Know What You Don't Know: Unanswerable Questions for SQuAD (ACL 2018) [[paper]](https://www.aclweb.org/anthology/P18-2124/)[[data]](https://github.com/rajpurkar/SQuAD-explorer/tree/master/dataset) - ***SQuAD 2.0***
- DuReader: a Chinese Machine Reading Comprehension Dataset from Real-world Applications (ACL 2018 QA workshop) [[paper]](https://www.aclweb.org/anthology/W18-2605/)[[data]](https://github.com/baidu/DuReader) - ***DuReader***
- HotpotQA: A Dataset for Diverse, Explainable Multi-hop Question Answering (EMNLP 2018) [[paper]](https://arxiv.org/abs/1809.09600)[[data]](https://hotpotqa.github.io/) - ***HotpotQA***
- A Question-Entailment Approach to Question Answering (BMC Bioinformatics 20, 511 (2019)) [[paper]](https://arxiv.org/abs/1901.08079)[[data]](https://github.com/abachaa/MedQuAD) - ***MedQuAD***
- CoQA: A Conversational Question Answering Challenge (NAACL 2019 & TACL 2019) [[paper]](https://arxiv.org/pdf/1808.07042.pdf) [[data]](https://stanfordnlp.github.io/coqa/) - ***CoQA***
- DROP: A Reading Comprehension Benchmark Requiring Discrete Reasoning Over Paragraphs (NAACL 2019) [[paper]](https://arxiv.org/abs/1903.00161) - ***DROP***
- ChID: A Large-scale Chinese IDiom Dataset for Cloze Test (ACL 2019) [[paper]](https://arxiv.org/abs/1906.01265) - ***ChID***
- Investigating Prior Knowledge for Challenging Chinese Machine Reading Comprehension (TACL 2020) [[paper]](https://arxiv.org/abs/1904.09679)[[data]](https://github.com/nlpdata/c3) - ***C3***
- TYDI QA: A Benchmark for Information-Seeking Question Answering in Typologically Diverse Languages (TACL 2020) [[paper]](https://storage.cloud.google.com/tydiqa/tydiqa.pdf)[[data]](https://github.com/google-research-datasets/tydiqa) - ***TyDi QA***
- MIMICS: A Large-Scale Data Collection for Search Clarification (CoRR 2020) [[paper]](https://arxiv.org/abs/2006.10174)[[data]](https://github.com/microsoft/MIMICS) - ***MIMICS***


## Repository & Toolkit
- [AllenAI / allennlp-reading-comprehension](https://github.com/allenai/allennlp-reading-comprehension) - Reading comprehension tookit by the AllenNLP team
- [ankit-ai / BertQA-Attention-on-Steroids](https://github.com/ankit-ai/BertQA-Attention-on-Steroids)
- [alsang / BiDAF-pytorch](https://github.com/galsang/BiDAF-pytorch)
- [cdqa-suite / cdQA](https://github.com/cdqa-suite/cdQA)
- [cdqa-suite / cdQA-annotator](https://github.com/cdqa-suite/cdQA-annotator)
- [deepset-ai / Haystack](https://github.com/deepset-ai/haystack)
- [efficientqa/retrieval-based-baselines](https://github.com/efficientqa/retrieval-based-baselines) - Tutorials on training and testing retrieval-based models
- [IndexFziQ / KMRC-Papers](https://github.com/IndexFziQ/KMRC-Papers)
- [lixinsu / RCZoo](https://github.com/lixinsu/RCZoo)
- [seriousran / awesome-qa](https://github.com/seriousran/awesome-qa)
- [Sogou / SMRCToolkit](https://github.com/sogou/SMRCToolkit) - Sogou Machine Reading Comprehension (SMRC) toolkit based on TensorFlow
- [songyingxin / BERT-MRC-RACE](https://github.com/songyingxin/BERT-MRC-RACE)
- [tangbinh / question-answering](https://github.com/tangbinh/question-answering)
- [THUNLP / RCPapers](https://github.com/thunlp/RCPapers)
- [YingZiqiang / PyTorch-MRCToolkit](https://github.com/YingZiqiang/PyTorch-MRCToolkit)
- [ymcui / Chinese-RC-Datasets](https://github.com/ymcui/Chinese-RC-Datasets)

## Blog Post
### English
- [Beyond Local Pattern Matching: Recent Advances in Machine Reading](https://ai.stanford.edu/blog/beyond-local-pattern-matching/)
### Chinese
- [“非自回归”也不差：基于MLM的阅读理解问答](https://kexue.fm/archives/7148)
- [万能的seq2seq：基于seq2seq的阅读理解问答](https://kexue.fm/archives/7115)

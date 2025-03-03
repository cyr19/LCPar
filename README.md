# LCPar

This repo contains the code for dependency parsers we used in [Syntactic Language Change in English and German: Metrics, Parsers, and Convergences](https://arxiv.org/abs/2402.11549), which intergrades the following GitHub repos and libraries:

* [NeuroNLP2](https://github.com/XuezheMax/NeuroNLP2): `StackPointer` ([Ma et al., 2018](https://aclanthology.org/P18-1130/))

* [SuPar](https://github.com/yzhangcs/parser): `Deep Biaffine` ([Dozat and Manning, 2017](https://openreview.net/forum?id=Hk95PK9le)), `CRF2O` ([Zhang et al., 2020](https://aclanthology.org/2020.acl-main.302/))
  
* [TowerParse](https://github.com/codogogo/towerparse): `TowerParse` ([Glavas and Vulic, 2021](https://aclanthology.org/2021.findings-acl.431/))

* [Stanza](https://stanfordnlp.github.io/stanza/): `Stanza` ([Qi et al., 2020](https://arxiv.org/abs/2003.07082)), `CoreNLP` ([Manning et al., 2014](https://aclanthology.org/P14-5010/))


## Checkpoints

We release all checkpoints trained by ourselves, which can be downloaded from:
| Parser | English | German |
|--------|--------|--------|
| StackPointer | [en](https://drive.google.com/file/d/1B4IItwuN5TQzWtSXEeEidtbrQZVW-AUu/view?usp=sharing) | [de](https://drive.google.com/file/d/13ISEZvAeAWX-7f6sJE4c-CRVkmaBd-GU/view?usp=sharing) |
| Biaffine | [en](https://drive.google.com/file/d/17rWkylDxDeSFWbU404wQQYA7XW8xP5fV/view?usp=sharing) | [de](https://drive.google.com/file/d/1MGTI4UIPmc-n8CBHXG3aTtsOPay59_xs/view?usp=sharing) |
| CRF2o | [en](https://drive.google.com/file/d/11npLVAwl2TCiWJXZ707JKEstctV1n3Z4/view?usp=sharing) | [de](https://drive.google.com/file/d/13NtQqyTv96rjCHMoNsras___COFSmUSy/view?usp=sharing) |


Those checkpoints were trained on the merged [UD v2.12](https://lindat.mff.cuni.cz/repository/xmlui/handle/11234/1-5150) treebank, as described in our paper.

Please check the original GitHub for training. 

## Example Usage
Check `parser.yaml` for the environment to run this repo. You can also easily create a similar environment using conda:
```
conda env create -n my_env -f parser.yaml
```

```
from stackpointer_parser import StackPointerParser
parser = StackPointerParser(ckpt_dir='my_checkpoint/', language='en', batch_size=64)
sentences = ['I love dogs and cats.', 'This is great.']
# output a string containing parsing results in collu format.
results = parser.parse(sentences=sentences, out='conllu', tokenized=False)
```

For more examples, please check the scripts `parse_all.py` and `evaluate.py`.

## Citation
If you use the code in this repo, we'd be happy if you cite us!
```
@misc{chen2024syntactic,
      title={Syntactic Language Change in English and German: Metrics, Parsers, and Convergences}, 
      author={Yanran Chen and Wei Zhao and Anne Breitbarth and Manuel Stoeckel and Alexander Mehler and Steffen Eger},
      year={2024},
      eprint={2402.11549},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```

If you use the code here for dependency parsing, please also cite the original parser papers.
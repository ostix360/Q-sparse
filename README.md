# Q-sparse
Unofficial implementation of Q-sparse: All Large Language Models can be Fully Sparsely-Activated https://arxiv.org/abs/2407.10969

## Introduction

This repository contains the implementation of the paper [Q-sparse: All Large Language Models can be Fully Sparsely-Activated](https://arxiv.org/abs/2407.10969). The paper proposes a new method to train large language models with sparse activations.

## Requirements

- Python 3.9+
- PyTorch

## Installation

```bash
pip install git+https://github.com/ostix360/Q-sparse.git
```

## Usage

```python
import torch
from q_sparse import QSparseLinear

model = QSparseLinear(10, 10, bias=False) # input_dim=10, output_dim=10
input = torch.randn(1, 10)
output = model(input)
```

## Tests

```bash
python -m unittest discover -s tests -p QSparseTest.py
```

## Contributing

Contributions are welcome! For bug reports or requests please submit an issue.


## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.


## Citation

```
@misc{wang2024qsparselargelanguagemodels,
      title={Q-Sparse: All Large Language Models can be Fully Sparsely-Activated}, 
      author={Hongyu Wang and Shuming Ma and Ruiping Wang and Furu Wei},
      year={2024},
      eprint={2407.10969},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2407.10969}, 
}
```

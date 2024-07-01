# LLM Selection via Fine-tuning Scaling Law

This repository contains the code for our ICML 2024 paper [Selecting Large Language Model to Fine-tune via Rectified Scaling Law](https://arxiv.org/pdf/2402.02314.pdf) by [Haowei Lin](https://linhaowei1.github.io/), [Baizhou Huang](https://scholar.google.com/citations?user=1Zx1wi8AAAAJ), [Haotian Ye](https://haotianye.com/), [Qinyue Chen](https://scholar.google.com/citations?user=y13QmxkAAAAJ&hl=zh-CN&oi=ao), [Zihao Wang](https://zhwang4ai.github.io/), [Sujian Li](https://pku-tangent.github.io/), [Jianzhu Ma](https://majianzhu.com/), [Xiaojun Wan](https://wanxiaojun.github.io/), [James Zou](https://www.james-zou.com/), [Yitao Liang](https://scholar.google.com/citations?user=KVzR1XEAAAAJ).

## Fine-tuning Performance of 30 models
The fine-tuning performance of the 30 models on various sizes (from 0 to 1638400) of subsets from 3 datasets (WMT19, Gigaword, FLAN) is presented in `benchmark/`.

## Source codes

There are three files in `src/`

- `draw.py`: Codes for generating all experimental results presented in the paper. The generated results will be saved in `results/`
- `fit_law.py`" Codes for fitting two fine-tuning scaling laws including vanilla law and our rectified law.
- `model_select.py`: Codes for implementing different model selection methods including ZeroShot, SubTuning, ModelSize, OurFit, VanillaFit and AtS.

## Environments

    ```
    pip install -r requirements.txt
    ```

## Scaling Law Fitting

```python
from fit_law import fit_our_law, our_law_transform
import numpy as np

# preprocess training data
x_train = np.array([200,400,800,1600,3200,6400,12800,25600,51200,102400,204800,409600,819200,1638400])
y_train = np.array([4.247949918,4.188166777,4.086900075,3.946466605,3.808449984,3.645450115,3.420799971,3.165299892,2.91552496,2.652625084,2.382649899,2.151550055,1.91655004,1.742699981])
log_x_train = np.log(x_train)
log_y_train = np.log(y_train)

# fit the law with training data
fitted_params, bestloss = fit_our_law(log_x_train, log_y_train)

# transform test data
log_x_test = np.log(1e6)
pred_y_test = np.exp(our_law_transform(log_x_test, *fitted_params))
```

## Model Selection

- input data: the performance of different models on various size of subsets. Suppose the data budget is $B$, the data format is a pandas dataframe similar to `benchmark/flan.csv`.

    |  config name   | ... | $B/2^j$ | ... |$B/2$ |$B$|
    |  ----  | ----  |  ----  | ----  | --| -|
    | model 1 | ... | $loss_{1j}$ | ...| ... | ... |
    | model 2 | ... | $loss_{2j}$ | ...| ... | ... |
    | model 3 | ... | $loss_{3j}$ |...| ... | ... |
    | ... | ... | ... | ... | ... | ... |

- usage:
    ```
    from model_select import ats_select

    data = pd.read_csv(f'benchmark/flan.csv', index_col=0)
    data.columns = [int(col) if col.isdigit() else col for col in data.columns]
    rank, _ = ats_select(data, max_data_num=data_budget, predict_data_num=number) # return the model ranking of AtS selection
    ```

## Citation

Please cite our paper if you use this code or part of it in your work:

@inproceedings{lin2024class,
      title={Class Incremental Learning via Likelihood Ratio Based Task Prediction}, 
      author={Haowei Lin and Yijia Shao and Weinan Qian and Ningxin Pan and Yiduo Guo and Bing Liu},
      year={2024},
      booktitle={International Conference on Learning Representations}
}
# TECTM
The repository is for the paper titled "Topic Model on learning Inter and Intra Topic Structure" by 
Chun-Wa Leung, Akihiko Takano, and Takeshi Abekawa.

## Dependencies
+ python 3.7.10
+ pyro-ppl 1.8.0
+ torch 1.11.0

## Datasets
The datasets can be found in following reference
+ 20Newsgroups: from `sklearn.datasets` package 
+ Reuters-21578: from `nltk` corpus
+ NeurIPS: https://www.kaggle.com/datasets/rowhitswami/nips-papers-1987-2019-updated
+ UN-debates: https://www.kaggle.com/datasets/unitednations/un-general-debates 


## Run
To execute the TECTM, run the following command for the default settings. 
For detailed configurations, you may read the parameter specifications in main.py. 
```
python main.py
```

## Citation

```
@article{leung2022topic,
  title={Topic Model on learning Inter and Intra Topic Structure},
  author={Chun-Wa Leung, Akihiko Takano, Takeshi Abekawa},
  year={2022}
}
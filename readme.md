## TSF-Prompt: Integrated Heterogeneous Prompts for Unlocking LLMs in Time Series Forecasting

------

## Introduction

------

TSF-Prompt comprises four components: dual-path prompting, semantic space embedding, cross-modal alignment, and time series forecasting. First, hard and soft prompts are constructed based on temporal data. Subsequently, semantic space embedding is performed through the pre-trained LLM and patch reprogramming module. Since soft prompts are inherently generated within this space, no additional embedding is required. Next, cross-modal alignment is executed. Finally, the fused information is fed into the pre-trained LLM and projected through output layers to generate forecasting results.  

![Framework](.\framework.png)

## Requirements

------

Use python 3.12 from Anaconda

- accelerate==1.7.0
- bitsandbytes==0.45.5
- certifi==2025.4.26
- charset-normalizer==3.4.2
- colorama==0.4.6
- contourpy==1.3.2
- cycler==0.12.1
- filelock==3.13.1
- fonttools==4.57.0
- fsspec==2024.6.1
- hf-xet==1.1.1
- huggingface-hub==0.31.1
- idna==3.10
- Jinja2==3.1.4
- joblib==1.4.2
- kiwisolver==1.4.8
- MarkupSafe==2.1.5
- matplotlib==3.10.1
- mpmath==1.3.0
- networkx==3.3
- numpy==2.1.2
- packaging==25.0
- pandas==2.2.3
- patool
- pillow==11.0.0
- psutil==7.0.0
- pyparsing==3.2.3
- python-dateutil==2.9.0.post0
- pytz==2025.2
- PyYAML==6.0.2
- regex==2024.11.6
- requests==2.32.3
- safetensors==0.5.3
- scikit-base==0.12.2
- scikit-learn==1.6.1
- scipy==1.15.2
- setuptools==80.1.0
- six==1.17.0
- sktime==0.37.0
- sympy==1.13.3
- threadpoolctl==3.6.0
- tokenizers==0.21.1
- torch==2.7.0+cu118
- torchaudio==2.7.0+cu118
- torchvision==0.22.0+cu118
- tqdm==4.67.1
- transformers==4.51.3
- typing_extensions==4.12.2
- tzdata==2025.2
- urllib3==2.4.0
- wheel==0.45.1

To install all dependencies:

```
pip install -r requirements.txt
```

## Datasets

------

You can access the well pre-processed datasets from [[Google Drive\]](https://drive.google.com/drive/folders/13Cg1KYOlzM5C7K8gK8NfC-F3EYxkM3D2) and [[CarbonMonitor\]](https://www.carbonmonitor.org.cn/), then place the downloaded contents under `./dataset`

## Quick Demos

------

1. Download `config.json`, `merges.txt`, `pytorch_model.bin`, and `vocab.json` from [[Hugging Face\]](https://huggingface.co/openai-community/gpt2), then place the downloaded contents under `./gpt2`.
2. Install all dependencies listed in `requirements.txt`. We provide a ready-to-run demo; simply execute `demo.py` to test TSF-Prompt on the ETTh1 dataset. For other datasets, download the corresponding dataset files additionally.

## Acknowledgement

------

Our implementation adapts [Time-Series-Library](https://github.com/thuml/Time-Series-Library), [Time-LLM](https://github.com/KimMeen/Time-LLM) and [TEMPO](https://github.com/DC-research/TEMPO) as the code base and have extensively modified it to our purposes. We thank the authors for sharing their implementations and related resources.

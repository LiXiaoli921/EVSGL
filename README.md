## Overview
### EVSGL: An Enhanced View-specific Graph Learning Framework for Multiview Clustering, 2022
Here we provide the implementation of the EVSGL method, which is submitted to IEEE TCYB.

## Architecture
+ `data/` contains the six real-world dataset file;
+ `evoke.py` contains the Probabilitic graph generation module
+ `model.py/` contains:
  + the implementation of the view-specific GMIM module;
  + the implementation of the view-wise attention fusion module;
  + the implementation of the total EVSGL framework;
+ `layers.py/` contains:
  + the implementation of the GCN layer;
  + the implementation of the Cluster layer;
+ `utils.py` contains data preprocessing of the six real-world dataset;
+ `metrics.py` contains four clustering metrics ACC, NMI, ARI and F1;
+ `run.py` training and validing the model;
+ `parameters.py` contains parameters of different datasets; 

```bash
$ python run.py
```

## Dataset

<img width="467" alt="image" src="https://user-images.githubusercontent.com/59239422/175770330-99a3ac67-c111-4df3-918e-ee325a27d804.png">

Download from Google Drive: [Dataset](https://drive.google.com/drive/folders/1P3-9Kk1ohNrw7-uMjpL49Vp7JQppPQHn?usp=sharing)

## Dependencies
The script has been tested running under Python 3.6.13, with the following packages installed (along with their dependencies):
+ `numpy==1.19.5`
+ `scipy==1.5.4`
+ `pytorch==1.14.0`
+ `scikit-learn==0.24.2`
+ `pandas==1.1.5`
+ `munkres==1.1.4`



## References
The source code of the baselines
[MVGL](https://github.com/kunzhan/MVGL)
[MCGC](https://github.com/kunzhan/MCGC)
[GMC](https://github.com/cshaowang/gmc)
[CGD](https://github.com/ChangTang/CGD)
[O2MAC](https://github.com/googlebaba/WWW2020-O2MAC)
[CDMGC](https://github.com/huangsd/CDMGC)
[DGF](https://github.com/youweiliang/Multi-view\_Graph\_Learning)

## License

MIT

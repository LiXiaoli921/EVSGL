## Overview
### EVSGL: An Enhanced View-specific Graph Learning Framework for Multiview Clustering, 2022


Here we provide the implementation of the EVSGL method , along with an execution example (on the Krogan-core dataset). The repository is organized as follows:
+ `data/` contains the six real-world dataset file;
+ `evoke.py` contains the Probabilitic graph generation module
+ `model.py/` contains:
  + the implementation of the view-specific GMIM module;
  + the implementation of the view-wise attention fusion module;
  + the implementation of the clustering-friendly fine-tune module;
+ `layer.py/` contains:
  + the implementation of the GCN layer;
  + the implementation of the Cluster layer;
+ `utils.py` contains data preprocessing of the six real-world dataset;
+ `metrics.py` contains four clustering metrics ACC, NMI, ARI and F1;
+ `run.py` training and validing the model;
+ `parameters.py` contains parameters of different datasets; 

```bash
$ python run.py
```
## Architecture

![image](https://github.com/aI-area/AdaPPI/blob/main/framework.png)


## Dataset
Download from Google Drive: ![url](https://drive.google.com/drive/folders/1P3-9Kk1ohNrw7-uMjpL49Vp7JQppPQHn?usp=sharing)
details

## Dependencies
The script has been tested running under Python 3.6.9, with the following packages installed (along with their dependencies):


## References
You may also be interested in the related articles：

## License

MIT

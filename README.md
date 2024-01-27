# anomaly-detection-3d-printing
During the 3d printing process potential anomalies are expected be supervised and warned. This model utilizing a AnoGAN to recognize possible anomalies appearing on layer image of the printing process.   

![Figure_1_p11311](https://github.com/xpzoe/anomaly-detection-3d-printing/assets/92227026/e25fad0c-1439-4a0e-9b44-fda8bee1756f)
The red part indicates an anomaly caused high cumulated energy.  
# Run
## Train
run anogan_train.py, then type in the path to training dataset.
## Test
run anogan_detect.py, then type in the path to testing dataset.

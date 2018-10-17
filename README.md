## Usage

**Step 1. Clone this repository to local.**
```
git clone https://github.com/JoyJulianGomes/CapsNet-Keras.git capsnet-keras
cd capsnet-keras
```
Create a new branch and make necessary commits in that branch
```
git branch a_new_branch
```
Refer to git workflow.txt to see detailed instruction on git usage

**Step 2. Train a CapsNet on MNIST**

Training with default settings:
```
python capsulenet.py
```
Testing with default settings:
```
python capsulenet.py -t
```
Testing with trained weights:
```
python capsulenet.py -w ./weight/trained_model-11-19.h5 -t
```
PrimaryCap/DigitCap/Mask Layer output:
```
python capsulenet.py  -w ./weight/trained_model-11-19.h5 -l pc
python capsulenet.py  -w ./weight/trained_model-11-19.h5 -l dc
python capsulenet.py  -w ./weight/trained_model-11-19.h5 -l mask
python capsulenet.py  -w ./weight/trained_model-11-19.h5 -l all
```
For Frame Difference and EPE:
```
python capsulenet.py  -w ./weight/trained_model-11-19.h5 -f
```

More detailed usage run for help:
```
python capsulenet.py -h
```

**Train on multi gpus**

```
python capsulenet-multi-gpu.py --gpus 2
```
It will automatically train on multi gpus for 50 epochs and then output the performance on test dataset.
But during training, no validation accuracy is reported.

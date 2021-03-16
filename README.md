![image](https://github.com/MistarQ/SIMGAN/blob/main/%E5%8F%91%E8%89%B2%E8%BD%AC%E6%8D%A2.png)


## Usage

### Install dependencies

```
python -m pip install -r requirements.txt
```

Our code was tested with python 3.6  and PyToch 1.0.0 or 1.2.0

###  Train
To train SIMGAN model on three unpaired images, put the first training image under `datas/task_name/trainA` and the second training image under `datas/task_name/trainB`, and the sthird training image under `datas/task_name/trainC`, and run

```
python train.py --input_name <input_name> --root <datas/task_name>
```
For example, 
```
python train.py --input_name apple --root datas/apple
```



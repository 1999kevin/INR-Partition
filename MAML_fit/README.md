## Data Preparation
Please download LSUN dataset from ```http://dl.yf.io/lsun/scenes/```



## Training the MAML algorithm in LSUN
Please use the following command to train the original MAML algorithm without partition:
```
python run_script.py  --mode train --data_root <path to lsun dataset> 
``` 

Please use the following command to train the MAML algorithm with partition:
```
python run_script.py  --mode train --data_root <path to lsun dataset> --MAML_partition --mask <grid or hfs>
``` 

The above commands will train a MAML algorithm on the LSUN dataset and store the trained models on directory ```trained_models```
If not changed, the model that is trained without partition is named ``MAML-None.pt``, the model that is trained with PoG is named ``partition-grid.pt`` and the model that is trained with PoS is named ``partition-hfs.pt``

## Test the MAML algorithm in LSUN
Please use the following command Test train the original MAML algorithm without partition:
```
python run_script.py  --mode val --data_root <path to lsun dataset> --trained_model_path ./trained_models/MAML-None.pt
``` 

As mentioned in the paper, we can also use partition only in the testing phase, for example:
```
python run_script.py  --mode val --data_root <path to lsun dataset> --MAML_partition --mask <grud or hfs> --trained_model_path ./trained_models/MAML-None.pt
``` 

Or use partition both in the training and testing phase: 
```
python run_script.py  --mode val --data_root <path to lsun dataset> --MAML_partition --mask <grud or hfs> --trained_model_path <./trained_models/partitionMAML-grid.pt or ./trained_models/partitionMAML-hfs.pt>
``` 
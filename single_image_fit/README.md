# Fit a single image
We provide the script ```single_image_fit.py ``` to fit one single image with different architectures.

The convergence curve for different architectures can be obtained via: 
```
python single_image_fit.py  --img_path <path to img> --log_save_path <path to save the training loss> --architecture <architecture name>  --mask <mask type> --hidden_features <hidden_features> --hidden_layers <hidden_layers> 
```

For example, fitting an image via a relu mlp with 512 hidden features and 3 hidden layers requires the follow command: 
```
python single_image_fit.py  --img_path ../data/001.jpg --architecture relu_mlp  --hidden_features 512 --hidden_layers 3
```


# Evaluate partition method with ReLU MLP
To illustrate the performance of partition on the relu mlp, please use the follow parameters to generate MLP architectures with close amount of total parameters: 


|architecture |mask       |hidden_features |hidden_layers   | total parameters | Comments      |         
|-------------|-----------|----------------|----------------|------|---------|
|relu-mlp     |-          |512               |3         |  912899   |A MLP with relu activation with 512 hidden_features and 3 hidden_layers|
|multi-relu     |hfs9          |148               |3        |  920439   |Partition into 9 parts with PoS rules, and fit with 9 smaller relu MLPs while each MLP contains 148 hidden features and 3 hidden layers|
|multi-relu     |9interval          |148               |3         |  920439  |Partition into 9 parts with PoG rules, and fit with 9 smaller relu MLPs while each MLP contains 148 hidden features and 3 hidden layers|



Please use the follow commands to generate the training curves of different architectures on the demon image:
``` 
python single_image_fit.py  --img_path ../data/001_L.png --architecture relu-mlp  --hidden_features 512 --hidden_layers 3 --log_save_path ./001_L_save_results/ --epoch 10000 --steps_til_summary 1000
python single_image_fit.py  --img_path ../data/001_L.png --architecture multi-relu  --mask hfs9 --hidden_features 148 --hidden_layers 3 --log_save_path ./001_L_save_results/ --epoch 10000 --steps_til_summary 1000
python single_image_fit.py  --img_path ../data/001_L.png --architecture multi-relu  --mask 9interval --hidden_features 148 --hidden_layers 3 --log_save_path ./001_L_save_results/ --epoch 10000 --steps_til_summary 1000
```

# Evaluate partition method with SIREN
To illustrate the performance of partition on the SIREN, please use the follow parameters to generate SIREN architectures with close amount of total parameters: 

|architecture |mask       |hidden_features |hidden_layers   | total parameters | Comments      |       
|-------------|-----------|----------------|----------------|-----|----------|
|Siren     |-          |512               |3             |   791043  |A MLP with relu activation with 512 hidden_features and 3 hidden_layers|
|multi-Siren     |hfs9          |170               |3         |  794097  |Partition into 9 parts with PoS rules, and fit with 9 smaller Siren while each Siren MLP contains 170 hidden features and 3 hidden layers|
|multi-Siren    |9interval          |170               |3         |  794097  |Partition into 9 parts with PoG rules, and fit with 9 smaller Siren while each Siren MLP contains 170 hidden features and 3 hidden layers|


Please use the follow commands to generate the training curves of different architectures on the demon image:
``` 
python single_image_fit.py  --img_path ../data/001_L.png --architecture Siren  --hidden_features 512 --hidden_layers 3 --log_save_path ./001_L_save_results/ --epoch 1500
python single_image_fit.py  --img_path ../data/001_L.png --architecture multi-Siren  --mask hfs9 --hidden_features 170 --hidden_layers 3 --log_save_path ./001_L_save_results/ --epoch 1500
python single_image_fit.py  --img_path ../data/001_L.png --architecture multi-Siren  --mask 9interval --hidden_features 170 --hidden_layers 3 --log_save_path ./001_L_save_results/ --epoch 1500
```


# See the results
Then use the follow commands to see the training curves:
```
python single_image_fit_curve_plots.py 
```

The training curves will be similar as:
<p align="center">
<img src=001_L_save_results/001_L_siren.png width="300" height="200"/>
<img src=001_L_save_results/001_L_relu.png width="300" height="200"/>
</p>

It is very easy to define your own mask. Please see the directory  ```masks```.
There are some different between the total parameters here and the total parameters in the paper. It is because we use one-channel images in the paper but RGB-channel images here. This different has not effect on the major results.
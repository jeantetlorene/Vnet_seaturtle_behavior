# Fully Convolutional Neural Network: A solution to infer animal behaviours from multi-sensor data

This github directory aims to provide the functions used in the article "Fully Convolutional Neural Network: A solution to infer animal behaviours from multi-sensor data" to automatically identify sea turtle behaviors from accelerometers. The data used in this article was collected by Damien Chevallier (CNRS France) and is available in Zenodo: https://zenodo.org/records/11643602

Link to the article : https://www.sciencedirect.com/science/article/abs/pii/S0304380021001253

keywords : Deep learning, Accelerometer, Sea turtle, Ecology, Behavioural classification, Convolutional neural network

The functions proposed in this github are easily reusable and applicable to another context. 
The scripts are organized into Python files, with Training_VNet_GT serving as the main file to run. It utilizes functions from the other Python files: Global_generator_GT, Prediction_GT, Training_helper_GT, and Vnet_Architecture_GT. The notebook Notebook=Training_the_Vnet_Green_Turtles can be run independently and contains all the necessary functions to operate on its own.


# Authors 
Lorène Jeantet [1], Vincent Vigon [2], Sébastien Geiger [1], Damien Chevallier [1,3] 

[1]: Université de Strasbourg, CNRS, IPHC UMR 7173, 67000, STrasbourg, France  
[2]: Université de Strasbourg, UFR math-info, 7 rue Descartes, 67081, Strasbourg, France   
[3]: UMR BOREA, CNRS-7208/MNHN/UPMC/IRD-207/UCN/UA


# Overview of the study
In this study, we developped a fully convolutional neural network, the V-Net, to automatically identify the behaviors of green turtle from acceleration, gyroscope, depth sensor data. 
The V-net was first described and developed by Milletari et al. (2016) to treat medical images (3D signals). It is an evolution of the U-net, developed for biomedical 2D images (Ronneberger et al., 2015). In this work, we present an adapted architecture for the V-net which fits with our multi-sensor data, i.e. 1D temporal series.

![vnet_architecture_vs2](https://github.com/jeantetlorene/Vnet_seaturtle_behavior/assets/105348746/cd444773-6307-4e30-a0dd-53f76e900d17)

With minimal preprocessing, we were able to obtain a F1-score of 81.1% and a Global accuracy of 97.2%. The 6 behavioral categories identified were : Breathing, Feeding, Gliding, Resting, Scratching, and Swimming. Any other observed behavior was categorized as Other. 

Comparaison of the predictions of the V-Net (in red) and the observed behaviors (in blue) on the testing dataset (3 individuals) : 
![compa on test](https://github.com/jeantetlorene/Vnet_seaturtle_behavior/assets/105348746/179ac3b6-7637-479d-aef3-cf3d4917359c)


Confusion Matrix on the testing dataset (3 individuals) : 
![whole validation](https://github.com/jeantetlorene/Vnet_seaturtle_behavior/assets/105348746/a9407897-1e50-4430-96d4-419e5fd269e7)

# References 
Jeantet, L., Planas-Bielsa, V., Benhamou, S., Geiger, S., Martin, J., Siegwalt, F., Lelong, P.,Gresser, J., Etienne, D., Hi´elard, G., Arque, A., Regis, S., Lecerf, N., Frouin, C.,Benhalilou, A., Murgale, C., Maillet, T., Andreani, L., Campistron, G., Delvaux, H.,Guyon, C., Richard, S., Lefebvre, F., Aubert, N., Habold, C., le Maho, Y.,Chevallier, D., 2020a. Behavioural inference from signal processing using animalborne multi-sensor loggers: a novel solution to extend the knowledge of sea turtle ecology. R. Soc. Open Sci. 7, 200139 https://doi.org/10.1098/rsos.200139. \\

Milletari, F., Navab, N., Ahmadi, S.A., 2016. V-Net: fully convolutional neural networks for volumetric medical image segmentation. In: Proc. - 2016 4th Int. Conf. 3D Vision, 3DV 2016, pp. 565–571. https://oi.org/10.1109/3DV.2016.79. \\

Ronneberger, O., Fischer, P., Brox, T., 2015. U-net: convolutional networks for biomedical image segmentation. Lecture Notes in Computer Science (Including Subseries Lecture Notes in Artificial Intelligence and Lecture Notes in Bioinformatics. Springer Verlag, pp. 234–241. https://doi.org/10.1007/978-3-319-24574-4_28 



# Library 

python=3.10.12
tensorflow=2.15
pandas=2.0.3
numpy=1.25.2
matplotlib=3.8.4
scikit-learn==1.2.2

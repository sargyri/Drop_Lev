# Drop_Lev

Welcome to the DropLev project. In this depository you can find the data and machine learning algorithm that was develeoped for the prediction of surface tension of acoustically levitated droplets containing surfactant solutions (see Ref. [1]).

A manuscript has been submitted on the use of this machine learning algorithm for the surface tension determination of acoutsically levitated droplets containing PNIPAM microgels.

The design of the levitator used in these studies was model Mk1 from Ref. [2], which was inspired by the TinyLev levitator [3]. 

The in-silico data were generated from the theory developed by Marston _et al._ [4-6]. The equation used for the generation of the theoretical contours can be found in Ref. [6].

In the folder ''Data'' you can download the zip folders containing the data used for the training and testing of the neural network. 


In the folder ''Machine learning'' you can find the machine learning algorithms applied on in-silico data and experimental data. 


In the folder ''Trained_model'' you can find a jupiter notebook where the best trained neural network we manages to achieve is tested with data that the algorithm hasn't been trained on. The files needed to reproduce the plots can be found in the file ''Requirements''.  


The file ''Trained_NN_MAE_088.h5'' contains the best trained neural network we managed to achieve. 


The file ''training_dataset_pre_norm.csv'' containes the pre-normalized dataset that was used for the training. Any new data on which we want to predict the surface tension need to be normalized based on the training dataset.


The file ''test_dataset.csv'' is an example of a dataset that the neural network has not been trained on. 


To use the trained neural network we need a dataset that will contain, in the following order:  


- 175 points defining the polar radius, r (mm), at constant polar radius, phi defined as:
phi=np.linspace(-np.pi, np.pi, 175)


** The coordinates of the droplet contours need to be corrected for tilting. **


- centered vertical position of the centre of the droplet (mm)


- voltage (V)


- current (A)



**References:**

[1] Argyri, Smaragda-Maria, Lars Evenäs, and Romain Bordes. "Contact-free measurement of surface tension on single droplet using machine learning and acoustic levitation." Journal of Colloid and Interface Science 640 (2023): 637-646.

[2] Argyri, Smaragda-Maria, Carl Andersson, Nicolas Paillet, Lars Evenäs, Jens Ahrens, Asier Marzo, Víctor Contreras, and Romain Bordes. "Customized and high-performing acoustic levitators for contact-free experiments." Journal of Science: Advanced Materials and Devices 9, no. 3 (2024): 100720.

[3] Marzo, Asier, Adrian Barnes, and Bruce W. Drinkwater. "TinyLev: A multi-emitter single-axis acoustic levitator." Review of Scientific Instruments 88.8 (2017): 085105.

[4] P. L. Marston, S. E. Lo Porto Arione, and G. L. Pullen, J. Acoust. Soc. Am. 69, 1499-1501 (1981).


[5] P.L . Marston,J . AcoustS. ac.A m. 67, 15-26 (1980).


[6] Trinh, Eugene H., and Chaur‐Jian Hsu. "Equilibrium shapes of acoustically levitated drops." The Journal of the Acoustical Society of America 79.5 (1986): 1335-1338.


------------------------------------------------------------------------------------------------------------------------------------------------------------------
The project from Ref. [1] was funded by the Swedish Research Council (VR) (Public, Sweden) and the Swedish Foundation for Strategic Research (SSF) (Non Profit, Sweden).

# On the Optimal Performance of Distributed Cell-Free Massive MIMO with LoS Propagation

This is a code package related to the following scientific work:

Noor Ul Ain, Lorenzo Miretti, Slawomir Stanczak, “On the Optimal Performance of Distributed Cell-Free Massive MIMO with LoS Propagation,” IEEE Wireless Communications and Networking Conference (WCNC), March 2025.

The package contains a simulation environment that reproduces the numerical results in the article.

## Abstract of the article
In this study, we revisit the performance analysis of distributed beamforming architectures in dense user-centric cell-free massive multiple-input multiple-output (mMIMO) systems in line-of-sight (LoS) scenarios. By incorporating a recently developed optimal distributed beamforming technique, called the team minimum mean square error (TMMSE) technique, we depart from previous studies that rely on suboptimal distributed beamforming approaches for LoS scenarios. Supported by extensive numerical simulations that follow 3GPP guidelines, we show that such suboptimal approaches may often lead to significant underestimation of the capabilities of distributed architectures, particularly in the presence of strong LoS paths. Considering the anticipated ultra-dense nature of cell-free mMIMO networks and the consequential high likelihood of strong LoS paths, our findings reveal that the team MMSE technique may significantly contribute in narrowing the performance gap between centralized and distributed architectures.

## Packages required:
  - numpy
  - scipy
  - matplotlib

## Content of Code Package
* all_functions.py contains the code of all the building blocks required for the results.
* results.ipynb has the complete pipeline to plot cdfs

**Note** 
-The simulation parameters used in the notebook are for a very small network setup (runnable on any laptop).
-Parameters used in the work are also obtainable by using a system(on server maybe) where 100Gb memory can be reserved for computational space.
-Dummy graphs that have dependency on density and power might show some deviations with the graphs presented in the paper. (specifically fig1b and fig2b. Exact results are reproduceable for large setup)

**Tips**
In beamforming functions, when calculating beamformers, a parameter 'batch' is used.
batch=200  # batch means computations will be performed in batchs of 200 realizations at a time. batch =1 means code will loop over 1 realization at a time, batch=realizations means all realizations at a time. 
*** Mathematically everything stays correct, its tradeoff between computation time and computation space depending upon system.


**Critical Points**
Parameter 'realizations' is critical. Realizations should be a large value if large number APs and UEs are used. In thesis, realizations =10000 for AP=100 and UE=40. In notebook, its set to 200, because small number of AP and UEs is used.

Warning: The name of some variables may differ from the notation adopted in the paper.

## License and Referencing
This code package is licensed under the GPLv2 license. If you in any way use this code for research that results in publications, please cite our original article listed above.

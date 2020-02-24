# Training

There are multiple folders indicating the data from different parks.
For example, ***Gonarezhou_datasets*** is the folder for Gonarezhou park.
In order to start, we need to get three files for each park and each resolution.
They are respectively ***allStaticFeat.csv***, ***All_X.csv***, and ***All_Y.csv***.
Thsee files should be put inside the ***Gonarezhou_datasets*** folder with the following structure:

    ├── ...
    ├── resolution                                   # Folder for different resolution
    │   ├── 200m                                     
    │   ├── 500m                                     
    │   └── 1000m                                    # Choose the corresponding resolution to store the data
    │       ├── input                                # Input folder
    │       │   ├── allStaticFeat.csv                # Static features used for predicting
    │       │   ├── All_X.csv                        # Features used for training
    │       │   └── All_Y.csv                        # Labels used for training
    │       └── output                               # Output folder
    │           ├── Maps                         
    │           │   └── Several Maps                 # Maps produced to visualize static features
    │           ├── Final.xlsx                       # The result for learning: AUC, F1, Recall, Precision...
    │           └── ...                              # Other stuffs
    └── ...

After putting the input inside the folder, you can run the following (e.g. Gonarezhou 1000m) to visualize the features

    python Static_Feature_Maps.py -r 1000 -p Gonarezhou

and run the following to train the model

    python Bagging_New_Cross_Blackbox.py -r 1000 -p Gonarezhou

There are also scripts for these two respectively in ***script.sh*** and ***static_script.sh***.
You can also use them.

Please follow the naming and the labels in the csv files.
If there is any error, one very common error is the fist label of ***allStaticFeat.csv***.
It should be ***var1*** but it sometimes is ***ID***.

--------------------------------------------------------

# Planning
Remember to finish the training part first.
The planning part will use the result from the training folder.

For the planning part, please go to the following directory:

    planning/PathPlanning/src/


You can directly run (e.g. Gonarezhou 1000m with 5 patrollers and each patroller can walk at most 20 grids per task.)

    python SinglePlanner.py -r 1000 -p Gonarezhou_human -T 20  -d 5

There is also a file ***script.sh***.
You can modify it and run it.

After running it, you can find the result inside ***kai_data***, which should be organized.
The risk map is also inside the ***Gonarezhou_exp/1000m/***.

# One-4-All

One-4-All (O4A) implementation for image-goal navigation in the Habitat simulator and some toy maze problems. See
the paper [here](https://arxiv.org/abs/2303.04011).

# Installation
## One-4-All

Download the repository with the submodules by running:

```
git clone --recurse-submodules git@github.com:MikeS96/plan2vec.git
```

To install the project run the following commands within the project folder.

```
# Install Venv
pip3 install --upgrade pip
pip3 install --user virtualenv

# Create Env
cd plan2vec
python3 -m venv venv
# Activate env
source venv/bin/activate
```

Then install the project dependencies as follows:

```
pip3 install -r requirements.txt
pip3 install geomloss
cd mazelab
pip3 install -e .
export PYTHONPATH=<path_to>/plan2vec/:$PYTHONPATH
```

## Habitat-sim
We'll install habitat-sim in the same venv. First make sure you have ```cmake>=3.10``` by checking ```cmake --version```.


Now let's install Habitat-Sim following the instructions [here](https://github.com/facebookresearch/habitat-sim/blob/main/BUILD_FROM_SOURCE.md):
```
git clone --branch stable https://github.com/facebookresearch/habitat-sim.git
cd habitat-sim
pip install -r requirements.txt  # Make sure you're still in the venv installed above

# Last argument may be facultative, but was required on Ubuntu 20.04. See this [issue](https://github.com/facebookresearch/habitat-sim/issues/658)
# You may need to install cuda with "sudo apt install nvidia-cuda-toolkit"
python setup.py --bullet --with-cuda build_ext --parallel 8 install --cmake-args="-DUSE_SYSTEM_ASSIMP=ON" # Specify parallel to avoid running out of memory

```
Now's a good time to test if the [test interactive scene](https://github.com/facebookresearch/habitat-sim#testing) works.

If you have issues with a non-nvidia GPU being used for rendering, try ``sudo prime-select nvidia`` then reboot. Make sure nvidia stuff is being used with
``glxinfo | grep OpenGL``.

## Habitat-lab
Install with:
```bash
git clone --branch stable https://github.com/facebookresearch/habitat-lab.git
cd habitat-lab
pip install -r requirements.txt
python setup.py develop --all # install habitat and habitat_baselines
```
Make sure to download some test scenes :
```bash
python -m habitat_sim.utils.datasets_download --uids habitat_test_scenes --data-path data/
python -m habitat_sim.utils.datasets_download --uids habitat_test_pointnav_dataset --data-path data/
```
and finally generate additional pointnav episodes. These can be customized by tweaking the script :
```bash
python script/create_pointnav_dataset.py
```

# One-4-All
In this section, we will go over how to generate data, train O4A and navigate the model in Habitat. You may want
to skip training and download trained checkpoints from [here](https://drive.google.com/file/d/1DUFptg6R2xUbTtzIRpIyIJnHECFvIOr5/view?usp=sharing).
## Generating data
We'll use the Habitat simulator for data generation and navigation. You first need to download the relevant scenes 
(at least Annawan) from the official [Gibson repositiory](https://github.com/StanfordVL/GibsonEnv/blob/master/gibson/data/README.md#download-gibson-database-of-spaces). 
Make sure to put the relevant ```.glb``` and ```.navmesh``` files under the 
```data_habitat/versioned_data/habitat_test_scenes_1.0``` directory. 

You should now be able to generate the pointnav episodes (we will use them for imagenav).
```bash
python script/create_pointnav_dataset.py
```
We are now ready to run data collection. The following command will run the data collection procedure for
Annawan, following the specifications of the paper 
```bash
python3 run_habitat.py env=habitat sim_env=Annawan policy=habitat_shortest difficulty=data_collection collect_data=true test_params.min_steps=499 test_params.max_steps=500 test_params.n_trajectories=60
```

## Training

Train your components is easy giving the configs we provided within this repository. If you want to train the models in
your own dataset you need to train them as follows.

### Local Backbone and Connectivity Head

As specified in the paper, local backbone and connectivity head are trained jointly. An example is given below where 
both models are trained for five epochs using a convolutional predictor using (by default) the Annawan environment:

```bash
python3 train.py env=habitat experiment=train_local_habitat model.net.predictor=conv1d epochs=5
```

### Forward Dynamics

Once local backbone and connectivity head is trained, forward dynamics can be trained with the next command:

```bash
python3 train.py experiment=train_fd_habitat env=habitat epochs=5
```

Note that you may need to make sure the path `checkpoints.backbone_path` is properly pointing to the checkpoint of your 
previously trained *Local Backbone*.

### Geodesic Regressor

Finally, the geodesic regressor is trained by running:

```bash
python3 train.py experiment=train_geodesic_habitat env=habitat epochs=5 datamodule.drop_last=false
```

Where the `datamodule.drop_last=false` ensures we work with all the states in our dataset.

For further information on the parameters available for training and the structure of the project, refer to `conf/config.yaml`

## Navigation
For navigation, you need to organize your trained checkpoints (or [download them](https://drive.google.com/file/d/1DUFptg6R2xUbTtzIRpIyIJnHECFvIOr5/view?usp=sharing)) in
the following way
```bash
components
└── habitat
    ├── backbone.ckpt
    ├── fd.ckpt
    └── geodesic_regressors
        ├── annawan.ckpt
        ...
```
or alternatively change the paths in ```conf/env/habitat.yaml```.

You can then benchmark the O4A agent for 10 trajectories in the Annawan environment by running
```bash
python3 run_habitat.py policy=habitat_o4a env=habitat sim_env=Annawan difficulty=hard test_params.n_trajectories=10
```

You should obtain something close to the following metrics
```bash
[2023-03-17 16:52:03,957][src.run_habitat][INFO] - Difficulty : hard.
[2023-03-17 16:52:03,958][src.run_habitat][INFO] - Success      : 0.9000
[2023-03-17 16:52:03,958][src.run_habitat][INFO] - Soft Success : 1.0000
[2023-03-17 16:52:03,958][src.run_habitat][INFO] - SPL          : 0.6545
[2023-03-17 16:52:03,958][src.run_habitat][INFO] - Soft SPL     : 0.6140
[2023-03-17 16:52:03,958][src.run_habitat][INFO] - DTG          : 0.9436
[2023-03-17 16:52:03,958][src.run_habitat][INFO] - Collisions   : 0.0000
[2023-03-17 16:52:03,958][src.run_habitat][INFO] - CFT          : 1.0000
[2023-03-17 16:52:03,958][src.run_habitat][INFO] - Failed Traj. : []
```

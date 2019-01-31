# torcs_rl

## Installation

### vtorcs
Vtorcs is used to exchange state and action with the game engine and has to be build to use torcs_rl.

```console
sudo apt-get install libglib2.0-dev  libgl1-mesa-dev libglu1-mesa-dev  freeglut3-dev  libplib-dev  libopenal-dev libalut-dev libxi-dev libxmu-dev libxrender-dev  libxrandr-dev libpng-dev torcs
cd vtorcs-RL-color
./configure
make
sudo make install
sudo make datainstall
```

### torcs_rl
Torcs_rl has the following dependencies which have to be installed prior to usage

* python 3.7
* keras 
* tensorflow 
* matplotlib 
* seaborn 
* gym
* xautomation

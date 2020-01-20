# Contribute 
Currently using AWS Deep Racer environment to test with the Reinforcement learning.
Lately with move on to Under Water Simulation

# Deep Racer

This Sample Application runs a simulation which trains a reinforcement learning (RL) model to drive a car around a track.

_AWS RoboMaker sample applications include third-party software licensed under open-source licenses and is provided for demonstration purposes only. Incorporation or use of RoboMaker sample applications in connection with your production workloads or a commercial products or devices may affect your legal rights or obligations under the applicable open-source licenses. Source code information can be found [here](https://s3.console.aws.amazon.com/s3/buckets/robomaker-applications-us-east-1-72fc243f9355/deep-racer/?region=us-east-1)._

Keywords: Reinforcement learning, AWS, RoboMaker

![deepracer-hard-track-world.jpg](docs/images/deepracer-hard-track-world.jpg)

## Requirements
### IMPORTANT

- ROS Kinetic / Melodic (optional) - To run the simulation locally. Other distributions of ROS may work, however they have not been tested
- Gazebo (optional) - To run the simulation locally
- OpenAI GYM
- Tensorflow 2.0 (GPU)

### tested
- ros-kinetic
- gazebo9
- ros-kinetic-gazebo9-ros-pkgs 
- python3
- tensorflow 2.0 (gpu)



### AWS Credentials
You will need to create an AWS Account and configure the credentials to be able to communicate with AWS services. You may find [AWS Configuration and Credential Files](https://docs.aws.amazon.com/cli/latest/userguide/cli-config-files.html) helpful.



## Usage

### Training the model

#### Building the simulation bundle

```bash
cd simulation_ws
rosws update
rosdep install --from-paths src --ignore-src -r -y
colcon build
```

#### Running the simulation


The following environment variables must be set when you run your simulation:

- `MARKOV_PRESET_FILE` - Defines the hyperparameters of the reinforcement learning algorithm. This should be set to `deepracer.py`.
- `WORLD_NAME` - The track to train the model on. Can be one of easy_track, medium_track, or hard_track.


Once the environment variables are set, you can run local training using the roslaunch command

```bash
source simulation_ws/install/setup.sh
roslaunch deepracer_simulation local_training.launch
python3 simulation_ws/sagemake_rl_agent/markov/local_machine_training.py
```

#### Seeing your robot learn


### Evaluating the model

#### Building the simulation bundle

You can reuse the bundle from the training phase again in the simulation phase.

#### Evaluate the modle

The evaluation phase requires that the same environment variables be set as in the training phase. Once the environment variables are set, you can run
evaluation using the roslaunch command

- Work on progress

```bash
source simulation_ws/install/setup.sh
roslaunch deepracer_simulation evaluation.launch
```

### Troubleshooting

###### The robot does not look like it is training

The training algorithm has two phases. The first is when the reinforcement learning model is used to make the car move in the track, 
while the second is when the algorithm uses the information gathered in the first phase to improve the model. In the second
phase, no new commands are sent to the car, meaning it will appear as if it is stopped, spinning in circles, or drifting off
aimlessly.

## Using this sample with AWS RoboMaker

You first need to install colcon. Python 3.5 or above is required.
TensorFlow 2 packages require a pip version >19.0.
```bash
apt-get update
apt-get install -y python3-pip python3-apt
pip3 install -U pip --user
pip3 install colcon-ros-bundle tensorflow gym numpy --user
```

After colcon is installed you need to build your robot or simulation, then you can bundle with:

```bash
# Bundling Simulation Application
cd simulation_ws
colcon build
```

This produces `simulation_ws/bundle/output.tar`.
You'll need to upload this artifact to an S3 bucket. You can then use the bundle to
[create a simulation application](https://docs.aws.amazon.com/robomaker/latest/dg/create-simulation-application.html),
and [create a simulation job](https://docs.aws.amazon.com/robomaker/latest/dg/create-simulation-job.html) in AWS RoboMaker.

## License

Most of this code is licensed under the MIT-0 no-attribution license. However, the sagemaker_rl_agent package is
licensed under Apache 2. See LICENSE.txt for further information.

## How to Contribute

Create issues and pull requests against this Repository on Github

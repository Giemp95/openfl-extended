# OpenFL + Gramine
This manual will help you run OpenFL with Aggregator-based workflow inside SGX enclave with Gramine.

## Prerequisites

This guide will take into account three different types of machines:
- the building machine, where the docker image will be built;
- the Aggregator machine, where the Aggregator will run;
- the Collaborator machines, where the Collaborators will run.

It is, by the way, not mandatory that these machines are physically different. A single machine can be used to carry out multiple functions; this choice is made only to make the following tutorial clearer.
</br>
In the following the requirements for the different machines will be presented.

Building machine:
- OpenFL
- Docker should be installed, user included in Docker group

Machines that will run an Aggregator and Collaborator containers should have the following:
- OpenFL (If used for certification through built-in tooling)
- FSGSBASE feature and SGX enabled from BIOS
- Intel SGX driver or Linux 5.11+ driver
- Intel SGX SDK/PSW
</br>
This is a short list, see more in [Gramine docs](https://gramine.readthedocs.io/en/latest/devel/building.html).
The use of a Python Virtual Environment (venv) is strongly encouraged.

## Workflow
The user will mainly interact with OpenFL CLI, docker CLI, and other command-line tools. But the user is also expected to modify plan.yaml file and Python code under workspace/src folder to set up an FL Experiment.
### On the building machine (Data Scientist's node):
1. As usual, **create a workspace**: 
```
export WORKSPACE_NAME=my_sgx_federation_workspace
export TEMPLATE_NAME=torch_cnn_histology

fx workspace create --prefix $WORKSPACE_NAME --template $TEMPLATE_NAME
cd $WORKSPACE_NAME
```
Be sure that the Python versions used on the building node and inside the Docker container both support the requirement.txt files of your selected workspace.
If that is not the case, proceed to manually modify your selected workspace requirement.txt file in the openfl-workspace installation folder to obtain a congruous environment.
Also, if you are planning to use the SGX support, be sure that that requirements.txt file does not contain any reference to CUDA distributions.

Modify the code and the plan.yaml, set up your training procedure. </br>
Pay attention to the following: 
- make sure the data loading code reads data from ./data folder inside the workspace
- if you download data (development scenario) make sure your code first checks if data exists, as connecting to the internet from an enclave may be problematic.
- make sure you do not use any CUDA driver-dependent packages. If you are using the ready-made OpenFL examples double-check that the `requirements.txt` suits these needs. This could require to modify the `requirements.txt` file located under the installed openfl/openfl-workspace example-specific folder on the building node.

Default workspaces (templates) in OpenFL differ in their data downloading procedures. Workspaces with data loading flow that do not require changes to run with Gramine include:
- torch_unet_kvasir
- torch_cnn_histology
- keras_nlp

Also the other workspaces can be used by taking care of placing the dataset used under a data/ folder and modifying the requirements.txt file so that it does not contain references to CUDA distributions.

2. **Initialize the experiment plan** </br> 
Find out the FQDN of the **Aggregator machine** and use it for plan initialization.
For example, on Unix-like OS try the following command:
```
hostname --all-fqdns | awk '{print $1}'
```
(In case this FQDN does not work for your federation, try putting the machine IP instead)
Then, on the building node, pass the result as `AGG_FQDN` parameter to:
```
fx plan initialize -a $AGG_FQDN
```

3. (Optional) **Generate a signing key** on the building machine if you do not have one.</br>
It will be used to calculate hashes of trusted files. If you plan to test the application without SGX (gramine-direct) you also do not need a signer key.
```
export KEY_LOCATION=.

openssl genrsa -3 -out $KEY_LOCATION/key.pem 3072
```
This key will not be packed into the final Docker image.

4. **Build the Experiment Docker image**

Before building the image, according to your SGX-capable processor, it could be necessary to turn off the logging TensorBoardX logging functionality from the plan.
This can be done by simply setting `write_logs: false` in the plan/plan.yaml file.

```
fx workspace graminize -s $KEY_LOCATION/key.pem
```
This command will build and save a Docker image with your Experiment. The saved image will contain all the required files to start a process in an enclave.</br>
If a signing key is not provided, the application will be built without SGX support, but it still can be started with gramine-direct executable.


### Image distribution:
Data scientist (builder) now must transfer the Docker image to the aggregator and collaborator machines. The Aggregator will also need initial model weights.

5. **Transfer files** to the aggregator and collaborator machines.
If there is a connection between machines, you may use `scp`. In other cases use the transfer channel that suits your situation.</br>
Send files to the aggregator machine:
```
scp BUILDING_MACHINE:WORKSPACE_PATH/WORKSPACE_NAME.tar.gz AGGREGATOR_MACHINE:SOME_PATH
scp BUILDING_MACHINE:WORKSPACE_PATH/save/TEMPLATE_NAME_init.pbuf AGGREGATOR_MACHINE:SOME_PATH
```

Send the image archive to collaborator machines:
```
scp BUILDING_MACHINE:WORKSPACE_PATH/WORKSPACE_NAME.tar.gz COLLABORATOR_MACHINE:SOME_PATH
```

Please, keep in mind, if you run a test Federation, with data downloaded from the internet, you should also transfer/download data to collaborator machines.

### On the running machines (Aggregator and Collaborator nodes):
6. **Load the image.**
Execute the following command on all running machines:
```
docker load < WORKSPACE_NAME.tar.gz
```

7. **Prepare certificates**
Certificates exchange is a big separate topic. To run an experiment following OpenFL Aggregator-based workflow, a user must follow the established procedure, please refer to [the docs](https://openfl.readthedocs.io/en/latest/running_the_federation.html#bare-metal-approach) (only Section 2 without the workspace import/export steps). 
Before starting to create the certificates, create an empty plan/cols.yaml file on the Aggregator node and an empty plan/data.yaml file on each Colleborator node.
Please double-check that the `cols.yaml` and `data.yaml` files contain only the names of the desired collaborators after the certification procedure.

We recommend replicating the OpenFL workspace folder structure on all the machines and following the usual certifying procedure. Finally, on the aggregator node you should have the following folder structure:
```
workspace/
--save/TEMPLATE_NAME_init.pbuf
--logs/
--plan/cols.yaml
--cert/
  --client/*col.crt
  --server/
    --agg_FQDN.crt
    --agg_FQDN.key
```

On collaborator nodes:
```
workspace/
--data/*dataset*
--plan/data.yaml
--cert/
  --client/
    --col_name.crt
    --col_name.key
```

To speed up the certification process for one-node test runs, it makes sense to utilize the OpenFL [integration test script](https://github.com/intel/openfl/blob/develop/tests/github/test_graminize.sh), that will create required folders and certify an aggregator and two collaborators.

### **Run the Federation in enclaves**
#### On the Aggregator machine run:
```
export WORKSPACE_NAME=your_workspace_name
export WORKSPACE_PATH=path_to_workspace
docker run -it --rm --device=/dev/sgx_enclave --volume=/var/run/aesmd/aesm.socket:/var/run/aesmd/aesm.socket \
--network=host \
--volume=${WORKSPACE_PATH}/cert:/workspace/cert \
--volume=${WORKSPACE_PATH}/logs:/workspace/logs \
--volume=${WORKSPACE_PATH}/plan/cols.yaml:/workspace/plan/cols.yaml \
--mount type=bind,src=${WORKSPACE_PATH}/save,dst=/workspace/save,readonly=0 \
${WORKSPACE_NAME} aggregator start
```

#### On the Collaborator machines run:
```
export WORKSPACE_NAME=your_workspace_name
export WORKSPACE_PATH=path_to_workspace
export COL_NAME=col_name
docker run -it --rm --device=/dev/sgx_enclave --volume=/var/run/aesmd/aesm.socket:/var/run/aesmd/aesm.socket \
--network=host \
--volume=${WORKSPACE_PATH}/cert:/workspace/cert \
--volume=${WORKSPACE_PATH}/plan/data.yaml:/workspace/plan/data.yaml \
--volume=${WORKSPACE_PATH}/data:/workspace/data \
${WORKSPACE_NAME} collaborator start -n ${COL_NAME}
```

In case you want to modify the running code, you need to go back to the building node, fix your code, re-build the Docker image through the graminize command and then sending the new image to the Aggregator and Collaborators nodes.
At this point you have to re-run the docker load command on the Aggregator and Collaborators machines; now you should be able to re-run successfully your experiment.

### **No SGX run (`gramine-direct`)**:
The user may run an experiment under gramine without SGX. Note how we do not mount `sgx_enclave` device and pass a `--security-opt` instead that allows syscalls required by `gramine-direct`

#### On the Aggregator machine run:
```
export WORKSPACE_NAME=your_workspace_name
export WORKSPACE_PATH=path_to_workspace
docker run -it --rm --security-opt seccomp=unconfined -e GRAMINE_EXECUTABLE=gramine-direct \
--network=host \
--volume=${WORKSPACE_PATH}/cert:/workspace/cert \
--volume=${WORKSPACE_PATH}/logs:/workspace/logs \
--volume=${WORKSPACE_PATH}/plan/cols.yaml:/workspace/plan/cols.yaml \
--mount type=bind,src=${WORKSPACE_PATH}/save,dst=/workspace/save,readonly=0 \
${WORKSPACE_NAME} aggregator start
```

#### On the Collaborator machines run:

```
export WORKSPACE_NAME=your_workspace_name
export WORKSPACE_PATH=path_to_workspace
export COL_NAME=col_name
docker run -it --rm --security-opt seccomp=unconfined -e GRAMINE_EXECUTABLE=gramine-direct \
--network=host \
--volume=${WORKSPACE_PATH}/cert:/workspace/cert \
--volume=${WORKSPACE_PATH}/plan/data.yaml:/workspace/plan/data.yaml \
--volume=${WORKSPACE_PATH}/data:/workspace/data \
${WORKSPACE_NAME} collaborator start -n ${COL_NAME}
```

## The Routine
Gramine+OpenFL PR brings in `openfl-gramine` folder, that contains the following files:
 - MANUAL.md - this manual
 - Dockerfile.gamine - the base image Dockerfile for all experiments, it starts from Python3.8 image and installs OpenFL and Gramine packages.
 - Dockerfile.graminized.workspace - this one is for building the final experiment image. It starts from the previous image and imports the experiment archive (executes 'fx workspace import') inside an image. At this stage, we have an experiment workspace and all the requirements installed inside the image. Then it runs a unified Makefile that uses the openfl.manifest.template to prepare required files to run OpenFL under gramine inside an SGX enclave.
 - Makefile - follows regular gramine app building workflow, please refer to [gramine examples](https://github.com/gramineproject/examples) for details
 - openfl.manifest.template - universal FL experiment [gramine manifest template](https://gramine.readthedocs.io/en/latest/manifest-syntax.html)
 - start_process.sh - bash script used to start an OpenFL actor in a container.

There is a files access peculiarity that should be kept in mind during debugging and development.
Both Dockerfiles are read from the bare-metal OpenFL installation, i.e. from an OpenFL package on a building machine.
While the gramine manifest template and the Makefile are read in image build time from the local (in-image) OpenFL package. </br>
Thus, if one wants to make changes to the gramine manifest template or the Makefile, they should change the OpenFL installation procedure in Dockerfile.gramine, so their changes may be pulled to the base image. One option is to push the changes to a GitHub fork and install OpenFL from this fork. 
```
*Dockerfile.gramine:*

RUN git clone https://github.com/your-username/openfl.git --branch some_branch
WORKDIR /openfl
RUN --mount=type=cache,target=/root/.cache/ \
    pip install --upgrade pip && \
    pip install .
WORKDIR /
```
In this case, to rebuild the image, use `fx workspace dockerize --rebuild` with `--rebuild` flag that will pass '--no-cache' to docker build command.

Another option is to copy OpenFL source files from an on-disk cloned repo, but it would mean that the user must build the graminized image from the repo directory using Docker CLI.


## Known issues:
- Kvasir experiment: aggregation takes really long, debug log-level does not show the reason
- We need workspace zip to import it and create certs. We need to know the number of collaborators prior to zipping the workspace. SOLUTION: mount cols.yaml and data.yaml
- During plan initialization we need data to initialize the model. so at least one collaborator should be in data.yaml and its data should be available. cols.yaml may be empty at first
During cert sign request generation cols.yaml on collaborators remain empty, data.yaml is extended if needed. On aggregator, cols.yaml are updated during signing procedure, data.yaml remains unmodified
- `error: Disallowing access to file '/usr/local/lib/python3.8/__pycache__/signal.cpython-38.pyc.3423950304'; file is not protected, trusted or allowed.`
- The TensorBoardX logging functionality is troublesome; it is better to deactivate the log functionality in the plan.
- Depending on the Python version inside the Docker container, it could be necessary to manually change the requirement.txt file present in the chosen workspace installation folder.
- Different OpenFL version used inside and outside the Docker container could lead into issues when running the experiments.

 ## TO-DO:
- [X] import manifest and makefile from OpenFL dist-package 
- [X] pass wheel repository to pip (for CPU versions of PyTorch for example)
- [ ] get rid of command line args (insecure)
- [ ] introduce `fx workspace create --prefix WORKSPACE_NAME` command without --template option to the OpenFL CLI, which will create just an empty workspace with the right folder structure.
- [ ] introduce `fx *actor* start --from image`
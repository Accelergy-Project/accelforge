# Lab 0: Infrastructure Setup

This lab tests out your infrastructure setup for the course. If you're seeing
this README from in a Docker container, you're almost there! Open the Jupyter
notebook and complete the lab, then submit the lab using the instructions below.

## Installing Docker

First, we need to setup [Docker](https://docs.docker.com/get-started/) to
run the code necessary for the lab. We provide you two sets of instructions for 
installing Docker. You can decide which app to install according to your operating
system.

- Mac:

  - If you have macOS 10.13.0 High Sierra or higher, please follow instructions in `docker-desktop-readme.md`
  - otherwise, follow instructions in `docker-toolbox-readme.md`

- Windows:

  - If you have Windows 10 Pro with enterprise/education,
    please follow instructions in `docker-desktop-readme.md`
      - NOTE: Most of you should not have this OS, and have windows 10 Home instead. 
      Please check carefully!
  - Otherwise, follow instructions in `docker-toolbox-readme.md`

- Windows Subsystem for Linux 2 (WSL2):

  - If you have Windows 10 with version 1903 or higher, you can install [WSL2](https://docs.microsoft.com/en-us/windows/wsl/install-win10)
  - You can then install Docker Desktop with WSL2, and use Linux workspaces. Follow instructions [here](https://docs.docker.com/docker-for-windows/wsl/)

- Linux:

  - If you have Linux distributions, such as Ubuntu, please check if Docker
    supports your distribution and architecture
    [here](https://docs.docker.com/engine/install/). Examples of supported
    distributions are: 
    - [Ubuntu](https://docs.docker.com/engine/install/ubuntu/)
    - [CentOS](https://docs.docker.com/engine/install/centos/)
    - [Debian](https://docs.docker.com/engine/install/debian/)
    - [Fedora](https://docs.docker.com/engine/install/fedora/)

## Git Basics

Since we will be using Git to distribute the labs, you should also install `git`.
If you are using Linux distributions or WSL2, `git` should either be
already installed or available in your relevant package manager. On MacOSX, the
simplest way to install `git` is to try running `git` in your terminal, which will
prompt you to install Xcode Command Line Tools if they are not already
installed. On Windows (without WSL2), you can download [Github
Desktop](https://desktop.github.com/). Below are basic git command examples:

[//]:
Clone this repository (replace `username` with your GitHub username)
```
git clone <your-repository-url> # Find this repository URL from the GitHub page
```

*In case you are using WSL2, note that it is recommended to clone this repository to Linux filesystems.*

*In case you are using Windows, note that you should clone this repository to somewhere in C drive because Docker doesn't play well with other drives.*

NOTE: When cloning, you can use the URL in your search bar for your repository, 
but you will need to  enter your GitHub login information whenever you use wish to 
sync your local code  with the cloud and to submit. You can avoid this in all 
future labs by adding an SSH key to your GitHub account following the instructions 
[here](https://docs.github.com/en/authentication/connecting-to-github-with-ssh/adding-a-new-ssh-key-to-your-github-account)
and then using the SSH url under the "Code" dropdown on the repository page. 
[This](https://education.github.com/git-cheat-sheet-education.pdf)  is also a 
handy Git cheat sheet for those not familiar.

## Starting Docker

Now, we will start Docker and launch the Jupyter notebook. Follow the instructions
in `docker-compose.yaml` file to change the `USER_UID` and set the environment
variable `DOCKER_ARCH`. You should then run `docker compose up` in a terminal from the 
directory containing the `docker-compose.yaml` file. You will now see a Jupyter server
launch and start running. Finally, open the Internet browser of your choice, and access the 
Jupyter server (default address will be `localhost:8888`, but may differ based on 
your Docker configuration). You have now opened the Jupyter notebook and may begin
the rest of the assignment. An example of the terminal commands you may execute 
to get to this point are provided below:

```
cd <your-repository-directory>
export DOCKER_ARCH=amd64

# If you are using arm CPU (Apple M1/M2), 
# export DOCKER_ARCH=arm64 

docker compose pull
docker compose up
```

Then, complete the lab and run the `make submit` with your local terminal at the root
of the repository.

If your docker configuration changes the default IP address, then use that
address when accessing the Jupyter server. If you want to avoid setting
`DOCKER_ARCH` everytime, you can permanently add `DOCKER_ARCH=<your
architecture>` into your `.bashrc` or `.zshrc`.

## Submission
After finishing the lab, please run `make submit` in the root directory (NOT in
the Docker container. Outside the workspace directory). of your repository to
submit your code. Check your submission on the GitHub website and ensure that
all notebooks have all cells run and all outputs visible. Additionally, ensure
that the `answers.yaml` file in the website matches the answers you have in your
notebooks. If either the notebooks or the `answers.yaml` file are not up to
date, you may lose points or receive a zero for the assignment.

FAILURE TO FOLLOW THESE INSTRUCTIONS WILL RESULT IN YOU RECEIVING A ZERO FOR THE
ASSIGNMENT.
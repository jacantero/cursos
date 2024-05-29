FROM nvcr.io/nvidia/pytorch:22.12-py3
ARG USER_ID
ARG USER_UID
LABEL mantainer=$USER_ID
USER root

# Install required tools.
RUN apt-get -qq update && apt-get -qq upgrade \
    && apt-get -qq --no-install-recommends install sudo \
    && apt-get -qq --no-install-recommends install python3-pip \
    && apt-get -qq --no-install-recommends install openssh-server \
    && apt-get -qq --no-install-recommends install libgl1-mesa-glx \
    && apt-get -qq clean    \
    && rm -rf /var/lib/apt/lists/*

# Update pip
RUN /usr/bin/python -m pip install --upgrade pip

# Install virtualenvwrapper as root
RUN pip install virtualenvwrapper

# Configure SSHD.
# SSH login fix. Otherwise user is kicked off after login
RUN sed 's@session\s*required\s*pam_loginuid.so@session optional pam_loginuid.so@g' -i /etc/pam.d/sshd
RUN mkdir /var/run/sshd
RUN bash -c 'install -m755 <(printf "#!/bin/sh\nexit 0") /usr/sbin/policy-rc.d'
RUN ex +'%s/^#\zeListenAddress/\1/g' -scwq /etc/ssh/sshd_config
RUN ex +'%s/^#\zeHostKey .*ssh_host_.*_key/\1/g' -scwq /etc/ssh/sshd_config
RUN RUNLEVEL=1 dpkg-reconfigure openssh-server
RUN ssh-keygen -A -v
RUN update-rc.d ssh defaults

# Configure sudo.
RUN ex +"%s/^%sudo.*$/%sudo ALL=(ALL:ALL) NOPASSWD:ALL/g" -scwq! /etc/sudoers

# Setup the default user.
RUN useradd -rm -d /home/$USER_ID -s /bin/bash -u $USER_UID -g root -G sudo $USER_ID
RUN echo "$USER_ID:$USER_ID" | chpasswd
USER $USER_ID
WORKDIR /home/$USER_ID

# Generate and configure user keys.
#USER $USER_ID
RUN ssh-keygen -t ed25519 -f ~/.ssh/id_ed25519
# To include an SSH key in the list of authorized keys of the docker image
#COPY --chown=$USER_ID:root "./files/authorized_keys" /home/$USER_ID/.ssh/authorized_keys

# virtualenv and virtualenvwrapper
RUN echo 'export WORKON_HOME=$HOME/.virtualenvs' >> ~/.bashrc
RUN echo 'export VIRTUALENVWRAPPER_PYTHON=/usr/bin/python3' >> ~/.bashrc
RUN echo 'source /usr/local/bin/virtualenvwrapper.sh' >> ~/.bashrc

# fix for 'libcusolver.so.10 not found'
RUN sudo ln -sf /usr/local/cuda/lib64/libcusolver.so.11 /usr/local/cuda/lib64/libcusolver.so.10
RUN echo 'export LD_LIBRARY_PATH=/usr/local/cuda/lib64/:$LD_LIBRARY_PATH' >> ~/.bashrc

# create virtualenv and install requirements
ENV VIRTUALENVWRAPPER_PYTHON /usr/bin/python3
RUN /bin/bash -c "source /usr/local/bin/virtualenvwrapper.sh; mkvirtualenv tf2"
ENV VENV /home/$USER_ID/.virtualenvs/tf2
ENV VENV_PIP $VENV/bin/pip

# Install some useful Python packages into the virtualenv
RUN $VENV_PIP install numpy opencv-contrib-python pandas scikit-learn matplotlib tqdm tensorflow-gpu imgaug imutils

# Setup default command and/or parameters.
EXPOSE 22
#CMD ["/usr/bin/sudo", "/usr/sbin/sshd", "-D", "-o", "ListenAddress=0.0.0.0"]
ENTRYPOINT /usr/bin/sudo /usr/sbin/sshd && /bin/bash
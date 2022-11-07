FROM nvidia/cuda:11.3.1-cudnn8-devel-ubuntu20.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHON_VERSION 3.8.6
WORKDIR /root
ENV HOME /root
ENV PYTHON_ROOT $HOME/local/python-$PYTHON_VERSION
ENV PATH $PYTHON_ROOT/bin:$PATH
ENV PYENV_ROOT $HOME/.pyenv
RUN rm /etc/apt/sources.list.d/cuda.list
#RUN rm /etc/apt/sources.list.d/nvidia-ml.list
RUN apt-get update && apt-get upgrade -y \
 && apt-get install -y \
    git \
    make \
    cmake \
    build-essential \
    libssl-dev \
    zlib1g-dev \
    libbz2-dev \
    libreadline-dev \
    libsqlite3-dev \
    wget \
    curl \
    llvm \
    libncurses5-dev \
    libncursesw5-dev \
    xz-utils \
    tk-dev \
    libffi-dev \
    liblzma-dev \
    software-properties-common \
    libpq-dev \
    openssh-server \
    rsyslog \
    tmux \
    libsndfile1 \
    sox \
    ffmpeg \
 && git clone https://github.com/pyenv/pyenv.git $PYENV_ROOT \
 && $PYENV_ROOT/plugins/python-build/install.sh \
 && /usr/local/bin/python-build -v $PYTHON_VERSION $PYTHON_ROOT \
 && rm -rf $PYENV_ROOT

# install common python packages
ADD ./requirements.txt /tmp
RUN pip install pip setuptools -U && pip install -r /tmp/requirements.txt

# setup jp
RUN apt-get update
RUN apt-get install -y language-pack-ja-base language-pack-ja locales
RUN locale-gen ja_JP.UTF-8
ENV LESSCHARSET utf-8

# pip installs
RUN pip install torch==1.12.0+cu113 torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu113
RUN jupyter contrib nbextension install --user
RUN jupyter nbextension enable codefolding/main
RUN git config --global user.email ""
RUN git config --global user.name "tosaka-m"
RUN echo 'alias jupy="jupyter notebook --ip=0.0.0.0 --allow-root"' >> ~/.bashrc

# julius
RUN apt-get install -y zlib1g-dev libsdl2-dev libasound2-dev

# mecab
RUN apt-get install -y mecab libmecab-dev unidic-mecab
RUN pip install mecab-python
ENV MECABRC /etc/mecabrc

# Emacs
RUN apt-get install -y emacs
RUN echo 'alias e="emacs"' >> ~/.bashrc

# SSH
CMD ["service", "rsyslog", "start"]
RUN mkdir ~/.ssh
RUN mkdir /var/run/sshd
RUN sed -i 's/PermitRootLogin prohibit-password/PermitRootLogin yes/' /etc/ssh/sshd_config
RUN sed -i 's/#PasswordAuthentication yes/PasswordAuthentication no/' /etc/ssh/sshd_config
RUN sed 's@session\s*required\s*pam_loginuid.so@session optional pam_loginuid.so@g' -i /etc/pam.d/sshd
ENV NOTVISIBLE "in users profile"
RUN echo "export VISIBLE=now" >> /etc/profile
EXPOSE 22
CMD ["/usr/sbin/sshd", "-D"]
COPY ./ssh_config /root/.ssh/config
COPY ./id_rsa_github_dl /root/.ssh/id_rsa_github_dl
RUN ssh-keyscan github.com >> /root/.ssh/known_hosts
RUN ssh -T github.com; exit 0
RUN git config --global --add safe.directory /root/project

# install private package
# RUN pip install git+ssh://git@github.com/tosaka-m/HifiGAN.git
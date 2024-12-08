FROM stablebaselines/rl-baselines3-zoo

#RUN apt-get update 

#COPY requirements.txt ./
#RUN python -m pip install --upgrade pip

#RUN apt-get update && apt-get install -y python3-pip git 
RUN /bin/bash "pip install panda-gym"

# Default powerline10k theme, no plugins installed
RUN sh -c "$(wget -O- https://github.com/deluan/zsh-in-docker/releases/download/v1.2.1/zsh-in-docker.sh)"
RUN PATH="$PATH:/usr/bin/zsh"
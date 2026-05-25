# syntax=docker/dockerfile:1
ARG PYTHON_VERSION=3.12
ARG NODE_VERSION=v23.3.0
ARG FNM_VERSION=1.37.1
ARG GIT_DELTA_VERSION=0.18.2
ARG ZSH_IN_DOCKER_VERSION=1.2.1
ARG CLAUDE_CODE_VERSION=2.1.111

# -- Base --------------------------------------------------------------------
FROM python:${PYTHON_VERSION}-slim-bullseye AS base

ARG NODE_VERSION
ARG FNM_VERSION
ARG GIT_DELTA_VERSION
ARG ZSH_IN_DOCKER_VERSION
ARG CLAUDE_CODE_VERSION

ENV DEBIAN_FRONTEND=noninteractive \
    DEVCONTAINER=true \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    NODE_OPTIONS="--max-old-space-size=4096" \
    NPM_CONFIG_IGNORE_SCRIPTS=true \
    NPM_CONFIG_AUDIT=true \
    NPM_CONFIG_FUND=false \
    NPM_CONFIG_SAVE_EXACT=true \
    NPM_CONFIG_UPDATE_NOTIFIER=false \
    NPM_CONFIG_MINIMUM_RELEASE_AGE=1440 \
    POWERLEVEL9K_DISABLE_GITSTATUS=true \
    FNM_DIR=/opt/fnm \
    PATH=/opt/fnm:/workspace:/home/vscode/.local/bin:$PATH

# -- Layer 1: System packages (changes rarely — long cache life) --------------
RUN apt-get update && apt-get install -y --no-install-recommends \
      less git procps sudo curl wget unzip gnupg2 \
      zsh fzf man-db vim jq make strace htop \
      iptables ipset iproute2 dnsutils aggregate ssh-client docker.io \
      python3-pip python3-venv libmagic1 libmagic-dev mlocate \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# -- Layer 2: git-delta -------------------------------------------------------
RUN ARCH=$(dpkg --print-architecture) && \
    wget -q "https://github.com/dandavison/delta/releases/download/${GIT_DELTA_VERSION}/git-delta_${GIT_DELTA_VERSION}_${ARCH}.deb" && \
    dpkg -i "git-delta_${GIT_DELTA_VERSION}_${ARCH}.deb" && \
    rm "git-delta_${GIT_DELTA_VERSION}_${ARCH}.deb"

# -- Layer 3: Node via fnm (required for Claude Code) ------------------------
RUN curl -fsSL https://fnm.vercel.app/install | bash -s -- --install-dir "$FNM_DIR" --skip-shell && \
    export PATH="$FNM_DIR:$PATH" && \
    eval "$(fnm env)" && \
    fnm install ${NODE_VERSION} && \
    fnm default ${NODE_VERSION} && \
    fnm exec --using=${NODE_VERSION} node --version

# -- Layer 4: Claude Code (global npm install) --------------------------------
RUN export PATH="$FNM_DIR:$PATH" && eval "$(fnm env)" && \
    npm install -g @anthropic-ai/claude-code@${CLAUDE_CODE_VERSION}

# -- Layer 5: Zsh + Oh My Zsh ------------------------------------------------
RUN sh -c "$(curl -fsSL https://github.com/deluan/zsh-in-docker/releases/download/v${ZSH_IN_DOCKER_VERSION}/zsh-in-docker.sh)" -- \
      -p git -x && \
    echo "alias ll='ls -lrta'" >> /root/.zshrc && \
    echo "alias c=clear"       >> /root/.zshrc && \
    echo "alias ll='ls -lrta'" >> /root/.bashrc && \
    echo "alias c=clear"       >> /root/.bashrc && \
    echo "eval \$(fnm env)"    >> /root/.bashrc && \
    echo "eval \$(fnm env)"    >> /root/.zshrc && \
    echo "PATH=$FNM_DIR:/workspace:/home/vscode/.local/bin:\$PATH" >> /root/.bashrc && \
    echo "PATH=$FNM_DIR:/workspace:/home/vscode/.local/bin:\$PATH" >> /root/.zshrc

# -- Layer 6: vscode user + persistent directories ---------------------------
RUN groupadd --gid 1000 vscode && \
    useradd  --uid 1000 --gid 1000 -m -s /bin/zsh vscode && \
    echo "vscode ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers.d/vscode && \
    mkdir -p /commandhistory /workspace /home/vscode/.claude /opt && \
    touch /commandhistory/.bash_history /commandhistory/.zsh_history && \
    cp /root/.zshrc /home/vscode/.zshrc && \
    cp /root/.bashrc /home/vscode/.bashrc && \
    cp -vR /root/.oh-my-zsh /home/vscode/.oh-my-zsh && \
    chown -R vscode:vscode /commandhistory /workspace /home/vscode && \
    usermod -aG docker vscode && \
    updatedb || true

USER vscode
WORKDIR /workspace

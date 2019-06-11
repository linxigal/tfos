# Python 环境

### 创建用户并授权

1. 创建test用户 `adduser test`
2. 给test用户设置密码`passwd test`（输入两次密码）
3. 给/etc/sudoers文件添加写权限`chmod -v u+w /etc/sudoers`
4. 给test用户授root权限，编辑/etc/sudoers`vi /etc/sudoers`, 添加一行` test ALL=(ALL)       ALL `
5. 取消/etc/sudoers文件写权限,`chmod -v u-w /etc/sudoers` （删除sudoers写权限）


### 安装zsh
1. 安装git
```
    sudo yum install git -y
``` 
2. 安装zsh
```
   sudo yum install zsh -y
   chsh -s /bin/zsh     # 设置默认shell
```
3. 安装oh-my-zsh
```
    # via curl
    sh -c "$(curl -fsSL https://raw.githubusercontent.com/robbyrussell/oh-my-zsh/master/tools/install.sh)"
    
    # via wget
    sh -c "$(wget https://raw.githubusercontent.com/robbyrussell/oh-my-zsh/master/tools/install.sh -O -)"
```
4. 安装autojump
```
    yum -y install epel-release  # 安装epel, 默认仓库没有autojump包
    yum repolist  # 刷新仓库
    sudo yum install autojump autojump-zsh -y
```
5. 安装zsh-syntax-highlighting
```
    git clone https://github.com/zsh-users/zsh-syntax-highlighting.git ${ZSH_CUSTOM:-~/.oh-my-zsh/custom}/plugins/zsh-syntax-highlighting
    
    plugins=( [plugins...] zsh-syntax-highlighting)
```
6. 安装autosuggestions
```
    git clone https://github.com/zsh-users/zsh-autosuggestions ${ZSH_CUSTOM:-~/.oh-my-zsh/custom}/plugins/zsh-autosuggestions
    
    plugins=(zsh-autosuggestions)
```
6. 安装fzf
```
    git clone --depth 1 https://github.com/junegunn/fzf.git ~/.fzf
    ~/.fzf/install
```

### 安装python虚拟环境
安装包
```
    sudo pip install virtualenv
    sudo pip install virtualenvwrapper
```
在.zshrc文件结尾处写入
```
    export WORKON_HOME=~/.virtualenvs
    source /usr/bin/virtualenvwrapper.sh
```
### 安装tmux, `sudo yum install tmux -y`
在主目录下创建.tmux.conf， 并写入
```
    set -g mode-mouse on
    set -g mouse-resize-pane on
    set -g mouse-select-pane on
    set -g mouse-select-window on
    set-window-option -g mode-mouse on
```





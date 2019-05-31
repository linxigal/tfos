# -*- coding: utf-8 -*-
"""
:Author  : weijinlong
:Time    : 30/05/2019 22:21
:File    : fabfile.py
"""

import six
from fabric.api import cd, env, local, run

if six.PY2:
    import commands
else:
    import subprocess as commands

env.source_virtualenvwrapper = 'export WORKON_HOME=$HOME/.virtualenvs && source /usr/local/bin/virtualenvwrapper.sh'
env.python_path = 'export PYTHONPATH=$(pwd)'
env.virtualenv_workon_prefix = env.source_virtualenvwrapper + ' && workon %s'

env.base_dir = ''  # 项目部署根目录(必须填写)
env.work_dir = '/home/wjl/gitlab/tfos'  # 项目目录(必须填写)
env.work_on = ''  # workon 项目名称(必须填写)
env.web_dir = ''  # web目录
env.es_dir = ''  # ES 目录
env.code_repo = ''  # 代码仓库
env.manage_dir = ''  # manage.py所在目录;相对于项目目录,如 ./apps
env.requirement_dir = ''  # requirements.txt 所在目录,相对于项目目录,如 ./deploy
env.init_json_data = []  # 初始化需要load数据列表


def hosts(servers, user='default'):
    """
    连接远端服务器,服务器别名之间用逗号分隔

    :param servers: 远程服务器别名(alias),用逗号分隔
    :param user: 登录用户名
    :return: None
    """
    env.hosts = []
    remote_hosts = [str(i) for i in range(100, 103)]
    for host in servers.split():
        if host in remote_hosts:
            login_user = 'wjl' if user == 'default' else user
            host_ip_user = '{}@192.168.21.{}'.format(login_user, host)
            env.hosts.append(host_ip_user)


def pull(code_dir=None, repo="upstream", stash=False, branch='master'):
    """
    pull线上代码: code_dir代码目录,repo仓库名,stash=False|True,branch=master
    """

    with cd(code_dir or env.work_dir):
        # run('pwd')
        run('git status')
        if stash:
            run('git stash')
        # run('git checkout %s' % branch)

        run('git pull %s %s' % (repo, branch))


def test(code_dir=None, local_repo='upstream', remote_repo='origin', stash=False):
    """
    发布指定分支代码到服务器的测试(test)分支

    :param code_dir: 代码目录, 默认是~/github/TalentMiner
    :param local_repo: 本地仓库名
    :param remote_repo: 远程仓库名, 默认是origin
    :param stash: 是否使用stash, 默认False
    :return: None

    """
    status, output = commands.getstatusoutput('git describe --contains --all HEAD|tr -s "\n"')
    if status != 0:
        raise TypeError(output)
    print('current branch: {}'.format(output))
    local('git push -f {} {}:test'.format(local_repo, output.strip()))

    with cd(code_dir or env.work_dir):
        run('pwd')
        run('git status')
        if stash:
            run('git stash')
        run('git checkout master')
        run('git branch -D test', warn_only=True)
        run('git checkout -b test')
        run('git pull --rebase {} test'.format(remote_repo))

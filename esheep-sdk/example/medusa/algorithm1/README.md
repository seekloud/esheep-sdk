#### 1、configurations:
该文件夹内包含了程序的各种参数文件，包括游戏的环境参数、训练的各种超参等。在示例dqn_conf_1001.ini中给出了相关注释。在程序启动时，将参数文件作为启动参数喂给程序即可。

#### 2、config.py:
接收程序启动参数文件，读取初始化参数

#### 3、dueling_dqn.py:
包含dueling_dqn以及基本dqn的基本模型，可自行更改

#### 4、replay_buffer.py:
程序的环境数据存储类， 包含对每一帧数据的存储方法add_sample(), 以及每次训练提取数据的方法random_batch(), 其中phi()的作用是将选择喂给模型的多帧数据拼接在一起

#### 5、q_learning.py:
dqn算法的具体实现， 包括训练policy_net，更新target_net， choose_action等

#### 6、player.py: 
模拟玩家。根据当前环境数据选择action，并得到环境反馈，将环境反馈数据进行存储。按照一定的步长调用policy_net的训练

#### 7、experiment.py: 
初始化程序的各种对象， 包含env对象， player对象， buffer对象等以及程序的入口函数train，依据一定频率更新target_net参数

#### 8、runner.py: 
程序的入口， 调用experiment的train()函数
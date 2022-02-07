**基于BicycleGAN的3D分子生成**
论文 From Target to Drug: Generative Modeling for the Multimodal Structure-Based Ligand Design的实现，模型的代码由论文附件提供，训练代码参考BicycleGAN官方库。论文复现分为两个部分：基于BicycleGAN的3d分子结构生成和3d分子结构转smiles序列

**依赖的其它工具包**
rdkit
torch
moleculekit

**BicycleGAN模型**（3d分子生成 + LiGANN两个模型的整体inference）

/bicycle_gan

需要提供的数据：data/dude和data/refined-set（即pdb），基于这两类数据集分别训练两类bicycleGAN模型，每个样本必须提供一个蛋白文件和一个对应作为GT的ligand文件。

训练：针对两类差异比较大的数据集，分别有两份训练代码--train_bicycle_gan_dude.py和train_bicycle_gan_pdb.py，对应分别有两份蛋白voxel化的代码generator_dude和generator_pdb（注：我把蛋白voxel化单独在训练前进行并保存成pkl文件的原因是在训练中生成voxel的话会非常耗时，而ligand的voxel化则是在训练中读取数据后进行，因为ligand的voxel化耗时很少），训练的模型分别默认保存在checkpoints目录下。

训练好的模型：针对两类数据集分别存在checkpoints/DUDE和checkpoints/PDB目录下，bicycleGAN需要训练四个子模型--net_D、net_D2、net_G和net_E。

inference.py：基于训练好的BicycleGAN和caption模型和输入的蛋白文件生成ligand，可以选择性提供蛋白口袋的中心坐标，如不提供则默认取蛋白所有点坐标均值。（注：caption模型在推理阶段不需要提供vae模型，同LigDream；bicycleGAN模型的加载不需要额外提供模型保存路径，模型路径已经默认在配置代码里给出）。



**caption模型（3d分子转smiles）**

bicycle_gan/caption

smi_prepare.py：预处理给定的zinc数据集，提取指定长度的smi、编码字符并在data目录下生成符合要求的npy数据文件（例如：我最后生成的是长度60以内的zinc15类药分子数据集zinc15_max60.npy，可以自己选择训练的zinc子集）

train_cation.py：基于npy文件训练caption模型的三个子模型encoder、decoder和vae。最后训练的每阶段模型保存在saved_models/zinc15_max60/下。

**注意**
训练好的模型将在晚些时候通过云盘提供，readme文件也会持续更新补充。





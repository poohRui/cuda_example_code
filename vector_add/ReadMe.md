# Vector Add Example
> 通过串行和并行两种方法，对向量相加进行实现，并分别输出其前十个结果和运行时间。

## 函数介绍
vecAddSerial：串行向量相加函数。

vecAdd：并行程序的stub函数。

vecAddKernel：向量相加的kernel。

## 编译运行
通过MakeFile，使用make命令进行编译，生成vectorAdd可执行文件，
如果在本地或单独服务器上运行程序，可直接通过以下命令运行：
```
./vectorAdd
```

如果在集群上应使用对应作业提交命令进行提交，举个例子：
作业提交脚本vectorAdd.slurm如下：
```
#!/bin/bash
   #SBATCH -o job_%j_%N.out
   #SBATCH --partition=gpu
   #SBATCH -J hice 
   #SBATCH --get-user-env
   #SBATCH --nodes=1
   #SBATCH --ntasks-per-node=1
   #SBATCH --cpus-per-task=1
   #SBATCH --gres=gpu:1
   #SBATCH --time=120:00:00 
   module add cuda/8.0

   ./vectorAdd

```
提交作业：
```
sbatch vectorAdd.slurm
```
通过.out文件查看输出结果。

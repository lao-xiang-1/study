
```
# 进入目标目录 
cd /path/to/directory 
# 删除所有文件（包括隐藏文件），但保留目录本身 
rm -rf .[^.]* ..?* *
```

# u盘挂载
```
# 创建挂载目录
sudo mkdir -p /media/usb

# 挂载U盘
sudo mount /dev/sda1 /media/usb

# 卸载u盘
sudo umount /media/usb
```

# 解压操作
```
# 查看zip内容而不解压
unzip -l filename.zip

# 解压到指定文件夹
unzip filename.zip -d /path/to/dir
```

# 查看设备信息
```
# 查看系统配置
lsb_release -a

# cpu核心数
nproc

# 内存大小
free -h

# 硬盘空间
df -h
```


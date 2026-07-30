# Linux 实用命令

#code

## 清空目录内容（保留目录本身）

进入目标目录后，删除所有文件（包括隐藏文件），但保留目录本身：

```bash
cd /path/to/directory
find . -mindepth 1 -delete
```

> - `-mindepth 1`：排除 `.` 本身，只匹配目录内的内容
> - `-delete`：隐含 `-depth`，先删子目录内容再删目录本身，非空目录也能删
> - `find` 不经过 shell 通配符展开，天然匹配隐藏文件，文件名带空格 / 特殊字符也不受影响
> - 纯 POSIX 环境无 `-delete` 时，可改用 `find . -mindepth 1 -exec rm -rf {} +`

### 备选写法（纯 rm）

```bash
rm -rf .[^.]* ..?* *
```

> shell 的 `*` 默认不匹配 dotfile，需拆三段覆盖：
> - `.[^.]*`：隐藏文件（`.` 开头、第二个字符非 `.`），如 `.bashrc`
> - `..?*`：以 `..` 开头且长度 ≥ 3 的文件，如 `..swp`（避免误匹配 `..` 本身）
> - `*`：所有非隐藏文件

## 解压操作

```bash
# 查看 zip 内容而不解压
unzip -l filename.zip

# 解压到指定文件夹
unzip filename.zip -d /path/to/dir
```

## U 盘挂载

```bash
# 创建挂载目录
sudo mkdir -p /media/usb

# 挂载 U 盘
sudo mount /dev/sda1 /media/usb

# 卸载 U 盘
sudo umount /media/usb
```

> 设备名 `/dev/sda1` 仅为示例，实际设备名可通过 `lsblk` 或 `sudo fdisk -l` 查看，避免误操作系统盘。

## 系统信息查看

```bash
# 查看系统发行版信息
lsb_release -a

# CPU 核心数
nproc

# 内存大小
free -h

# 硬盘空间
df -h
```

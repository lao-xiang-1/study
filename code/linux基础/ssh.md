---
sr-due: 2026-07-29
sr-interval: 3
sr-ease: 250
---
#code 

# SSH（Secure Shell）

SSH 是一种加密的网络协议，用于在不安全的网络中安全地远程登录和执行命令。默认端口 **22**。

## 基本用法

```bash
# 远程登录
ssh user@host

# 指定端口（默认 22）
ssh -p 2222 user@host

# 指定私钥
ssh -i ~/.ssh/my_key user@host

# 执行单条命令（不进入交互式 shell）
ssh user@host "ls -la /var/log"
```

---

## 密钥认证（Key-based Authentication）

比密码登录更安全，推荐使用。

### 原理：非对称加密（Asymmetric Cryptography）

SSH 密钥认证基于**非对称加密**——密钥是**成对**生成的：

- **私钥（Private Key）**：只有你自己持有，必须保密。用于**签名**（证明"我是我"）。
- **公钥（Public Key）**：可以公开分发给任何人。用于**验证签名**（确认"你确实是你"）。

核心特性：

> **用公钥加密的数据，只有对应的私钥才能解密；用私钥签名的数据，公钥可以验证。**
>
> 从公钥推导出私钥在数学上是不可行的（基于大整数分解或椭圆曲线离散对数难题）。


**常见算法：**

| 算法 | 原理 | 密钥长度 | 特点 |
|---|---|---|---|
| **RSA** | 大整数分解难题 | 2048 / 4096 bit | 兼容性最好，速度较慢 |
| **Ed25519** | 椭圆曲线（Curve25519） | 256 bit | 安全性与 RSA 3072 相当，速度快，推荐 |
| **ECDSA** | 椭圆曲线（NIST 曲线） | 256 / 384 / 521 bit | 比 RSA 快，但 Ed25519 更优 |

### 认证过程（Challenge-Response）

SSH 密钥认证采用**质询-应答**（Challenge-Response）机制，整个过程私钥从不出现在网络上：

```
┌──────────────┐                          ┌──────────────┐
│   客户端      │                          │   服务器      │
│  (持有私钥)   │                          │ (存有多个公钥) │
└──────┬───────┘                          └──────┬───────┘
       │                                         │
       │  ① 发送认证请求，附带公钥                  │
       │  "用这个公钥登录 user"                     │
       │─────────────────────────────────────────▶│
       │                                         │
       │  ② 服务器在 authorized_keys               │
       │     中查找匹配的公钥                        │
       │     找到 → 生成随机数并用它加密              │
       │     没找到 → 直接拒绝                      │
       │◀─────────────────────────────────────────│
       │                                         │
       │  ③ 客户端用私钥解密，得到随机数             │
       │     将随机数和会话 ID 一起做哈希             │
       │     生成签名（Signature）                  │
       │                                         │
       │  ④ 发送签名（Response）                    │
       │─────────────────────────────────────────▶│
       │                                         │
       │  ⑤ 服务器用公钥验证签名                    │
       │     如果匹配 → 认证成功，建立加密会话        │
       │     如果不匹配 → 拒绝连接                  │
       │◀─────────────────────────────────────────│
       │                                         │
       ▼                                         ▼
```

**简单步骤**：
1. 服务端生成随机数，使用已有的公钥来加密，在把加密后的随机数发送给客户端
2. 客户端使用私钥解密，重新把随机数发给服务端，完成认证

**基本使用场景**：
1. ssh连接github，自己的电脑作为客户端，负责生成公钥和私钥；github作为服务器，存储公钥。
2. ssh连接服务器 来远程登录和执行命令

**Q: 服务器有多个公钥，怎么知道用哪个？**

> `authorized_keys` 是一个**文本文件**，每行一个公钥。客户端在发送请求时中会**主动把自己的公钥发给服务器**，服务器拿它和 `authorized_keys` 逐行比对（字符串匹配），找到匹配的那一行就用它来加密质询。所以服务器不需要"猜"——是客户端先告诉它"用这把"。

**Q: 客户端有多个私钥呢？**

> 客户端也会依次尝试 `~/.ssh/` 下所有私钥去解密质询，直到有一把能成功解密。也可以用 `ssh -i /path/to/key` 显式指定。`ssh -v` 可以看到客户端逐个尝试的过程。

### 生成密钥对

```bash
# Ed25519（推荐，更安全更快）
ssh-keygen -t ed25519 -C "your_email@example.com"

# RSA 4096（兼容性更好）
ssh-keygen -t rsa -b 4096 -C "your_email@example.com"
# 后面的-C表示注释，可以不写
```

生成的文件：
- `~/.ssh/id_ed25519` — 私钥（Private Key），**绝不能泄露**
- `~/.ssh/id_ed25519.pub` — 公钥（Public Key），放到远程服务器上

### 将公钥复制到远程服务器

```bash
# 方法1：ssh-copy-id（推荐，自动处理权限）
ssh-copy-id user@host

# 方法2：手动追加
cat ~/.ssh/id_ed25519.pub | ssh user@host "mkdir -p ~/.ssh && cat >> ~/.ssh/authorized_keys"
```

> 远程服务器的 `~/.ssh/authorized_keys` 权限应为 **600**，`~/.ssh` 目录权限应为 **700**，否则 SSH 会拒绝使用密钥。

---

## SSH 客户端配置（~/.ssh/config）

通过配置文件简化连接，避免每次输入长参数。

```bash
# ~/.ssh/config
Host myserver
    HostName 192.168.1.100
    User root
    Port 22
    IdentityFile ~/.ssh/id_ed25519

Host github
    HostName github.com
    User git
    IdentityFile ~/.ssh/github_key

Host jump
    HostName 10.0.0.1
    User admin
    # 通过跳板机访问
    ProxyJump jump
```

配置后可直接用别名连接：

```bash
ssh myserver          # 等同于 ssh -i ~/.ssh/id_ed25519 root@192.168.1.100
ssh github             # 等同于 ssh -i ~/.ssh/github_key git@github.com
```

**常用配置项：**

| 配置项 | 说明 |
|---|---|
| `HostName` | 远程主机 IP 或域名 |
| `User` | 登录用户名 |
| `Port` | 端口号（默认 22） |
| `IdentityFile` | 私钥路径 |
| `ProxyJump` | 跳板机/堡垒机 |
| `ServerAliveInterval` | 心跳间隔（秒），防止断连 |
| `ForwardAgent` | 是否转发 ssh-agent |

---

## 文件传输

### scp（Secure Copy）

```bash
# 上传：本地 → 远程
scp local_file.txt user@host:/remote/path/

# 下载：远程 → 本地
scp user@host:/remote/file.txt ./local/path/

# 上传整个目录（递归）
scp -r ./local_dir user@host:/remote/path/

# 指定端口
scp -P 2222 file.txt user@host:/path/
```

### sftp（SSH File Transfer Protocol）

```bash
# 连接
sftp user@host

# 常用命令（进入 sftp 交互模式后）
ls           # 列出远程目录
lls          # 列出本地目录
cd /path     # 切换远程目录
lcd /path    # 切换本地目录
get file.txt # 下载文件
put file.txt # 上传文件
get -r dir/  # 递归下载目录
put -r dir/  # 递归上传目录
```

---

## SSH 端口转发（Tunneling）

通过 SSH 隧道在不同端口之间转发流量。

```bash
# 本地转发（Local Forwarding）
# 将本地 8080 端口流量通过 SSH 转发到远程的 80 端口
ssh -L 8080:localhost:80 user@host
# 访问 http://localhost:8080 等同于访问远程主机的 80 端口

# 远程转发（Remote Forwarding）
# 将远程 9090 端口流量转发回本地的 3000 端口
ssh -R 9090:localhost:3000 user@host
# 远程主机访问 localhost:9090 等同于访问本机的 3000 端口

# 动态转发（Dynamic Forwarding / SOCKS 代理）
# 在本地 1080 端口创建 SOCKS5 代理
ssh -D 1080 user@host
# 浏览器或其他应用可将代理设置为 socks5://localhost:1080
```

**常见场景：**

| 场景 | 用法 |
|---|---|
| 访问远程内网服务 | `ssh -L 本地端口:内网IP:服务端口 user@跳板机` |
| 暴露本地服务到公网 | `ssh -R 远程端口:localhost:本地端口 user@公网服务器` |
| 临时科学上网 | `ssh -D 1080 user@境外服务器` |

---

## ssh-agent 管理密钥

`ssh-agent` 在后台缓存解密后的私钥，避免反复输入密钥密码。

```bash
# 启动 agent（通常系统已自动启动）
eval "$(ssh-agent -s)"

# 添加私钥到 agent
ssh-add ~/.ssh/id_ed25519

# 列出已加载的密钥
ssh-add -l

# 删除所有密钥
ssh-add -D
```

---

## 常见问题排查

```bash
# 调试连接（打印详细日志）
ssh -v user@host     # 基本调试
ssh -vv user@host    # 更详细
ssh -vvv user@host   # 最详细

# 检查远程服务器权限
ls -la ~/.ssh/
# authorized_keys 应为 600（-rw-------）
# .ssh 目录应为 700（drwx------）

# 检查 SELinux（CentOS/RHEL）
restorecon -R -v ~/.ssh
```

# git log相关
```bash
# 示例
git log --all --oneline --graph
# 参数说明
# --all ：查看所有分支（包括远程分支）
# --oneline ：一行显示
# --graph ：图形化显示（asc字符）

```

# 关联远程仓库
```bash
# 查看远程仓库地址
git remote -v

# 添加远程仓库地址
git remote add origin git@github.com:username/repo.git

# 修改远程仓库地址
git remote set-url origin git@github.com:username/repo.git
```

# 切换分支
```bash
# 新建分支
git switch -c <分支名>

# 切换分支
git switch <分支名>

# 基于远程分支创建本地分支
git switch -c <分支名> origin/<分支名>

# 将历史提交追加到当前分支
git cherry-pick <提交的哈希值>
```

# 提交
```bash
# 提交到指定远程仓库
git push <仓库名> main

# 提交远程新建的分支
git push --set-upstream <仓库名> <分支名>
```

# 生成ssh密钥
```bash
ssh-keygen -t ed25519
```

## 设置用户名和邮箱
仅仅本地提交时也是需要的
```bash
git config --global user.name "Aoi"
git config --global user.email "2697179230@qq.com"
```

## 设置中文
```bash
git config --global core.quotepath false
```

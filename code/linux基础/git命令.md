# Git 常用命令速查

#code/git
## 1. 基础配置
?
### 1.1 用户名与邮箱

每次提交都会记录用户名和邮箱，即使不推送远程也需要设置。

```bash
git config --global user.name "YourName"
git config --global user.email "your@email.com"
```

### 1.2 中文显示

防止 `git log`、`git status` 等命令将中文文件名显示为转义序列：

```bash
git config --global core.quotepath false
```

### 1.3 SSH 密钥

生成 Ed25519 密钥对（推荐），公钥添加到 GitHub/GitLab 的 SSH Keys 设置中：

```bash
ssh-keygen -t ed25519 -C "your@email.com"
```

> 默认保存在 `~/.ssh/id_ed25519`，`id_ed25519.pub` 为公钥。
<!--SR:!2026-08-03,4,270-->

---

#code/git
## 2. 仓库操作
?
### 2.1 初始化与克隆

```bash
# 在当前目录初始化仓库
git init

# 克隆到指定目录
git clone <url> <目录名>
```

### 2.2 关联远程仓库

```bash
# 查看已关联的远程仓库
git remote -v

# 添加远程仓库
git remote add origin git@github.com:username/repo.git

# 修改远程仓库地址
git remote set-url origin git@github.com:username/repo.git

# 删除远程仓库关联
git remote remove origin
```
<!--SR:!2026-08-03,4,270-->

---

#code/git
## 3. 分支操作
?

### 3.1 创建与切换

```bash
# 新建并切换到分支
git switch -c <分支名>

# 切换到已有分支
git switch <分支名>

# 基于远程分支创建本地分支
git switch -c <分支名> origin/<分支名>
```

> 旧版使用 `git checkout -b <分支名>` 创建分支，`git checkout <分支名>` 切换分支，效果相同。

### 3.2 查看与删除

```bash
# 查看本地分支
git branch

# 查看所有分支（含远程）
git branch -a

# 删除已合并的分支
git branch -d <分支名>

# 强制删除分支（未合并也可删）
git branch -D <分支名>
```

### 3.3 合并分支

```bash
# 将指定分支合并到当前分支
git merge <分支名>

# 变基合并（线性历史，更整洁）
git rebase <分支名>
```
<!--SR:!2026-08-02,3,250-->

---

#code/git
## 4. 暂存与提交
?
### 4.1 基本流程

```bash
# 查看工作区状态
git status

# 暂存指定文件
git add <文件>

# 暂存所有改动
git add .

# 提交
git commit -m "提交信息"

# 暂存并一次提交所有已跟踪文件的修改
git commit -am "提交信息"
```

### 4.2 查看已追踪文件

```bash
# 列出所有已追踪的文件
git ls-files

# 查看未追踪的文件
git ls-files --others --exclude-standard

# 查看已修改的文件
git ls-files --modified

# 查看已删除但尚未提交的文件
git ls-files --deleted
```

### 4.3 修改提交

```bash
# 修改最近一次提交信息
git commit --amend -m "新的提交信息"

# 撤销暂存（保留工作区修改）
git restore --staged <文件>

# 撤销工作区修改
git restore <文件>
```
<!--SR:!2026-08-02,3,250-->

---

#code/git
## 5. 推送与拉取
?
### 5.1 推送

```bash
# 推送到远程仓库
git push origin main

# 首次推送新分支，建立上游追踪
git push --set-upstream origin <分支名>

# 推送并建立上游追踪（简写）
git push -u origin <分支名>
```

### 5.2 拉取与获取

```bash
# 拉取远程更新并合并
git pull

# 仅获取远程更新（不自动合并）
git fetch origin

# 拉取指定远程分支
git pull origin <分支名>
```
<!--SR:!2026-08-03,4,270-->

---

#code/git
## 6. 查看历史
?
### 6.1 git log

```bash
# 简洁图形化查看（推荐）
git log --all --oneline --graph

# 查看最近 N 条提交
git log -n 5

# 查看某个文件的提交历史
git log -- <文件路径>
```

> 参数说明：`--all` 显示所有分支，`--oneline` 单行显示，`--graph` 图形化分支关系。

### 6.2 查看差异

```bash
# 工作区与暂存区的差异
git diff

# 暂存区与最近提交的差异
git diff --staged

# 两个提交之间的差异
git diff <commit1> <commit2>
```
<!--SR:!2026-08-03,4,270-->

---

#code/git
## 7. 其他实用命令
?

```bash
git reflog # 查看git所有的操作历史
```

### 7.1 cherry-pick

将指定提交应用到当前分支（跨分支摘取提交）：

```bash
git cherry-pick <commit-hash>
```

### 7.2 stash 暂存

临时保存工作区修改，方便切换分支或拉取代码：

```bash
# 暂存当前修改
git stash

# 暂存并附带说明
git stash save "说明信息"

# 查看暂存列表
git stash list

# 恢复最近一次暂存
git stash pop

# 恢复指定暂存
git stash apply stash@{n}
```

### 7.3 标签

```bash
# 查看所有标签
git tag

# 创建轻量标签
git tag <标签名>

# 创建附注标签
git tag -a <标签名> -m "说明"

# 推送标签到远程
git push origin <标签名>

# 推送所有标签
git push origin --tags
```

### 7.4 撤销与回退

```bash
# 不加参数时，默认使用--mixed
git reset --mixed <commit>

# 回退最近一次提交（保留修改在暂存区）
git reset --soft HEAD~1

# 回退到指定提交（丢弃所有修改）
git reset --hard <commit>
```

`--mixed`和`--soft` 的异同：
- 两者都切换HEAD，保留工作区（当前写好的代码不变）
- `mixed`重置暂存区，相当于没有多余的操作，当前代码 所有相对于HEAD的修改 都是unstaged
- `soft`保留暂存区，比如说：你现在的暂存区里有删除1.txt的操作，但实际上回退的HEAD中根本没有1.txt。而回退后这个“删除1.txt的操作”仍然保留在暂存区。
> 在这讲不清楚，建议自己试试
<!--SR:!2026-08-02,2,230-->

---

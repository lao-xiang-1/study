[官方文档](https://cmake.com.cn/cmake/help/latest/manual/cmake.1.html)

CMake 的命令行参数主要分为**项目配置**和**构建执行**两大类。

### ⚙️ 项目配置阶段

这个阶段的核心是读取 `CMakeLists.txt`，生成特定构建系统的文件（如 Makefile 或 Visual Studio 解决方案）。

**核心参数**

*   **`-D <var>=<value>`**：用于在命令行定义或覆盖 CMake 缓存变量。
    *   **常见变量**：
        *   `CMAKE_BUILD_TYPE`：指定构建类型，如 `Debug`、`Release`。
        *   `CMAKE_INSTALL_PREFIX`：指定`make install`的安装路径。
        *   `BUILD_SHARED_LIBS`：控制是构建动态库(`ON`)还是静态库(`OFF`)。
    *   **示例**：`cmake -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=ON ..`
*   **`-G <generator-name>`**：指定构建系统生成器。
    *   **常见生成器**：`Unix Makefiles`, `Ninja`, `Visual Studio 17 2022`, `Xcode` 等。
    *   **示例**：`cmake -G "Ninja" ..`
*   **`-S <path-to-source>` 和 `-B <path-to-build>`**：显式指定**源码目录**和**构建目录**。
    *   **示例**：`cmake -S /path/to/source -B /path/to/build`

**其他有用的参数**

*   **`-C <initial-cache>`**：用脚本文件预加载缓存变量。
*   **`-U <globbing_expr>`**：从 CMake 缓存中移除匹配的变量。
*   **`-L[A][H]`**：列出当前缓存的变量（非高级选项）。
*   **`-N`**：仅加载缓存，不执行配置和生成步骤。
*   **`--graphviz=[file]`**：生成项目的依赖关系图。
*   **`--system-information [file]`**：输出当前系统的详细信息。
*   **`--debug-trycompile`**：保留 `try_compile` 的临时文件，用于调试。

### 🛠️ 构建执行阶段

在构建目录生成构建文件后，使用此模式执行实际的编译和链接。

*   **`--build <dir>`**：**核心命令**，用于构建项目。
    *   **常用选项**：
        *   `--target <tgt>`：指定要构建的目标。
        *   `--config <cfg>`：对多配置生成器指定构建类型。
        *   `--parallel [<jobs>]`：并行构建。
        *   `--clean-first`：构建前先执行 `clean`。
    *   **示例**：`cmake --build build --target install --config Release --parallel 4`

### 🧰 其他实用模式

*   **`-E <command>`**：**命令行工具模式**，执行跨平台的内置命令。
    *   **常见命令**：`echo`, `rm`, `copy`, `make_directory`, `tar`, `time` 等。
    *   **示例**：`cmake -E make_directory build`
*   **`-P <file>`**：**脚本模式**，执行一个 CMake 脚本文件，不进行配置或生成。
    *   **示例**：`cmake -P MyScript.cmake`
*   **`--find-package`**：**查找包模式**，模仿 `pkg-config` 查找库。

### 💎 示例

`cmake` 命令的核心用法可以概括为：

```bash
# 1. 配置项目 (在 build 目录外执行)
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release

# 2. 构建项目
cmake --build build --parallel 4

# 3. (可选) 安装项目
cmake --install build --prefix /path/to/install
```
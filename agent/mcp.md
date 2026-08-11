# Model Context Protocol（MCP）

> Model Context Protocol（MCP，模型上下文协议）是由 Anthropic 提出的一种开放协议，用于让大语言模型（LLM）安全、结构化地访问外部工具、数据源和服务。

---

## 1. 为什么需要 MCP

### 1.1 传统 Function Calling 的局限

- 每个模型、每个平台有各自的工具调用格式。
- 应用开发者需要为每个外部服务写一遍适配层。
- 上下文（Context）传递方式不统一，模型难以理解复杂数据源。

### 1.2 MCP 的目标

- **统一接口**：让任何支持 MCP 的模型都能调用任何支持 MCP 的服务。
- **安全可控**：数据留在本地，模型通过协议请求而非直接访问敏感资源。
- **可组合**：多个 MCP Server 可以像插件一样同时挂载到一个 Host 上。

---

## 2. 核心架构

MCP 采用**客户端-服务器**架构，角色定义清晰：

### 2.1 三种核心角色

| 角色 | 英文名 | 职责 |
| --- | --- | --- |
| **Host（宿主）** | Host | 运行 LLM 的应用程序，例如 Claude Desktop、Claude Code、IDE 插件等。 |
| **Client（客户端）** | Client | Host 内代表某个 Server 的连接实例，负责协议通信。 |
| **Server（服务器）** | Server | 提供具体能力的外部服务，例如文件系统、数据库、GitHub、Slack 等。 |

### 2.2 数据流向

```text
┌─────────────────────────────────────────────────────────┐
│                         Host                            │
│  ┌─────────────────┐      ┌─────────────────────────┐   │
│  │   LLM / Agent   │◄────►│  MCP Client (per Server)│   │
│  └─────────────────┘      └─────────────────────────┘   │
│                                      │                   │
│                                      │ MCP Protocol      │
│                                      ▼                   │
│                           ┌─────────────────────┐        │
│                           │     MCP Server      │        │
│                           │  (Tools / Resources │        │
│                           │   / Prompts)        │        │
│                           └─────────────────────┘        │
└─────────────────────────────────────────────────────────┘
```

> 一个 Host 可以同时连接多个 MCP Server，每个 Server 暴露自己的能力清单（Capability）。

---

## 3. 三大能力原语

MCP Server 向 Host 暴露三类原语（Primitives）：

### 3.1 Tools（工具）

- **定义**：可被模型调用的函数或操作，例如查询天气、发送邮件、执行代码。
- **特点**：
  - 模型决定是否调用（Decision）。
  - 调用结果返回给模型，模型继续生成回复。
  - 通常带有 JSON Schema 描述参数。

### 3.2 Resources（资源）

- **定义**：只读的数据引用，例如文件内容、数据库记录、网页 URL。
- **特点**：
  - 不修改外部状态。
  - 可被模型读取以丰富上下文。
  - 类似“带有 URI 的数据附件”。

### 3.3 Prompts（提示模板）

- **定义**：预定义的提示词模板，Server 可以主动向 Host 提供可复用的提示结构。
- **特点**：
  - 帮助用户快速发起常见任务。
  - 可包含变量插槽，由 Host 填充。

---

## 4. 通信协议

### 4.1 传输层（Transport）

MCP 支持多种传输方式：

| 传输方式 | 适用场景 |
| --- | --- |
| **stdio** | 本地进程，最常见，Server 作为子进程启动。 |
| **SSE / HTTP** | 远程服务，通过 HTTP 或 Server-Sent Events 通信。 |
| **WebSocket** | 需要双向实时通信的场景。 |

### 4.2 消息格式

- 基于 **JSON-RPC 2.0**。
- 消息类型：
  - `initialize`：握手与能力协商。
  - `tools/list`、`tools/call`：工具发现与调用。
  - `resources/list`、`resources/read`：资源发现与读取。
  - `prompts/list`、`prompts/get`：提示模板发现与获取。

---

## 5. 典型工作流

以“让 Claude 读取本地项目文件并总结”为例：

1. **配置**：在 Claude Desktop / Claude Code 中注册文件系统 MCP Server。
2. **发现**：Host 向 Server 请求可用工具列表（如 `read_file`、`list_directory`）。
3. **用户提问**：用户说“请总结我当前项目的 README”。
4. **模型决策**：LLM 判断需要调用 `read_file` 工具读取 `README.md`。
5. **执行**：Host 通过 MCP Client 发送 `tools/call` 请求给 Server。
6. **返回结果**：Server 返回文件内容，LLM 基于内容生成总结。

---

## 6. MCP 与 Function Calling 的对比

| 维度 | 传统 Function Calling | MCP |
| --- | --- | --- |
| 调用格式 | 各模型不同（OpenAI、Anthropic、Google 等） | 统一 JSON-RPC 协议 |
| 服务发现 | 应用硬编码 | 动态 capabilities 协商 |
| 生态复用 | 每个应用单独对接 | 一次开发，多处复用 |
| 安全模型 | 依赖应用实现 | 协议层面强调本地、用户授权 |
| 数据上下文 | 手动注入 | Resources / Prompts 原生支持 |

---

## 7. 写一个 MCP Server 的关键要素

### 7.1 最小实现结构

```python
from mcp.server import Server
from mcp.types import Tool, TextContent

server = Server("my-server")

@server.list_tools()
async def list_tools():
    return [Tool(
        name="hello",
        description="Say hello",
        inputSchema={"type": "object", "properties": {}}
    )]

@server.call_tool()
async def call_tool(name, arguments):
    if name == "hello":
        return [TextContent(type="text", text="Hello, MCP!")]
```

### 7.2 注意事项

- **Schema 要准确**：模型依赖参数描述决定如何调用。
- **错误处理要清晰**：返回可理解的错误信息，帮助模型修正。
- **权限最小化**：Server 只暴露必要的资源和操作。
- **状态隔离**：尽量避免 Server 内部维护全局可变状态。

---

## 8. 小结

- MCP 是 LLM 与外部世界交互的**开放协议**。
- 核心角色是 **Host、Client、Server**。
- 三大原语是 **Tools、Resources、Prompts**。
- 通信基于 **JSON-RPC**，传输层常用 **stdio / HTTP / WebSocket**。
- 优势在于**统一、安全、可组合**，降低模型应用与外部服务集成的成本。

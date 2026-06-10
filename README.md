# 🤖 AI Daily Digest

每日自动收集 AI 领域资讯，通过 Claude 智能摘要后发送邮件。

## ✨ 功能特点

- 📡 **多源采集**：Hacker News、Reddit、RSS 三大来源，覆盖全面
- 🧠 **AI 智能整理**：Claude 自动筛选、分类、生成中文摘要
- 📧 **精美邮件**：响应式 HTML 邮件，分类清晰，来源彩色标注
- 🕘 **定时发送**：GitHub Actions 每天早上 9 点（北京时间）自动运行
- 💰 **极低成本**：每天约 ¥0.1-0.5（仅 Claude API 调用费用）

## 🚀 快速开始

### 1. Fork 或克隆本项目

```bash
git clone https://github.com/zhangyiheng0216/infomation_send.git
cd infomation_send
```

### 2. 配置 GitHub Secrets

在 GitHub 仓库中设置以下 Secrets：

**Settings → Secrets and variables → Actions → New repository secret**

| Secret 名称 | 说明 | 示例值 |
|------------|------|--------|
| `ANTHROPIC_API_KEY` | Claude API 密钥 | `sk-ant-...` |
| `SMTP_USER` | QQ 邮箱地址 | `123456@qq.com` |
| `SMTP_PASSWORD` | QQ 邮箱授权码（非 QQ 密码） | `abcdefghijklmnop` |
| `EMAIL_TO` | 收件人邮箱 | `you@example.com` |

### 3. 获取 QQ 邮箱授权码

1. 登录 QQ 邮箱网页版
2. 进入 **设置 → 账户**
3. 找到 **POP3/SMTP 服务**
4. 点击 **开启**
5. 按提示获取 16 位授权码

### 4. 获取 Claude API Key

1. 访问 [console.anthropic.com](https://console.anthropic.com)
2. 注册/登录账号
3. 进入 **API Keys**
4. 创建新的 API Key

## 🧪 本地测试

```bash
# 安装依赖
pip install -r requirements.txt

# 设置环境变量
export ANTHROPIC_API_KEY="sk-ant-..."
export SMTP_USER="your-qq@qq.com"
export SMTP_PASSWORD="your-auth-code"
export EMAIL_TO="recipient@example.com"

# 运行
python main.py
```

## 📁 项目结构

```
ai-daily-digest/
├── .github/
│   └── workflows/
│       └── daily.yml          # GitHub Actions 定时任务
├── config.py                   # 配置文件（API Keys、RSS 源、关键词）
├── collectors.py               # 数据采集（HN、Reddit、RSS）
├── curator.py                  # Claude 智能筛选、分类、摘要
├── emailer.py                  # HTML 邮件构建 + QQ SMTP 发送
├── main.py                     # 主入口
├── template.html               # 邮件模板（Jinja2）
├── requirements.txt            # Python 依赖
└── README.md                   # 本文件
```

## 🔧 自定义配置

编辑 `config.py` 可调整：

### 信息来源
- **HN 搜索词**：`HN_SEARCH_QUERIES` — 添加/删除关键词
- **Reddit 子版块**：`REDDIT_SUBREDDITS` — 添加感兴趣的 AI 社区
- **RSS 订阅源**：`RSS_FEEDS` — 添加/删除博客或新闻网站

### 筛选规则
- **HN 最低分数**：`HN_MIN_POINTS` — 默认 15，提高可减少噪音
- **Reddit 最低分数**：`REDDIT_MIN_SCORE` — 默认 20

### 发送时间
- 修改 `.github/workflows/daily.yml` 中的 cron 表达式
- `0 1 * * *` = UTC 01:00 = 北京时间 09:00
- 改为 `0 12 * * *` = UTC 12:00 = 北京时间 20:00

## 📊 数据来源

### Hacker News
- 使用 Algolia Search API
- 按日期搜索昨天的 AI 相关帖子
- 关键词：AI, LLM, GPT, Claude, OpenAI, machine learning 等

### Reddit
- r/MachineLearning — 学术论文、技术讨论
- r/artificial — AI 行业新闻
- r/LocalLLaMA — 开源模型、本地部署

### RSS Feeds
- **官方博客**：OpenAI, Anthropic, Google AI, DeepMind, Hugging Face
- **科技媒体**：MIT Tech Review, The Verge, VentureBeat, Ars Technica
- **社区博客**：TLDR AI, MarkTechBlog, Synced Review

## 📧 邮件示例

邮件包含：
- 📊 **统计信息**：总共收集了多少条，各来源分布
- 📄 **研究论文**：学术论文、技术报告
- 🚀 **产品发布**：新模型、新功能、API 更新
- 🏢 **行业动态**：融资、收购、合作
- 💻 **开源项目**：GitHub 仓库、开源工具
- 🛠️ **工具与框架**：开发工具、库、平台
- 📊 **数据集与基准**：新数据集、基准测试结果
- 🧠 **观点与讨论**：行业趋势、专家观点

每条信息包含：
- 英文原标题
- 原文链接
- 中文摘要（2-3 句话）
- 来源标签（HN 橙色 / Reddit 红色 / RSS 蓝色）
- 热度分数（如适用）

## 🐛 常见问题

### Q: 邮件没有收到？
- 检查 GitHub Actions 日志（Actions 标签页）
- 确认所有 Secrets 已正确设置
- 检查 QQ 邮箱的 SMTP 服务是否已开启
- 查看垃圾邮件文件夹

### Q: Claude API 调用失败？
- 程序会自动降级：使用基础分类，仍会发送邮件
- 邮件顶部会显示警告信息
- 检查 API Key 是否有效，账户是否有余额

### Q: 某个数据源没有内容？
- 正常现象，某些日期可能某个来源没有符合条件的内容
- 程序会继续处理其他来源的内容

### Q: 如何修改发送时间？
- 编辑 `.github/workflows/daily.yml`
- 修改 cron 表达式（UTC 时间）
- 北京时间 = UTC + 8 小时

### Q: 费用是多少？
- GitHub Actions：免费（每月 2000 分钟）
- Claude API：约 $0.03-0.08/次（每天一次）
- QQ 邮箱 SMTP：免费
- **总计：约 ¥0.1-0.5/天**

## 📝 更新日志

### v1.0.0 (2026-06-10)
- ✨ 初始版本
- ✅ 支持 HN、Reddit、RSS 三大数据源
- ✅ Claude 智能筛选、分类、中文摘要
- ✅ QQ 邮箱 SMTP 发送
- ✅ GitHub Actions 定时任务
- ✅ 精美 HTML 邮件模板

## 🤝 贡献

欢迎提交 Issue 和 Pull Request！

## 📄 许可证

MIT License

---

**Powered by [Claude API](https://www.anthropic.com) · 数据来自 Hacker News, Reddit, RSS**

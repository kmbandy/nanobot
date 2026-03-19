# Discord Shortcuts

All commands are handled by **eng-1** (OmniCoder-9B). Commands are processed
before the LLM — instant returns, no tool calls, no context used.

---

## Fleet / System

| Command | Description |
|---------|-------------|
| `/help` | Show all available commands |
| `!ping` | Bot liveness check — model name + uptime |
| `!status` | Fleet GPU/RAM/agent status (all bots) |
| `!tasks` | Pending pipeline tasks from ChromaDB |
| `!models` | List all model files in `~/models/` |
| `!mad-hot-swap <bot> <model>` | Swap running model on a bot and restart it. e.g. `!mad-hot-swap eng-1 qwen3.5-9b` |
| `!restart <bot>` | Restart eng-1, eng-2, or arch-1 |
| `/new` or `!reset` | Clear current session history |
| `/stop` | Stop current running task |

---

## Research / Web

| Command | Description |
|---------|-------------|
| `!summarize <url>` | Fetch a webpage and summarize it |
| `!wikipedia <subject>` | Wikipedia summary for any subject (REST API, falls back to search) |
| `!gh <owner/repo>` | GitHub repo info + latest release. e.g. `!gh anthropics/claude-code` |
| `!search <query>` | Web search, 5 results, then stops |
| `!arxiv <query>` | arxiv search, 5 papers, then stops |
| `!reddit <sub> <topic>` | Subreddit search, 5 posts, then stops |

---

## Code / Dev

| Command | Description |
|---------|-------------|
| `!diff` | Show nanobot commits since last upstream sync |
| `!mad-code-summary <path>` | Summarize a file's purpose. Resolves against `~/mad-lab-mcp/` and `~/nanobot/` |
| `!python <description>` | Generate a Python script or function |
| `!explain <code/concept>` | Plain-English explanation |

---

## Mad-Lab

| Command | Description |
|---------|-------------|
| `!brief` | On-demand overnight summary (same as morning-brief cron) |
| `!mem <query>` | Search ChromaDB knowledge base |
| `!new-nanobot` | Interactive wizard — create a new bot config + systemd service files via Discord |
| `!new-nanobot cancel` | Cancel an in-progress wizard |
| `!remind <time> <msg>` | Set a reminder. e.g. `!remind 20m check arch-1` |

---

## Sports / Entertainment

| Command | Description |
|---------|-------------|
| `!united-latest` | Man Utd last result, next fixture (EST), PL standings (±2 teams), r/ManUtd talking points |

---

## Cron Jobs

These run automatically — no Discord command needed.

| Script | Schedule | Description |
|--------|----------|-------------|
| `morning-brief` | Daily 8am EST | Overnight ChromaDB summary posted to Discord |
| `fut-daily` | Daily 1:30pm EST | EA FC 26 / FUT content from fut.gg |
| `run_scraper.py` | Daily midnight EST | arxiv, HackerNews, GitHub trending + releases scraped to ChromaDB |
| `upstream-check-cron` | Monday 8am EST | Alert if nanobot upstream has new commits; warns on protected file changes |
| `nightly-shutdown.sh` | Configurable | Nightly server shutdown |
| `weekly-digest` | Configurable | Weekly ChromaDB digest |

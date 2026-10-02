# Codex 评审 runbook（本地调用规则）

> 建立于 2026-10-01（LMABO AF 移植评审）。约定沿用 runs/ 下既有的
> `codex_*_prompt.txt` / `codex_*_review.log` / `codex_*_auditN.log` 历史。

## 模型与配置

- 模型：**`gpt-6.1-sol`**（用户指定，2026-10-01；`~/.codex/config.toml`
  已设 `model = "gpt-6.1-sol"`、`model_reasoning_effort = "high"`，
  命令行仍显式传 `-m` 以防配置漂移）。
- 评审一律 **read-only sandbox**（`-s read-only`），prompt 里写明
  "do NOT edit files, do NOT run tests/training"。

## 命令模板

```bash
cd /home/lcy/Documents/projects/opt_agent_v3
codex exec -m gpt-6.1-sol -s read-only \
  -C /home/lcy/Documents/projects/opt_agent_v3 \
  "$(cat runs/codex_<topic>_prompt.txt)" \
  > runs/codex_<topic>_review.log 2>&1
```

多轮审计（round N）沿用 `runs/codex_<topic>_auditN.log`；
每轮 prompt 引用上一轮日志并要求对上轮 findings 逐条给出
ADDRESSED / PARTIAL / UNADDRESSED。

## 网络问题的代理回退

直连失败（超时 / 5xx / stream error）时，先导出代理再原命令重试：

```bash
export https_proxy=http://127.0.0.1:7890 \
       http_proxy=http://127.0.0.1:7890 \
       all_proxy=socks5://127.0.0.1:7890
```

（规则来源：用户 2026-10-01 指示。）

## Prompt 的house style（照抄先例，见 runs/codex_boundary_final5_prompt.txt）

1. 自包含：评审者只有仓库读权限——背景、变更清单（文件级）、
   "WHAT CHANGED / REVIEW TASKS / Answer format" 三段式。
2. 明确 scope 与非目标：无关的未提交改动要显式排除
   （例："file also carries PRE-EXISTING uncommitted edits … audit only
   the X hunks here"）。
3. 任务分项 A/B/C…，每项一个短段落或 PASS。
4. findings 标签 `[BLOCKER|MAJOR|MINOR]` + `file:line` + 一行修复。
5. 最后一行固定：`VERDICT: APPROVE | REVISE`。
6. 可读任意仓库文件、git log/diff、runs/*.json；PDF 可用 pdftotext。

## 产物落盘

| 产物 | 路径 |
|---|---|
| 评审 prompt | `runs/codex_<topic>_prompt.txt`（入库，可复现） |
| 评审日志 | `runs/codex_<topic>_review.log`（或多轮 `_auditN.log`） |
| 处置记录 | 视变更性质进对应 dated doc / PROGRESS / guide |

历史先例：`runs/codex_boundary_round{2..8}_review.log`、
`runs/codex_boundary_final{,2..5}_audit*.log`、`runs/codex_D_value_audit.log`。

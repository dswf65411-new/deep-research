# Multilingual Search Configuration

## Extended-language engines (enable by topic)

| Language | Serper parameters | Typical platforms | Enablement condition |
|------|-----------|---------|---------|
| Simplified Chinese (Google) | gl=cn, hl=zh-CN | Zhihu, CSDN, Juejin | Topic touches Chinese tech/market |
| Simplified Chinese (Baidu) | Baidu Search API | WeChat Official Accounts, Baidu Tieba | Enabled together with Serper ZH-CN |
| Japanese | gl=jp, hl=ja | Qiita, Zenn | Japan market/manufacturing/gaming |
| German | gl=de, hl=de | Heise, Golem | GDPR/Industry 4.0 |
| French | gl=fr, hl=fr | Le Monde Informatique | EU regulation |
| Spanish | gl=es, hl=es | — | Latin America market |
| Portuguese | gl=br, hl=pt | — | Brazil market |
| Russian | gl=ru, hl=ru | Habr | Russia/Eastern Europe tech |

## Search Matrix

```
EN query   -> WebSearch + Brave                                    (2)
ZH-TW      -> WebSearch + Serper(gl=tw, hl=zh-TW)                  (2)
Academic   -> Serper(site:semanticscholar.org + site:arxiv.org)    (1-2, for academic topics)
--- Extended languages (when enabled) ---
ZH-CN      -> Serper(gl=cn, hl=zh-CN) + Baidu                      (2)
JA/DE/FR/ES/PT/RU -> Serper(matching parameters)                   (1 each)
```

## Translation Rules

- Do not translate word-for-word; use the **idiomatic terminology** of that language's community
- No mature local term -> keep English for the search
- Extended-language query count = half of the core language (to save budget)

## China Mainland Restrictions

Google is blocked; closed platforms such as WeChat Official Accounts and Baidu Tieba are not indexed by Google. When searching for Chinese information, run Serper ZH-CN and Baidu in parallel.

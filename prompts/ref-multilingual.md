# 多語言搜尋設定

## 擴展語言引擎（依主題啟用）

| 語言 | Serper 參數 | 典型平台 | 啟用條件 |
|------|-----------|---------|---------|
| 🇨🇳 简中 (Google) | gl=cn, hl=zh-CN | 知乎、CSDN、掘金 | 涉及中國技術/市場 |
| 🇨🇳 简中 (百度) | Baidu Search API | 微信公眾號、百度貼吧 | 與 Serper ZH-CN 同時啟用 |
| 🇯🇵 日文 | gl=jp, hl=ja | Qiita、Zenn | 日本市場/製造/遊戲 |
| 🇩🇪 德文 | gl=de, hl=de | Heise、Golem | GDPR/工業 4.0 |
| 🇫🇷 法文 | gl=fr, hl=fr | Le Monde Informatique | 歐盟法規 |
| 🇪🇸 西文 | gl=es, hl=es | — | 拉美市場 |
| 🇧🇷 葡文 | gl=br, hl=pt | — | 巴西市場 |
| 🇷🇺 俄文 | gl=ru, hl=ru | Habr | 俄羅斯/東歐技術 |

## 搜尋矩陣

```
EN query   → WebSearch + Brave                                    (2)
ZH-TW      → WebSearch + Serper(gl=tw, hl=zh-TW)                 (2)
學術        → Serper(site:semanticscholar.org + site:arxiv.org)    (1-2，學術主題時)
--- 擴展語言（啟用時）---
ZH-CN      → Serper(gl=cn, hl=zh-CN) + Baidu                     (2)
JA/DE/FR/ES/PT/RU → Serper(對應參數)                              (各1)
```

## 翻譯規則

- 不逐字直譯，用該語言社群的**慣用術語**
- 無成熟本地術語 → 保留英文搜尋
- 擴展語言 query 數量 = 核心語言的一半（省預算）

## 中國大陸限制

Google 被封鎖，微信公眾號、百度貼吧等封閉平台不被 Google 索引。搜中國資訊時 Serper ZH-CN + 百度同時搜。

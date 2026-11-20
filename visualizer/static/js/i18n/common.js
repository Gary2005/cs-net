/**
 * 四个页面共用的文案：顶栏导航、品牌副标题、文档标题、几个到处都出现的按钮。
 *
 * 键名用 `common.` 前缀；各页面自己的文案在 replay.js / library.js /
 * players.js 里，前缀分别是 `replay.` / `lib.` / `player.`（玩家索引和玩家
 * 档案同属"玩家"这块，共用一个文件）。
 *
 * 导航三项的中文必须和模板里的原文一字不差 —— `app/tests/test_ui_wiring.py`
 * 会拿这四个页面的顶栏逐字比对，改词要连着测试一起改。
 */

export default {
  zh: {
    'common.nav.library': 'Demo 库',
    'common.nav.players': '玩家',
    'common.nav.replay': '可视化回放',

    'common.brand.player': '玩家档案',
    'common.brand.version': '第 5 版',

    'doc.library': 'cs-net — Demo 库',
    'doc.players': 'cs-net — 玩家',
    'doc.player': 'cs-net — 玩家档案',
    'doc.replay': 'cs-net — 可视化回放',

    'common.action.cancel': '取消',
    'common.action.delete': '删除',
    'common.action.remove': '移除',
    'common.action.add': '添加',
    'common.action.loading': '加载中…',

    // ── 分享图（share-card.js 画在 canvas 上，两个页面共用一套文案）──────────
    'common.share.btn': '分享图片',
    'common.share.btn.title': '把当前这份数据导出成一张 PNG（带 logo 和项目地址）',
    'common.share.saving': '正在生成图片…',
    'common.share.done': '已导出 {name}',
    'common.share.fail': '导出失败：{msg}',

    'common.share.kind.match': '单场数据',
    'common.share.kind.player': '玩家数据',
    'common.share.project': 'github.com/Gary2005/cs-net',
    'common.share.generated': '由 cs-net 生成',
    'common.share.rounds': '{n} 回合',
    'common.share.col.player': '选手',
    'common.share.col.rounds': '回合',
    'common.share.tile.matches': '场次',
    'common.share.tile.hs': '爆头率',
    'common.share.section.trad': '传统数据',
    'common.share.section.ai': 'AI 数据',
    'common.share.recent': '最近 {n} 场',
    'common.share.lastSeen': '最近出场 {date}',

    'common.share.ai.killImpact': '击杀影响',
    'common.share.ai.roundImpact': '回合影响',
    'common.share.ai.deathRisk': '死亡风险',
    'common.share.ai.killRate': '击杀率',
    'common.share.ai.duelScore': '对枪评分',
  },
  en: {
    'common.nav.library': 'Demo library',
    'common.nav.players': 'Players',
    'common.nav.replay': '3D replay',

    'common.brand.player': 'Player profile',
    'common.brand.version': 'Version 5',

    'doc.library': 'cs-net — Demo library',
    'doc.players': 'cs-net — Players',
    'doc.player': 'cs-net — Player profile',
    'doc.replay': 'cs-net — 3D replay',

    'common.action.cancel': 'Cancel',
    'common.action.delete': 'Delete',
    'common.action.remove': 'Remove',
    'common.action.add': 'Add',
    'common.action.loading': 'Loading…',

    'common.share.btn': 'Share image',
    'common.share.btn.title': 'Export this data as a PNG with the logo and project link',
    'common.share.saving': 'Rendering image…',
    'common.share.done': 'Saved {name}',
    'common.share.fail': 'Export failed: {msg}',

    'common.share.kind.match': 'Match stats',
    'common.share.kind.player': 'Player stats',
    'common.share.project': 'github.com/Gary2005/cs-net',
    'common.share.generated': 'Made with cs-net',
    'common.share.rounds': '{n} rounds',
    'common.share.col.player': 'Player',
    'common.share.col.rounds': 'Rounds',
    'common.share.tile.matches': 'Matches',
    'common.share.tile.hs': 'Headshot',
    'common.share.section.trad': 'Traditional stats',
    'common.share.section.ai': 'AI stats',
    'common.share.recent': 'Last {n} matches',
    'common.share.lastSeen': 'Last seen {date}',

    'common.share.ai.killImpact': 'Kill impact',
    'common.share.ai.roundImpact': 'Round impact',
    'common.share.ai.deathRisk': 'Death risk',
    'common.share.ai.killRate': 'Kill rate',
    'common.share.ai.duelScore': 'Duel score',
  },
};

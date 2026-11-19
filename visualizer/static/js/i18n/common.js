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

    'doc.library': 'cs-net — Demo 库',
    'doc.players': 'cs-net — 玩家',
    'doc.player': 'cs-net — 玩家档案',
    'doc.replay': 'cs-net — 可视化回放',

    'common.action.cancel': '取消',
    'common.action.delete': '删除',
    'common.action.remove': '移除',
    'common.action.add': '添加',
    'common.action.loading': '加载中…',
  },
  en: {
    'common.nav.library': 'Demo library',
    'common.nav.players': 'Players',
    'common.nav.replay': '3D replay',

    'common.brand.player': 'Player profile',

    'doc.library': 'cs-net — Demo library',
    'doc.players': 'cs-net — Players',
    'doc.player': 'cs-net — Player profile',
    'doc.replay': 'cs-net — 3D replay',

    'common.action.cancel': 'Cancel',
    'common.action.delete': 'Delete',
    'common.action.remove': 'Remove',
    'common.action.add': 'Add',
    'common.action.loading': 'Loading…',
  },
};

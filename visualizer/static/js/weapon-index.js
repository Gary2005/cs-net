/**
 * weapon-index.js — 显示名 → 武器模型，以及模型 → 朝向类别。
 *
 * **纯逻辑，不 import three。** 这样 `visualizer/tests/check_weapon_coverage.mjs`
 * 能在 node 里直接跑同一份解析代码，而不是靠"读一遍代码觉得对"。
 *
 * ## 为什么需要一张别名表
 *
 * Valve 的**图标文件名**和**模型文件名**在好几处对不上，而且没有规律可循：
 *
 * | 显示名 | 图标文件 | 模型文件 |
 * |---|---|---|
 * | M4A4 | `weapon_m4a1_png.png` | `weapon_rif_m4a4.vmdl_c` |
 * | Glock-18 | `weapon_glock_png.png` | `weapon_pist_glock18.vmdl_c` |
 * | Incendiary Grenade | `weapon_incgrenade_png.png` | `weapon_incendiarygrenade.vmdl_c` |
 * | M9 Bayonet | `weapon_knife_m9_bayonet_png.png` | `weapon_knife_m9.vmdl_c` |
 * | Navaja Knife | `weapon_knife_gypsy_jackknife_png.png` | `weapon_knife_navaja.vmdl_c` |
 * | Talon Knife | `weapon_knife_widowmaker_png.png` | `weapon_knife_talon.vmdl_c` |
 * | Bowie Knife | `weapon_knife_survival_bowie_png.png` | `weapon_knife_bowie.vmdl_c` |
 *
 * 所以**不能**从图标文件名推模型名。别名表的 target 一律是**模型文件名的短 key**
 * （去掉 `weapon_` 和类别中缀之后的那个），因为模型文件名才是唯一真值。
 *
 * 早先版本用"模糊匹配"（`key.endsWith(dir)`）来兜，结果是 `M4A1-S` 有概率命中
 * M4A4 的模型，而 `Knife` 一个都匹配不上 —— 静默、无报错、只是不显示。
 */

/** `weapon_ak47_png.png` → `ak47`；`weapon_knife_karambit_png.png` → `knife_karambit`。 */
export function iconFileToKey(file) {
  return String(file || '').replace(/^weapon_/, '').replace(/_png\.png$/, '');
}

/**
 * 模型文件名 → 短 key：去掉 `weapon_` 前缀和**类别中缀**。
 *
 * `weapon_rif_ak47` → `ak47`、`weapon_pist_hkp2000` → `hkp2000`、
 * `weapon_knife_m9` → `knife_m9`（`knife` **不当中缀剥掉**，因为它是名字的一部分）、
 * `weapon_smokegrenade` → `smokegrenade`（没有中缀）。
 */
export function modelShortKey(modelKey) {
  return String(modelKey || '')
    .replace(/^weapon_/, '')
    .replace(/^(?:rif|snip|smg|pist|mach|shot)_/, '');
}

/**
 * 图标 key → 模型短 key 的别名。**只列真正对不上的**，
 * 对得上的靠 `modelShortKey` 直接命中。
 */
export const ICON_KEY_ALIAS = {
  // 手枪
  glock: 'glock18',
  // 步枪：M4A4 的图标文件叫 weapon_m4a1（M4A1-S 才是 weapon_m4a1_silencer）
  m4a1: 'm4a4',
  // 投掷物
  incgrenade: 'incendiarygrenade',
  // 刀：图标名和模型名完全不是一回事
  bayonet: 'knife_bayonet',
  knife_m9_bayonet: 'knife_m9',
  knife_survival_bowie: 'knife_bowie',
  knife_gypsy_jackknife: 'knife_navaja',
  knife_widowmaker: 'knife_talon',
};

/**
 * "Knife" 这个显示名**不指定刀型**（demo 里没记录皮肤/刀型时就是这样）。
 * 它不该匹配不上 —— 随便一把刀都比不显示合理。优先默认刀。
 */
const GENERIC_KNIFE_NAMES = new Set(['knife', 'Knife', 'knife_default']);
const PREFERRED_DEFAULT_KNIFE = 'knife_default_ct';

/**
 * 投掷物类名 → 模型短 key（`modelShortKey` 的那个）。
 *
 * **demo 里 `events.grenades[].ty` 是引擎类名，不是显示名。** 实测
 * （`notes/scratch/grenade_type_probe.py`，de_inferno 24 回合 / 327266 个采样点）
 * 只有这 5 种，没有第 6 种：
 *
 * | 类名 | 采样点 |
 * |---|---|
 * | `CSmokeGrenadeProjectile` | 249909 |
 * | `CHEGrenadeProjectile` | 46101 |
 * | `CMolotovProjectile` | 15435 |
 * | `CFlashbangProjectile` | 14456 |
 * | `CDecoyProjectile` | 1365 |
 *
 * 所以**不能**用 `grenadeCategory()` 那套显示名子串匹配来选模型 —— 那条路是给
 * 颜色用的，`'CHEGrenadeProjectile'.includes('HE')` 恰好成立只是运气。
 * 这里用**完整类名精确匹配**，认不出来就返回 null（宁可退回球，也不要挂错模型）。
 */
export const NADE_PROJECTILE_MODEL = {
  CSmokeGrenadeProjectile: 'smokegrenade',
  CHEGrenadeProjectile: 'hegrenade',
  CFlashbangProjectile: 'flashbang',
  CDecoyProjectile: 'decoy',
  // 莫洛托夫和燃烧弹**共用**这一个类名（游戏里是同一个 projectile 类），
  // 光看类名分不出来 —— 见 nadeModelKey()。
  CMolotovProjectile: 'molotov',
};

/** `NADE_PROJECTILE_MODEL` 的值去重 + 燃烧弹（CT 那份）。投掷物一共 6 个模型。 */
export const NADE_MODEL_KEYS = [
  ...new Set([...Object.values(NADE_PROJECTILE_MODEL), 'incendiarygrenade']),
];

/**
 * 投掷物 → 模型短 key。认不出来返回 `null`。
 *
 * ## 莫洛托夫 vs 燃烧弹：只能靠队伍猜，而且**有实测的例外**
 *
 * 两者共用 `CMolotovProjectile`，但模型完全不同：燃烧弹用的是**烟雾弹的网格**
 * （实测两者的 POSITION 包围盒一模一样：0.0527 × 0.1181 × 0.0593 m，只是贴图
 * 不同），而莫洛托夫是自己的瓶子网格（0.0908 × 0.2495 × 0.2097 m）。
 *
 * CS2 里莫洛托夫只有 T 能买、燃烧弹只有 CT 能买，所以按投掷者队伍区分。
 * **但这条规则不是 100%**：把 139 个火焰实例和引爆时刻最近的投掷物配对后
 * （`notes/scratch/molotov_team_probe.py` / `molotov_exception_probe.py`），
 * 135 个吻合、4 个不吻合，其中 2 个是配对误差、2 个是**真的**：
 * 回合 idx=3 有个 CT 丢出了莫洛托夫（火焰 type_name = molotov），
 * 回合 idx=10 有个 T 丢出了燃烧弹。原因是 CS2 允许**捡起地上的投掷物再丢**。
 *
 * 也就是说这个函数大约有 1.4% 的概率选错模型。这是**已知的、有据可查的**误差，
 * 不是没想到 —— 要完全准确就得拿投掷物落点去和 `fire[].inferno_type` 配对
 * （那个字段是游戏自己的真值），代价是要在 trajectories 里耦合火焰数据。
 *
 * @param {string} typeStr demo 的 `events.grenades[].ty`（引擎类名）
 * @param {string|null} team 投掷者在**本回合**的队伍（`'CT'` / `'T'`）
 * @returns {string|null} 模型短 key
 */
export function nadeModelKey(typeStr, team) {
  const key = NADE_PROJECTILE_MODEL[String(typeStr ?? '').trim()];
  if (!key) return null;
  // 只有 CT 那份是燃烧弹；队伍未知（th = -1 等）时退回莫洛托夫
  if (key === 'molotov' && team === 'CT') return 'incendiarygrenade';
  return key;
}

/**
 * 每个类别怎么挂到手上：旋转、偏移、锚点。
 *
 * 放在这个**纯模块**里而不是 `weapon-models.js`，因为后者 import three、
 * node 里跑不了，于是"每个类别都有姿态"这条断言就没法测。
 * 而 `CLASS_ORIENT[cls] || CLASS_ORIENT.rifle` 这种静默兜底正是之前
 * **刀被按步枪姿态挂**（刀竖着插在手里）的成因 —— 现在有测试盯着它。
 *
 * 朝向依据（§9.4.2，65 把逐把量过）：
 *  - 枪械长轴一律 z、枪口 +z、+Y 上 → 不旋转，锚点用**原点**
 *    （AK 包围盒 z∈[−0.296,+0.663]，原点就在握把附近；归零反而前推 0.18 m）
 *  - 刀/投掷物竖直建模（长轴 +Y）→ 绕 X 转 −90° 把 +Y 映射到 +Z，
 *    原点不在握持处，按**包围盒中心**归零
 */
const HALF_PI = Math.PI / 2;

export const CLASS_ORIENT = {
  rifle:   { rot: [0, 0, 0], off: [0.0, 0.02, 0.10], anchor: 'origin' },
  sniper:  { rot: [0, 0, 0], off: [0.0, 0.02, 0.14], anchor: 'origin' },
  smg:     { rot: [0, 0, 0], off: [0.0, 0.02, 0.08], anchor: 'origin' },
  pistol:  { rot: [0, 0, 0], off: [0.0, 0.00, 0.06], anchor: 'origin' },
  // 霰弹枪 / 机枪的建模约定与步枪一致（mag7 枪口 +z、Y 以上 88%；negev 枪口 +z），
  // 但类别独立，这样日志和测试报的是真实类别而不是笼统的 "rifle"
  shotgun: { rot: [0, 0, 0], off: [0.0, 0.02, 0.10], anchor: 'origin' },
  mg:      { rot: [0, 0, 0], off: [0.0, 0.02, 0.12], anchor: 'origin' },
  knife:   { rot: [-HALF_PI, 0, 0], off: [0.0, 0.02, 0.06], anchor: 'centre' },
  grenade: { rot: [-HALF_PI, 0, 0], off: [0.0, 0.00, 0.05], anchor: 'centre' },
  c4:      { rot: [-HALF_PI, 0, 0], off: [0.0, 0.00, 0.08], anchor: 'centre' },
};

/**
 * 模型文件名 → 挂载姿态类别。
 *
 * 用 Valve 自己的**类别中缀**（`rif`/`snip`/`smg`/`pist`/`mach`/`shot`）判断，
 * 而不是从图标 key 反推 —— 早先版本拿图标 key 判断，`weapon_knife_karambit`
 * 被剥成 `karambit`，落进"默认步枪"，刀就按步枪姿态挂了。
 *
 * **返回的是真实类别**（霰弹枪就是 `shotgun`、机枪就是 `mg`），
 * 不合并成 `rifle`。合并会让日志/测试打出 `MAG-7 -> [rifle]` 这种误导性输出，
 * 而这正是排查朝向问题时最需要看的信息。姿态上的共用交给 `CLASS_ORIENT`
 * 去表达（`shotgun`/`mg` 的条目和 `rifle` 一样），两者职责分开。
 */
export function weaponClass(modelKey) {
  const k = String(modelKey || '').replace(/^weapon_/, '');
  if (k.startsWith('knife') || k.startsWith('bayonet')) return 'knife';
  if (k.startsWith('c4')) return 'c4';
  if (/^(?:hegrenade|flashbang|smokegrenade|molotov|incendiarygrenade|decoy)/.test(k)) {
    return 'grenade';
  }
  if (k.startsWith('rif_')) return 'rifle';
  if (k.startsWith('snip_')) return 'sniper';
  if (k.startsWith('smg_')) return 'smg';
  if (k.startsWith('pist_')) return 'pistol';
  // 霰弹枪和机枪的建模约定和步枪完全一致（+Z 前 / +Y 上），但类别照实报
  if (k.startsWith('shot_')) return 'shotgun';
  if (k.startsWith('mach_')) return 'mg';
  // 医疗针是小握持物，按手枪挂更合理
  if (k === 'healthshot') return 'pistol';
  return 'rifle';
}

/**
 * 建立 `显示名 -> manifest 里的武器条目` 索引。
 *
 * @param {Array<{name:string, dir:string, glb:string}>} weaponEntries manifest.weapons
 * @param {Record<string, string|string[]>} iconMap 显示名 -> 图标文件名
 * @returns {{index: Record<string, object>, unresolved: string[], byShortKey: Record<string, object>}}
 *   `unresolved` 是"有图标但没有模型"的显示名 —— 返回而不是吞掉，
 *   这样测试能断言 demo 用的武器全都解析成功。
 *   `byShortKey` 是**模型短 key**（`modelShortKey`）→ 条目，不经过图标表。
 *   投掷物要用它：demo 只给引擎类名，压根没有图标 key 可走
 *   （`NADE_PROJECTILE_MODEL` 的值就是这些短 key）。
 */
export function buildWeaponIndex(weaponEntries, iconMap) {
  const byShort = new Map();
  const byClass = new Map();
  for (const w of weaponEntries || []) {
    if (!w || !w.name) continue;
    byShort.set(modelShortKey(w.name), w);
    const cls = weaponClass(w.name);
    if (!byClass.has(cls)) byClass.set(cls, []);
    byClass.get(cls).push(w);
  }

  const byShortKey = {};
  for (const [short, w] of byShort) {
    byShortKey[short] = { ...w, short, cls: weaponClass(w.name) };
  }

  const index = {};
  const unresolved = [];
  for (const [display, files] of Object.entries(iconMap || {})) {
    const file = Array.isArray(files) ? files[0] : files;
    if (!file) continue;
    const iconKey = iconFileToKey(file);

    // 1) 直接命中  2) 别名
    let hit = byShort.get(iconKey) || byShort.get(ICON_KEY_ALIAS[iconKey]);

    // 3) 通用刀："Knife" 不带刀型，优先默认刀，否则任意一把刀
    if (!hit && GENERIC_KNIFE_NAMES.has(iconKey)) {
      const knives = byClass.get('knife') || [];
      hit = knives.find((w) => modelShortKey(w.name) === PREFERRED_DEFAULT_KNIFE)
        || knives[0];
    }

    if (hit) {
      const entry = { ...hit, short: modelShortKey(hit.name), cls: weaponClass(hit.name) };
      index[display] = entry;
      // 也登记**内部 key**（图标文件名里那个）。demoparser2 有时会漏出内部名
      // 而不是显示名 —— 这份 demo 里就有 `knife_skeleton`（图标表里叫
      // "Skeleton Knife"）。不登记的话这种条目图标和模型都会缺。
      if (!(iconKey in index)) index[iconKey] = entry;
    } else {
      unresolved.push(display);
    }
  }
  return { index, unresolved, byShortKey };
}

// Verify the zh/en translation tables and that the pages actually use them.
//
// Five failure modes this catches, all of which look fine in a browser until
// someone switches the language:
//
//   * a key exists in zh but not en (or the reverse) -- the English page shows
//     the Chinese fallback or the raw key;
//   * two page tables define the same key -- Object.assign silently keeps the
//     last one, so one page quietly wins;
//   * an HTML `data-i18n` key that no table defines -- the element keeps its
//     Chinese text, i.e. a Chinese label in the middle of the English UI;
//   * Chinese text/attributes in a template that no `data-i18n*` covers --
//     the same thing, but the element would never translate at all;
//   * inline Chinese drift: the default text in the template no longer equals
//     the zh table's value, so the no-JS page and the translated page disagree.
//
// Usage: node visualizer/tests/check_i18n.mjs
import { readFileSync } from 'node:fs';
import { join, resolve, dirname } from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';

const HERE = dirname(fileURLToPath(import.meta.url));
const ROOT = resolve(HERE, '..', '..');
const DICT_DIR = join(ROOT, 'visualizer', 'static', 'js', 'i18n');
const MODULES = ['common', 'replay', 'library', 'players'];

const TEMPLATES = [
  'app/templates/library.html',
  'app/templates/players.html',
  'app/templates/player.html',
  'visualizer/templates/index.html',
];

// JS files that call t(). The dictionaries themselves are checked above and
// must not be scanned for `t('...')` (they contain the values, not calls).
const JS_FILES = [
  'visualizer/static/js/main.js',
  'visualizer/static/js/ai-chart.js',
  // 分享图：`t` 是参数（不 import），但键名仍然是字面量，所以要一起查
  'visualizer/static/js/share-card.js',
  'app/static/js/library.js',
  'app/static/js/players.js',
  'app/static/js/player.js',
];

const CJK = /[\u4e00-\u9fff]/;

let passed = 0;
const failed = [];
function check(name, ok, detail = '') {
  if (ok) {
    passed++;
    console.log(`  \u2713 ${name}`);
  } else {
    failed.push(name);
    console.log(`  \u2717 ${name}${detail ? `\n      ${detail}` : ''}`);
  }
}

// ── 1. load the tables ───────────────────────────────────────────────────────

const tables = {};
for (const name of MODULES) {
  const mod = await import(pathToFileURL(join(DICT_DIR, `${name}.js`)).href);
  tables[name] = mod.default || {};
}

const merged = { zh: {}, en: {} };
const owners = { zh: {}, en: {} };
for (const name of MODULES) {
  const table = tables[name];
  for (const lang of ['zh', 'en']) {
    for (const [key, value] of Object.entries(table[lang] || {})) {
      merged[lang][key] = value;
      (owners[lang][key] ||= []).push(name);
    }
  }
}

console.log('\nkey parity');
for (const name of MODULES) {
  const zh = Object.keys(tables[name].zh || {});
  const en = Object.keys(tables[name].en || {});
  const missingEn = zh.filter((k) => !(k in (tables[name].en || {})));
  const extraEn = en.filter((k) => !(k in (tables[name].zh || {})));
  check(`${name}.js: zh and en have the same keys (${zh.length})`,
    missingEn.length === 0 && extraEn.length === 0,
    [missingEn.length ? `missing in en: ${missingEn.join(', ')}` : '',
     extraEn.length ? `not in zh: ${extraEn.join(', ')}` : ''].filter(Boolean).join('\n      '));
}

const zhKeys = Object.keys(merged.zh);
const enKeys = Object.keys(merged.en);
check('merged tables have the same keys',
  zhKeys.length === enKeys.length && zhKeys.every((k) => k in merged.en),
  `zh ${zhKeys.length}, en ${enKeys.length}`);

console.log('\nduplicate keys across page tables');
for (const lang of ['zh', 'en']) {
  for (const [key, where] of Object.entries(owners[lang])) {
    if (where.length > 1) {
      check(`${key} is defined once`, false, `defined in ${where.join(', ')}`);
    }
  }
}
check('no key is defined by two page tables',
  Object.values(owners.zh).every((w) => w.length === 1));

console.log('\nuntranslated english');
const untranslated = enKeys.filter((k) => CJK.test(merged.en[k]));
check('no english value still contains Chinese', untranslated.length === 0,
  untranslated.slice(0, 12).join(', '));
const emptyEn = enKeys.filter((k) => merged.en[k] === '');
check('no english value is empty (except deliberate suffixes)',
  emptyEn.length === 0, emptyEn.join(', '));

// ── 2. tiny HTML scanner ────────────────────────────────────────────────────
//
// Templates contain Jinja (`{% ... %}` / `{{ ... }}`). They are stripped out of
// text before it is compared, because what the browser receives has already
// been rendered; the tags themselves are not page copy.

const VOID = new Set(['area', 'base', 'br', 'col', 'embed', 'hr', 'img', 'input',
  'link', 'meta', 'param', 'source', 'track', 'wbr']);

function tokenize(html) {
  const out = [];
  let i = 0;
  while (i < html.length) {
    const lt = html.indexOf('<', i);
    if (lt < 0) { out.push({ type: 'text', text: html.slice(i) }); break; }
    if (lt > i) out.push({ type: 'text', text: html.slice(i, lt) });
    if (html.startsWith('<!--', lt)) {
      const end = html.indexOf('-->', lt + 4);
      i = end < 0 ? html.length : end + 3;
      continue;
    }
    if (html.startsWith('<!', lt) || html.startsWith('<?', lt)) {
      const end = html.indexOf('>', lt + 2);
      i = end < 0 ? html.length : end + 1;
      continue;
    }
    const closing = html[lt + 1] === '/';
    let j = lt + (closing ? 2 : 1);
    let quote = null;
    while (j < html.length) {
      const ch = html[j];
      if (quote) { if (ch === quote) quote = null; }
      else if (ch === '"' || ch === "'") quote = ch;
      else if (ch === '>') break;
      j++;
    }
    const raw = html.slice(lt, j + 1);
    const m = raw.match(/^<\/?\s*([a-zA-Z][\w:-]*)([\s\S]*?)\/?\s*>$/);
    if (!m) { i = j + 1; continue; }
    const attrs = {};
    const body = m[2];
    const re = /([a-zA-Z_:@][-\w:.]*)(?:\s*=\s*(?:"([^"]*)"|'([^']*)'|([^\s"'=<>`]+)))?/g;
    let a;
    while ((a = re.exec(body))) attrs[a[1].toLowerCase()] = a[2] ?? a[3] ?? a[4] ?? '';
    out.push({ type: 'tag', name: m[1].toLowerCase(), closing, attrs, selfClose: /\/\s*>$/.test(raw) });
    i = j + 1;
  }
  return out;
}

function stripJinja(text) {
  return text.replace(/\{%[\s\S]*?%\}/g, '').replace(/\{\{[\s\S]*?\}\}/g, '');
}

function norm(s) {
  return String(s).replace(/\s+/g, ' ').trim();
}

function scanTemplate(file) {
  const html = readFileSync(join(ROOT, file), 'utf8');
  const tokens = tokenize(html);
  const problems = { missingKeys: [], unusedAttr: [], leaks: [], drift: [] };
  const stack = [];
  let inRaw = null;              // <script> / <style> contents are not page copy

  for (let ti = 0; ti < tokens.length; ti++) {
    const tok = tokens[ti];
    if (tok.type === 'text') {
      if (inRaw) continue;
      const text = stripJinja(tok.text);
      if (!CJK.test(text)) continue;
      const nearest = stack[stack.length - 1];
      const covered = nearest && (nearest.attrs['data-i18n'] || nearest.attrs['data-i18n-html']);
      const ancestorHtml = stack.some((el) => el.attrs['data-i18n-html']);
      if (!covered && !ancestorHtml) {
        problems.leaks.push(`${norm(text).slice(0, 48)}`);
      }
      continue;
    }

    if (inRaw) {
      if (tok.closing && tok.name === inRaw) inRaw = null;
      continue;
    }
    if (tok.name === 'script' || tok.name === 'style') {
      if (!tok.closing) inRaw = tok.name;
      continue;
    }

    if (tok.closing) {
      for (let s = stack.length - 1; s >= 0; s--) {
        if (stack[s].name === tok.name) { stack.length = s; break; }
      }
      continue;
    }
    if (!tok.selfClose && !VOID.has(tok.name)) stack.push(tok);
    if (VOID.has(tok.name)) continue;

    const el = tok;
    for (const [attr, value] of Object.entries(el.attrs)) {
      if (attr.startsWith('data-i18n')) continue;
      if (!CJK.test(value)) continue;
      const cover = `data-i18n-${attr}`;
      if (!(cover in el.attrs)) problems.leaks.push(`<${el.name} ${attr}="${norm(value).slice(0, 36)}">`);
    }

    // key existence for every data-i18n* attribute
    for (const attr of Object.keys(el.attrs)) {
      if (!attr.startsWith('data-i18n')) continue;
      const key = el.attrs[attr];
      if (!key) { problems.missingKeys.push(`empty ${attr} on <${el.name}>`); continue; }
      if (!(key in merged.zh)) problems.missingKeys.push(key);
      else if (!(key in merged.en)) problems.missingKeys.push(`${key} (en)`);
    }

    // inline text must equal the zh value (only for elements whose whole
    // content is the text -- an element with child tags is a data-i18n-html case)
    const key = el.attrs['data-i18n'];
    if (key && merged.zh[key] !== undefined) {
      const next = tokens[ti + 1];
      const after = tokens[ti + 2];
      if (next && next.type === 'text' && after && after.type === 'tag'
          && after.closing && after.name === el.name) {
        const inline = norm(stripJinja(next.text));
        if (inline && !/\{\{|\{%/.test(next.text) && inline !== norm(merged.zh[key])) {
          problems.drift.push(`${key}: template "${inline.slice(0, 40)}" != zh "${norm(merged.zh[key]).slice(0, 40)}"`);
        }
      }
    }
  }

  return problems;
}

console.log('\ntemplates');
for (const file of TEMPLATES) {
  const p = scanTemplate(file);
  check(`${file}: every data-i18n key exists`, p.missingKeys.length === 0, p.missingKeys.join('\n      '));
  check(`${file}: no untranslated Chinese left`, p.leaks.length === 0, p.leaks.join('\n      '));
  check(`${file}: inline defaults match the zh table`, p.drift.length === 0, p.drift.join('\n      '));
}

// ── 3. keys used from JS ────────────────────────────────────────────────────

console.log('\njavascript');
const used = new Map();          // key -> [file]
for (const file of JS_FILES) {
  const src = readFileSync(join(ROOT, file), 'utf8');
  for (const m of src.matchAll(/\bt\(\s*(['"`])([^'"`]+)\1/g)) {
    const key = m[2];
    if (!used.has(key)) used.set(key, []);
    used.get(key).push(file);
  }
}
const missing = [...used.keys()].filter((k) => !(k in merged.zh) || !(k in merged.en));
check(`every t() key exists in both tables (${used.size} keys used)`, missing.length === 0,
  missing.slice(0, 20).join('\n      '));

const nsOK = [...used.keys()].every((k) => /^(replay|lib|player|common|doc)\./.test(k));
check('every t() key is namespaced', nsOK,
  [...used.keys()].filter((k) => !/^(replay|lib|player|common|doc)\./.test(k)).slice(0, 10).join(', '));

// ── 4. coverage report ──────────────────────────────────────────────────────

console.log('\ncoverage');
const htmlKeys = new Set();
for (const file of TEMPLATES) {
  const html = readFileSync(join(ROOT, file), 'utf8');
  for (const m of html.matchAll(/data-i18n(?:-html|-title|-placeholder|-aria-label)?="([^"]+)"/g)) {
    htmlKeys.add(m[1]);
  }
}
const unused = zhKeys.filter((k) => !htmlKeys.has(k) && !used.has(k));
console.log(`      keys: ${zhKeys.length}  used in html: ${htmlKeys.size}  used in js: ${used.size}`);
if (unused.length) console.log(`      (defined but not used anywhere: ${unused.length}) ${unused.slice(0, 10).join(', ')}`);

console.log(`\n${passed} passed, ${failed.length} failed`);
if (failed.length) {
  for (const name of failed) console.log(`  FAILED: ${name}`);
  process.exit(1);
}

// Verify that every named import actually exists as an export in the target
// module. A syntax check cannot see this: `import { foo } from './x.js'` parses
// fine even when x.js never exports foo -- and in a browser it is a hard
// runtime error that blanks the page.
//
// This is a static check (no execution), so it works on browser-only modules.
//
// Usage: node visualizer/tests/check_imports.mjs [dir]
import { readFileSync, readdirSync, statSync, existsSync } from 'node:fs';
import { join, extname, dirname, resolve, relative } from 'node:path';
import vm from 'node:vm';

const root = resolve(process.argv[2] || 'visualizer/static/js');

const files = [];
(function walk(d) {
  for (const e of readdirSync(d)) {
    const p = join(d, e);
    if (statSync(p).isDirectory()) walk(p);
    else if (extname(p) === '.js') files.push(p);
  }
})(root);

/** Collect `export function/const/class/let/var NAME` and `export { a, b }`. */
function exportsOf(src) {
  const names = new Set();
  for (const m of src.matchAll(/^\s*export\s+(?:async\s+)?(?:function|const|let|var|class)\s+([A-Za-z_$][\w$]*)/gm)) {
    names.add(m[1]);
  }
  for (const m of src.matchAll(/^\s*export\s*\{([^}]*)\}/gm)) {
    for (const part of m[1].split(',')) {
      const t = part.trim();
      if (!t) continue;
      // `a as b` exports the name b
      const as = t.match(/\bas\s+([A-Za-z_$][\w$]*)$/);
      names.add(as ? as[1] : t.split(/\s+/)[0]);
    }
  }
  if (/^\s*export\s+default\b/m.test(src)) names.add('default');
  return names;
}

/** Bare specifiers (three, three/addons/...) are resolved by the importmap. */
function isBare(spec) {
  return !spec.startsWith('.') && !spec.startsWith('/');
}

let problems = 0;
const cache = new Map();
function exportsCached(path) {
  if (!cache.has(path)) {
    cache.set(path, existsSync(path) ? exportsOf(readFileSync(path, 'utf8')) : null);
  }
  return cache.get(path);
}

for (const f of files.sort()) {
  const src = readFileSync(f, 'utf8');
  const specs = [];
  // `import ... from 'spec'` and `export ... from 'spec'`
  for (const m of src.matchAll(/^\s*(?:import|export)[\s\S]*?from\s+['"]([^'"]+)['"]/gm)) {
    specs.push(m[1]);
  }
  for (const spec of specs) {
    if (isBare(spec)) continue;                 // importmap territory
    const target = resolve(dirname(f), spec);
    const ex = exportsCached(target);
    if (ex === null) {
      console.log(`  MISSING FILE ${relative(root, f)} -> ${spec}`);
      problems++;
      continue;
    }
    // find the names this file imports from that spec
    const re = new RegExp(
      `^\\s*(?:import|export)\\s*\\{([^}]*)\\}\\s*from\\s*['"]${spec.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')}['"]`,
      'gm');
    for (const m of src.matchAll(re)) {
      for (const part of m[1].split(',')) {
        const t = part.trim();
        if (!t) continue;
        const name = t.split(/\s+as\s+/)[0].trim();
        if (!name) continue;
        if (!ex.has(name)) {
          console.log(`  BAD IMPORT  ${relative(root, f)}`);
          console.log(`              '${name}' is not exported by ${spec}`);
          problems++;
        }
      }
    }
  }
}

console.log(`\n  ${files.length} modules scanned, ${problems} problem(s)`);
process.exit(problems ? 1 : 0);

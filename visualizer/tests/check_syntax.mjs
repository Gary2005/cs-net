// Syntax-check the visualizer's ES modules without executing them.
//
// These are browser modules: they import bare 'three' and touch `document`, so
// they cannot be imported in Node. `vm.SourceTextModule` parses without
// evaluating, which is exactly the check wanted -- a typo in a rarely-hit
// branch should fail here rather than in the browser console.
//
// Usage: node --experimental-vm-modules visualizer/tests/check_syntax.mjs [dir]
import { readFileSync, readdirSync, statSync } from 'node:fs';
import { join, extname, relative } from 'node:path';
import vm from 'node:vm';

// `vm.SourceTextModule` only exists behind a flag. Without this guard a forgotten
// flag reports "0 ok, N failed" with "SourceTextModule is not a constructor" on
// every file -- which reads exactly like a repo-wide syntax catastrophe instead
// of a missing CLI flag. Fail with the fix instead.
if (typeof vm.SourceTextModule !== 'function') {
  console.error(
    'check_syntax.mjs needs the vm modules flag:\n' +
    '  node --experimental-vm-modules visualizer/tests/check_syntax.mjs\n' +
    '(vm.SourceTextModule is unavailable, so nothing was checked.)');
  process.exit(2);
}

const root = process.argv[2] || 'visualizer/static/js';
const files = [];
(function walk(d) {
  for (const e of readdirSync(d)) {
    const p = join(d, e);
    if (statSync(p).isDirectory()) walk(p);
    else if (extname(p) === '.js') files.push(p);
  }
})(root);

let bad = 0;
for (const f of files.sort()) {
  const src = readFileSync(f, 'utf8');
  try {
    new vm.SourceTextModule(src, { identifier: f });
    console.log(`  OK   ${relative(root, f)}`);
  } catch (e) {
    bad++;
    console.log(`  FAIL ${relative(root, f)}`);
    console.log(`       ${e.message}`);
  }
}
console.log(`\n  ${files.length - bad} ok, ${bad} failed`);
process.exit(bad ? 1 : 0);

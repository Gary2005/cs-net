/* CS-NET v4 — showcase site interactions (vanilla, no deps) */
(function () {
  'use strict';

  var LS_KEY = 'csnet-lang';

  /* ── i18n ───────────────────────────────────────────── */
  var LANG = (function () {
    var saved = null;
    try { saved = localStorage.getItem(LS_KEY); } catch (e) {}
    if (saved === 'en' || saved === 'zh') return saved;
    return (navigator.language || 'en').toLowerCase().indexOf('zh') === 0 ? 'zh' : 'en';
  })();

  function setLang(lang) {
    LANG = lang;
    try { localStorage.setItem(LS_KEY, lang); } catch (e) {}
    applyLang(lang);
  }

  function applyLang(lang) {
    document.documentElement.lang = lang;

    // Swap every element that carries both translations (static content = EN)
    // <title> / <meta> are handled explicitly below (no innerHTML on void elements)
    document.querySelectorAll('[data-en][data-zh]:not(title):not(meta)').forEach(function (el) {
      el.innerHTML = lang === 'zh' ? el.getAttribute('data-zh') : el.getAttribute('data-en');
    });

    // <title> + meta description
    var t = document.querySelector('title');
    if (t) t.textContent = t.getAttribute(lang === 'zh' ? 'data-zh' : 'data-en');
    var m = document.querySelector('meta[name="description"]');
    if (m) m.setAttribute('content', m.getAttribute(lang === 'zh' ? 'data-zh' : 'data-en'));

    // Toggle buttons
    document.querySelectorAll('#lang-toggle button').forEach(function (b) {
      b.classList.toggle('active', b.getAttribute('data-lang') === lang);
    });

    // Copy buttons: reflect current language (unless mid-copied state)
    document.querySelectorAll('.copy[data-copy]').forEach(function (b) {
      if (b.classList.contains('copied')) {
        b.textContent = lang === 'zh' ? '已复制 ✓' : 'copied ✓';
      } else {
        b.textContent = lang === 'zh' ? '复制' : 'copy';
      }
    });
  }

  // Nav scroll state
  var nav = document.getElementById('nav');
  function onScroll() {
    if (!nav) return;
    nav.classList.toggle('scrolled', window.scrollY > 24);
  }
  window.addEventListener('scroll', onScroll, { passive: true });
  onScroll();

  // Language toggle
  var toggle = document.getElementById('lang-toggle');
  if (toggle) {
    toggle.addEventListener('click', function (e) {
      var btn = e.target.closest('button[data-lang]');
      if (!btn) return;
      var lang = btn.getAttribute('data-lang');
      if (lang !== LANG) setLang(lang);
    });
  }

  // Reveal-on-scroll
  var revealEls = document.querySelectorAll('.section, .gallery, .cards, .model-grid, .codeblock, .doc-grid');
  if ('IntersectionObserver' in window) {
    var io = new IntersectionObserver(function (entries) {
      entries.forEach(function (e) {
        if (e.isIntersecting) {
          e.target.classList.add('in');
          io.unobserve(e.target);
        }
      });
    }, { threshold: 0.08 });
    revealEls.forEach(function (el) {
      el.classList.add('reveal');
      io.observe(el);
    });
  } else {
    revealEls.forEach(function (el) { el.classList.add('in'); });
  }

  // Copy buttons on code blocks
  document.querySelectorAll('.copy[data-copy]').forEach(function (btn) {
    btn.addEventListener('click', function () {
      var pre = btn.closest('.codeblock').querySelector('pre');
      var text = pre ? pre.innerText : '';
      function done() {
        btn.textContent = LANG === 'zh' ? '已复制 ✓' : 'copied ✓';
        btn.classList.add('copied');
        setTimeout(function () {
          btn.textContent = LANG === 'zh' ? '复制' : 'copy';
          btn.classList.remove('copied');
        }, 1600);
      }
      if (navigator.clipboard && window.isSecureContext) {
        navigator.clipboard.writeText(text).then(done, function () { fallback(); });
      } else { fallback(); }
      function fallback() {
        var ta = document.createElement('textarea');
        ta.value = text;
        ta.style.position = 'fixed';
        ta.style.opacity = '0';
        document.body.appendChild(ta);
        ta.select();
        try { document.execCommand('copy'); done(); } catch (e) {}
        document.body.removeChild(ta);
      }
    });
  });

  // Apply initial language (after listeners are attached)
  applyLang(LANG);
})();

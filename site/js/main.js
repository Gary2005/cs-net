/* CS-NET v4 — showcase site interactions (vanilla, no deps) */
(function () {
  'use strict';

  // Sticky nav: add .scrolled after scrolling past hero
  var nav = document.getElementById('nav');
  function onScroll() {
    if (!nav) return;
    nav.classList.toggle('scrolled', window.scrollY > 24);
  }
  window.addEventListener('scroll', onScroll, { passive: true });
  onScroll();

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
  var buttons = document.querySelectorAll('.copy[data-copy]');
  buttons.forEach(function (btn) {
    btn.addEventListener('click', function () {
      var pre = btn.closest('.codeblock').querySelector('pre');
      var text = pre ? pre.innerText : '';
      function done() {
        btn.textContent = 'copied ✓';
        btn.classList.add('copied');
        setTimeout(function () {
          btn.textContent = 'copy';
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
})();

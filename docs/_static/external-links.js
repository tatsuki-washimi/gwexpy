(function (root, factory) {
  if (typeof module === "object" && module.exports) {
    module.exports = factory();
    return;
  }

  root.GWDocsExternalLinks = factory();
})(typeof globalThis !== "undefined" ? globalThis : this, function () {
  "use strict";

  function getHref(anchor) {
    if (!anchor || typeof anchor.getAttribute !== "function") {
      return null;
    }

    var href = anchor.getAttribute("href");
    if (typeof href !== "string") {
      return null;
    }

    href = href.trim();
    return href === "" ? null : href;
  }

  function isCrossOriginHttpUrl(href, baseUrl) {
    if (typeof href !== "string" || typeof baseUrl !== "string") {
      return false;
    }

    try {
      var resolvedUrl = new URL(href, baseUrl);
      var resolvedBaseUrl = new URL(baseUrl);
      var protocol = resolvedUrl.protocol.toLowerCase();

      if (protocol !== "http:" && protocol !== "https:") {
        return false;
      }

      return resolvedUrl.origin !== resolvedBaseUrl.origin;
    } catch (error) {
      return false;
    }
  }

  function mergeRel(existingRel) {
    var tokens = typeof existingRel === "string" ? existingRel.split(/\s+/) : [];
    var orderedTokens = [];
    var seen = new Set();

    function addToken(token) {
      var normalized = String(token || "").trim().toLowerCase();
      if (normalized === "" || seen.has(normalized)) {
        return;
      }

      seen.add(normalized);
      orderedTokens.push(normalized);
    }

    tokens.forEach(addToken);
    addToken("noopener");
    addToken("noreferrer");

    return orderedTokens.join(" ");
  }

  function decorateExternalLink(anchor, baseUrl) {
    var href = getHref(anchor);
    if (!href || !isCrossOriginHttpUrl(href, baseUrl)) {
      return false;
    }

    if (typeof anchor.setAttribute !== "function") {
      return false;
    }

    anchor.setAttribute("target", "_blank");
    anchor.setAttribute("rel", mergeRel(anchor.getAttribute("rel")));
    return true;
  }

  function enhanceExternalLinks(container, baseUrl) {
    if (!container || typeof container.querySelectorAll !== "function") {
      return 0;
    }

    var decoratedCount = 0;
    container.querySelectorAll("a[href]").forEach(function (anchor) {
      if (decorateExternalLink(anchor, baseUrl)) {
        decoratedCount += 1;
      }
    });

    return decoratedCount;
  }

  if (
    typeof document !== "undefined" &&
    document &&
    typeof document.addEventListener === "function" &&
    typeof window !== "undefined" &&
    window &&
    window.location &&
    typeof window.location.href === "string"
  ) {
    document.addEventListener("DOMContentLoaded", function () {
      enhanceExternalLinks(document, window.location.href);
    });
  }

  return {
    decorateExternalLink: decorateExternalLink,
    enhanceExternalLinks: enhanceExternalLinks,
    getHref: getHref,
    isCrossOriginHttpUrl: isCrossOriginHttpUrl,
    mergeRel: mergeRel,
  };
});

document.addEventListener("DOMContentLoaded", function () {
  function fallbackCopy(text) {
    var textarea = document.createElement("textarea");
    textarea.value = text;
    textarea.setAttribute("readonly", "");
    textarea.style.position = "absolute";
    textarea.style.left = "-9999px";
    document.body.appendChild(textarea);
    textarea.select();
    try {
      document.execCommand("copy");
    } finally {
      document.body.removeChild(textarea);
    }
  }

  document.querySelectorAll(".highlight").forEach(function (block) {
    var pre = block.querySelector("pre");
    if (!pre || block.querySelector(".gw-copy-btn")) {
      return;
    }

    block.classList.add("gw-copy-wrap");

    var button = document.createElement("button");
    button.className = "gw-copy-btn";
    button.type = "button";
    button.setAttribute("aria-label", "Copy code");
    button.title = "Copy code";
    button.textContent = "Copy";

    button.addEventListener("click", async function () {
      var text = pre.innerText;
      try {
        if (navigator.clipboard && navigator.clipboard.writeText) {
          await navigator.clipboard.writeText(text);
        } else {
          fallbackCopy(text);
        }
        button.textContent = "Copied";
        window.setTimeout(function () {
          button.textContent = "Copy";
        }, 1500);
      } catch (err) {
        fallbackCopy(text);
        button.textContent = "Copied";
        window.setTimeout(function () {
          button.textContent = "Copy";
        }, 1500);
      }
    });

    block.appendChild(button);
  });
});

// Filtering, approval state and export for the image summary report.
//
// The script is inlined into the page by render.py, so it must never contain the
// literal closing script tag, must make no external request, and must run from a
// plain <script> element with no build step: no modules, no arrow functions.
(function () {
  "use strict";

  var PRUNE_AFTER_DAYS = 30;
  var DAY_MS = 86400000;
  var KEY_PREFIX = "pytest-pyvista:approvals:";
  var STALE_NOTICE =
    "Approvals from an earlier run of this suite were not restored: this report describes different images.";

  var runId = document.documentElement.getAttribute("data-run-id");
  var storeKey = KEY_PREFIX + runId;
  // The manifest text escapes <, > and & as \uXXXX; JSON.parse decodes them itself.
  var manifest = JSON.parse(document.getElementById("manifest").textContent);
  var byKey = {};
  manifest.records.forEach(function (record) {
    byKey[record.key] = record;
  });

  var cards = Array.prototype.slice.call(
    document.querySelectorAll("article.card"),
  );
  var filters = Array.prototype.slice.call(
    document.querySelectorAll(".status-filter"),
  );
  var search = document.getElementById("search");
  var sort = document.getElementById("sort");
  var count = document.getElementById("count");
  var notice = document.getElementById("notice");
  var acceptNew = document.getElementById("accept-new");

  // localStorage throws rather than degrading when storage is disabled, full, or
  // blocked by a privacy setting. Every access goes through these three helpers so
  // that a browser without usable storage still filters, sorts and exports.
  function readEntry(key) {
    var saved;
    try {
      saved = JSON.parse(localStorage.getItem(key));
    } catch (err) {
      return null;
    }
    return saved && typeof saved === "object" ? saved : null;
  }

  function writeEntry(key, value) {
    try {
      localStorage.setItem(key, JSON.stringify(value));
    } catch (err) {
      // Nothing to do: the selection stays live in the page and still exports.
    }
  }

  function removeEntry(key) {
    try {
      localStorage.removeItem(key);
    } catch (err) {
      // As above; a key that cannot be removed is not worth failing the page over.
    }
  }

  // Storage.key/length rather than Object.keys(localStorage): it is the interface
  // Storage actually defines, and it snapshots into an array so that callers may
  // remove keys while iterating.
  function approvalKeys() {
    var keys = [];
    var index;
    var key;
    try {
      for (index = 0; index < localStorage.length; index += 1) {
        key = localStorage.key(index);
        if (key !== null && key.indexOf(KEY_PREFIX) === 0) {
          keys.push(key);
        }
      }
    } catch (err) {
      return [];
    }
    return keys;
  }

  function pruneOldKeys() {
    var cutoff = Date.now() - PRUNE_AFTER_DAYS * DAY_MS;
    approvalKeys().forEach(function (key) {
      var saved = readEntry(key);
      if (!saved || !saved.saved_at || saved.saved_at < cutoff) {
        removeEntry(key);
      }
    });
  }

  function restore() {
    var saved = readEntry(storeKey);
    var stored = saved && saved.run_id === runId ? saved.approved || {} : {};
    var approved = {};
    Object.keys(stored).forEach(function (key) {
      // Drop anything the current report has no approvable card for, so that the
      // footer count and the exported payload cannot disagree.
      if (stored[key] === true && byKey[key]) {
        approved[key] = true;
      }
    });
    return approved;
  }

  function save(approved) {
    writeEntry(storeKey, {
      run_id: runId,
      // Identifies the suite, so that state belonging to an unrelated report is
      // never mistaken for an earlier run of this one. See earlierRunWasDropped.
      cache_dir: manifest.cache_dir,
      saved_at: Date.now(),
      approved: approved,
    });
  }

  // True only when this report really did leave selections behind: state saved by a
  // different run of the same suite, still within the pruning window, that held at
  // least one approval. Merely finding some other report's key is not enough --
  // that would put the notice on every report of every project the reader owns.
  function earlierRunWasDropped() {
    var found = false;
    approvalKeys().forEach(function (key) {
      var saved = key === storeKey ? null : readEntry(key);
      if (
        !saved ||
        saved.run_id === runId ||
        saved.cache_dir !== manifest.cache_dir
      ) {
        return;
      }
      if (Object.keys(saved.approved || {}).length > 0) {
        found = true;
      }
    });
    return found;
  }

  function showNotice(text) {
    var dismiss = document.createElement("button");
    dismiss.type = "button";
    dismiss.textContent = "Dismiss";
    dismiss.addEventListener("click", function () {
      notice.hidden = true;
    });
    notice.textContent = text;
    notice.appendChild(dismiss);
    notice.hidden = false;
  }

  // Pruning runs before anything reads storage: an expired key must not be restored,
  // and must not raise a notice about state that is being discarded for age anyway.
  pruneOldKeys();
  var approved = restore();
  if (Object.keys(approved).length === 0 && earlierRunWasDropped()) {
    showNotice(STALE_NOTICE);
  }

  // One entry per card that carries a live checkbox. Cards already written to the
  // cache have none, and are therefore never approvable.
  var approvable = [];
  cards.forEach(function (card) {
    var box = card.querySelector("label.approve input[type=checkbox]");
    if (box) {
      approvable.push({
        card: card,
        box: box,
        key: card.getAttribute("data-key"),
      });
    }
  });

  function selectedKeys() {
    return approvable
      .filter(function (entry) {
        return approved[entry.key] === true && byKey[entry.key];
      })
      .map(function (entry) {
        return entry.key;
      });
  }

  function refreshCount() {
    var total = selectedKeys().length;
    count.textContent =
      total + (total === 1 ? " image" : " images") + " selected for approval";
  }

  function setApproved(entry, checked) {
    entry.box.checked = checked;
    if (checked) {
      approved[entry.key] = true;
    } else {
      delete approved[entry.key];
    }
  }

  function applyFilters() {
    var wanted = {};
    var term = search.value.trim().toLowerCase();
    filters.forEach(function (box) {
      if (box.checked) {
        wanted[box.value] = true;
      }
    });
    cards.forEach(function (card) {
      var visible =
        wanted[card.getAttribute("data-status")] === true &&
        (term === "" || card.getAttribute("data-name").indexOf(term) !== -1);
      card.classList.toggle("hidden", !visible);
    });
  }

  function applySort() {
    var mode = sort.value;
    var main = document.getElementById("cards");
    cards
      .slice()
      .sort(function (a, b) {
        var byName = a
          .getAttribute("data-name")
          .localeCompare(b.getAttribute("data-name"));
        if (mode === "name") {
          return byName;
        }
        // A card with no comparable error carries data-error="0", which would sink
        // it beneath every passing test under the error-descending order below --
        // and those cards (`new` above all) are the ones most likely to want
        // approving. Float them to the top instead.
        var aMissing = a.getAttribute("data-missing-error") === "1";
        var bMissing = b.getAttribute("data-missing-error") === "1";
        if (aMissing !== bMissing) {
          return aMissing ? -1 : 1;
        }
        // data-error is always a finite number, so this comparator is total; the
        // name breaks ties, which keeps the order the same on every browser.
        return (
          parseFloat(b.getAttribute("data-error")) -
            parseFloat(a.getAttribute("data-error")) || byName
        );
      })
      .forEach(function (card) {
        main.appendChild(card);
      });
  }

  function exportApprovals() {
    var payload = {
      // Echoed from the manifest rather than hardcoded: Python owns the schema
      // version, and the approve CLI validates the export against that constant.
      schema_version: manifest.schema_version,
      run_id: runId,
      exported_at: new Date().toISOString(),
      cache_dir: manifest.cache_dir,
      image_format: manifest.image_format,
      approved: selectedKeys().map(function (key) {
        var record = byKey[key];
        return {
          test_name: record.test_name,
          image_name: record.image_name,
          call_index: record.call_index,
          status: record.status,
          source: record.source,
          destination: record.destination,
        };
      }),
    };
    var blob = new Blob([JSON.stringify(payload, null, 2)], {
      type: "application/json",
    });
    var url = URL.createObjectURL(blob);
    var link = document.createElement("a");
    link.href = url;
    link.download = "approvals.json";
    link.rel = "noopener";
    // Attached before clicking and revoked on a later turn of the event loop:
    // a detached anchor does not click everywhere, and revoking the object URL
    // in the same turn cancels the download in some browsers.
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    setTimeout(function () {
      URL.revokeObjectURL(url);
    }, 1000);
  }

  approvable.forEach(function (entry) {
    entry.box.checked = approved[entry.key] === true;
    entry.box.addEventListener("change", function () {
      setApproved(entry, entry.box.checked);
      save(approved);
      refreshCount();
    });
  });

  var newEntries = approvable.filter(function (entry) {
    return entry.card.getAttribute("data-status") === "new";
  });
  if (newEntries.length) {
    acceptNew.hidden = false;
  }

  acceptNew.addEventListener("click", function () {
    newEntries.forEach(function (entry) {
      setApproved(entry, true);
    });
    save(approved);
    refreshCount();
  });

  document.getElementById("export").addEventListener("click", exportApprovals);
  filters.forEach(function (box) {
    box.addEventListener("change", applyFilters);
  });
  search.addEventListener("input", applyFilters);
  sort.addEventListener("change", applySort);

  applyFilters();
  applySort();
  refreshCount();
})();

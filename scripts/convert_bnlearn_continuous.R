#!/usr/bin/env Rscript
# scripts/convert_bnlearn_continuous.R
#
# One-time conversion of bnlearn's Gaussian (GBN) and Conditional Linear
# Gaussian (CLGBN) networks — distributed as R .rda files — into a
# Python-readable JSON format bundled under benchmarking/data/bnlearn/.
#
# Users do NOT need R for normal use; they consume the committed JSON.
# Re-run this only when bnlearn updates a network. Requires R with the
# `bnlearn` and `jsonlite` packages.
#
# Schema reference: docs/phase4-design-draft.md §5.
#
# Node-class handling (verified against the actual bn.fit objects):
#   - bn.fit.gnode  (pure Gaussian): $coefficients named vector
#                   ["(Intercept)", <parents>], $sd scalar.
#   - bn.fit.dnode  (discrete, in CLG nets): $prob multidim table; axis 1 is
#                   the node's own states, axes 2.. are parents (in $parents
#                   order). Serialized as dim + dimnames + column-major values.
#   - bn.fit.cgnode (CLG continuous): $coefficients is a
#                   [coef_name x config] matrix and $sd a per-config vector.
#                   Configs are the flat index over the discrete parents'
#                   levels, first discrete parent varying fastest (verified on
#                   mehra t2m: Year x Month x Hour = 5760 configs). Emitted as
#                   the compact schema — parallel arrays (intercepts/
#                   coefficients/sds) indexed by config id, plus dlevels for
#                   Python-side decoding. Far smaller than an explicit per-
#                   config list (mehra wd has K=66960). See design doc §5.2.

suppressPackageStartupMessages({
  library(bnlearn)
  library(jsonlite)
})

# Networks to convert: 4 Gaussian + 3 CLG. `file` overrides the download
# basename when it differs from the canonical network name (mehra is shipped
# as mehra-complete; the "mehra" entry is a latent-variable variant).
NETWORKS <- list(
  list(name = "ecoli70",    kind = "gaussian"),
  list(name = "magic-niab", kind = "gaussian"),
  list(name = "magic-irri", kind = "gaussian"),
  list(name = "arth150",    kind = "gaussian"),
  list(name = "healthcare", kind = "clg"),
  list(name = "sangiovese", kind = "clg"),
  list(name = "mehra",      kind = "clg", file = "mehra-complete")
)

OUTPUT_DIR <- "benchmarking/data/bnlearn"
if (!dir.exists(OUTPUT_DIR)) dir.create(OUTPUT_DIR, recursive = TRUE)

CACHE_DIR <- path.expand("~/.cache/nbn/bnlearn")
if (!dir.exists(CACHE_DIR)) dir.create(CACHE_DIR, recursive = TRUE)

download_network <- function(name, file = NULL) {
  if (is.null(file)) file <- name
  url <- paste0("https://www.bnlearn.com/bnrepository/", name, "/", file, ".rda")
  local <- file.path(CACHE_DIR, paste0(file, ".rda"))
  if (!file.exists(local)) {
    cat("  Downloading", url, "\n")
    download.file(url, local, mode = "wb", quiet = TRUE)
  }
  local
}

# ---- Per-node-class extraction ------------------------------------------

extract_gnode <- function(node, fit) {
  coefs <- fit$coefficients
  parents <- fit$parents
  # names=character(0) so an empty map serializes as {} (not []) in jsonlite.
  coef_list <- structure(list(), names = character(0))
  for (p in parents) coef_list[[p]] <- unname(coefs[[p]])
  list(
    name = node,
    type = "gaussian",
    parents = as.list(parents),
    intercept = unname(coefs[["(Intercept)"]]),
    coefficients = coef_list,
    sd = unname(fit$sd)
  )
}

extract_dnode <- function(node, fit) {
  prob <- fit$prob
  dn <- dimnames(prob)
  # Axis 1 is the node; axes 2.. are parents in $parents order.
  list(
    name = node,
    type = "discrete",
    parents = as.list(fit$parents),
    states = as.list(dn[[1]]),
    prob_dim = as.list(dim(prob)),
    prob_dimnames = dn,            # named list: node states + per-parent states
    # Column-major flatten (R default). Python reshapes with order="F".
    prob = as.numeric(prob)
  )
}

extract_cgnode <- function(node, fit) {
  # bnlearn stores CLG continuous params natively as a [coef_name x config]
  # matrix + a per-config sd vector. We keep that compact layout: parallel
  # arrays indexed by a flat config id. The config id decodes to discrete-
  # parent states with the first discrete parent varying fastest, which is
  # bnlearn's encoding (verified on mehra t2m). See design doc §5.2.
  coefs <- fit$coefficients          # [coef_name x config] matrix
  dparent_names <- fit$parents[fit$dparents]
  gparent_names <- fit$parents[fit$gparents]

  # Row 1 is the intercept; remaining rows are continuous-parent coefficients.
  # names=character(0) so empty maps serialize as {} (not []) in jsonlite —
  # a cgnode with only discrete parents has no continuous coefficients.
  intercepts <- as.numeric(coefs["(Intercept)", ])
  gp_coefs <- structure(list(), names = character(0))
  for (gp in gparent_names) gp_coefs[[gp]] <- as.numeric(coefs[gp, ])

  # dlevels preserved in dparents order (= the config-encoding order).
  dlev <- structure(list(), names = character(0))
  for (dp in dparent_names) dlev[[dp]] <- as.character(fit$dlevels[[dp]])

  # Sanity: array lengths must equal the product of discrete-parent levels.
  k_expected <- if (length(dparent_names) > 0) {
    prod(sapply(dparent_names, function(dp) length(dlev[[dp]])))
  } else {
    1
  }
  stopifnot(length(intercepts) == k_expected)

  list(
    name = node,
    type = "clg_continuous",
    discrete_parents = as.list(dparent_names),
    continuous_parents = as.list(gparent_names),
    dlevels = dlev,
    intercepts = intercepts,
    coefficients = gp_coefs,
    sds = as.numeric(fit$sd)
  )
}

extract_node <- function(node, bn) {
  fit <- bn[[node]]
  fc <- class(fit)[1]
  if (fc == "bn.fit.gnode") {
    extract_gnode(node, fit)
  } else if (fc == "bn.fit.dnode") {
    extract_dnode(node, fit)
  } else if (fc == "bn.fit.cgnode") {
    extract_cgnode(node, fit)
  } else {
    stop(paste0("Unknown node class for '", node, "': ", paste(class(fit), collapse = ",")))
  }
}

# ---- Per-network conversion ---------------------------------------------

convert_network <- function(net_spec) {
  name <- net_spec$name
  kind <- net_spec$kind
  cat("Converting", name, "...\n")

  rda_path <- download_network(name, net_spec$file)
  env <- new.env()
  load(rda_path, envir = env)
  obj_names <- ls(env)
  if (length(obj_names) != 1) {
    stop(paste0("Expected 1 object in ", rda_path, ", found ", length(obj_names)))
  }
  bn <- env[[obj_names[1]]]

  a <- arcs(bn)
  edges <- if (nrow(a) > 0) {
    lapply(seq_len(nrow(a)), function(i) list(from = a[i, 1], to = a[i, 2]))
  } else {
    list()
  }

  result <- list(
    name = name,
    kind = kind,
    nodes = as.list(nodes(bn)),
    edges = edges,
    cpds = lapply(nodes(bn), function(n) extract_node(n, bn))
  )

  out_path <- file.path(OUTPUT_DIR, paste0(name, ".json"))
  write_json(result, out_path, pretty = TRUE, auto_unbox = TRUE, digits = 10, na = "null")
  cat("  Wrote", out_path, "(", length(result$nodes), "nodes,", length(edges), "edges )\n")
}

# ---- Main loop -----------------------------------------------------------

failed <- character(0)
for (net in NETWORKS) {
  tryCatch(convert_network(net), error = function(e) {
    cat("  FAILED:", conditionMessage(e), "\n")
    failed <<- c(failed, net$name)
  })
}

cat("\nConverted", length(NETWORKS) - length(failed), "of", length(NETWORKS), "networks.\n")
if (length(failed) > 0) {
  cat("Failed:", paste(failed, collapse = ", "), "\n")
  quit(status = 1)
}

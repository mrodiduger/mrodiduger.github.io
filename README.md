# Notes for future me

This is the Quarto corner where I keep my website, technical notes, photos,
and the occasional informal thought. It is meant to stay quiet, fast, and simple. If an idea starts
making the site feel busy, it is probably worth leaving out.

## The usual loop

When I want to write or adjust the site, start the live preview from the repository root:

```sh
quarto preview
```

Quarto will print the local address and refresh the browser as files change. `Ctrl+C` stops it.

Before publishing, render the site once from scratch:

```sh
quarto render
```

The result lands in `_site/`. That folder is generated, ignored by Git, and never something to edit
by hand.

If I want to inspect the rendered version exactly as it will be published:

```sh
python3 -m http.server 8000 --directory _site
```

Then open <http://localhost:8000/>. This server does not rebuild anything, so render again after
making changes.

## When I want to write

There are two kinds of writing here.

### A technical note

1. Make a folder at `posts/<slug>/`.
2. Copy `templates/post.qmd.template` to `posts/<slug>/index.qmd`.
3. Keep images, bibliography files, and any small scripts beside the post.

The home and writing pages will find the note automatically. No need to add it to a listing by hand.

The front matter looks like this:

```yaml
---
title: "A clear, specific title"
description: "One sentence explaining why this note is worth reading."
author: "Rodi Düger"
date: 2026-07-10
categories: [software]
# bibliography: references.bib
---
```

### An informal entry

1. Make a folder at `posts/blog/<slug>/`.
2. Copy `templates/blog.qmd.template` to `posts/blog/<slug>/index.qmd`.
3. Set the title and date, then start writing.

These entries appear automatically on `/blog.html` and only need:

```yaml
---
title: "A short title"
date: 2026-07-24
---
```

## A few things I tend to forget

- Dates stay in ISO `YYYY-MM-DD` format. Quarto handles the display formatting.
- LaTeX works directly: `$f(x) = x^2$` inline or `$$ ... $$` for a display block.
- For citations, keep the BibTeX file beside the post, add `bibliography: references.bib`, and cite
  with `[@key]`.
- Normal Markdown handles images. Quarto's figure syntax adds captions and cross-references.
- Mermaid and Graphviz are available when a diagram genuinely helps.
- Source files are the real site. `_site/` and Quarto's working files are disposable output.

## When it is ready to go

Check the changes, commit the files I actually meant to change, and push to `main`.
The GitHub Action takes it from there and publishes the rendered site.

## Commands worth keeping close

```sh
quarto check    # make sure the local Quarto installation is healthy
quarto preview  # rebuild and refresh while I work
quarto render   # build the production site into _site/
python3 -m http.server 8000 --directory _site  # serve the rendered site
```

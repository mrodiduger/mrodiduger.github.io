# Rodi’s website

A Quarto-powered personal site, designed as a small static website and published on GitHub Pages.

## Run the site locally

Install [Quarto](https://quarto.org/docs/get-started/) and Python 3. Confirm both are available:

```sh
quarto check
python3 --version
```

For writing and styling, start Quarto's live preview from the repository root:

```sh
quarto preview
```

Quarto prints the local address in the terminal and refreshes the browser when source files change.
Stop it with `Ctrl+C`.

To test the same static output that will be published, render first and then serve `_site/` with
Python:

```sh
quarto render
python3 -m http.server 8000 --directory _site
```

Open <http://localhost:8000/>. The Python server does not rebuild after edits, so run
`quarto render` again whenever the source changes, then refresh the browser. Stop the server with
`Ctrl+C`.

The generated site is written to `_site/`. It is ignored by Git and should not be edited directly.

## The everyday writing workflow

1. Create a folder at `posts/<slug>/`.
2. Copy `templates/post.qmd.template` to `posts/<slug>/index.qmd`.
3. Add any images, bibliography files, or small JavaScript modules beside the post.
4. Preview while you write:

   ```sh
   quarto preview
   ```

5. Push your finished note to `main`:

   ```sh
   git add .
   git commit -m "Write <post title>"
   git push
   ```

The GitHub Action renders and publishes the site automatically. You never need to hand-write a
blog card or an HTML page: `writing.qmd` discovers posts and builds the index from their front
matter.

## One-time publishing setup

After pushing this repository to GitHub:

1. Open **Settings → Actions → General** and set **Workflow permissions** to **Read and write**.
2. Push to `main` once; this creates the `gh-pages` branch.
3. Open **Settings → Pages** and choose **Deploy from a branch** → `gh-pages` → `/(root)`.

After that, every push to `main` publishes the rendered site.

## Post metadata

Every post starts with YAML front matter:

```yaml
---
title: "A clear, specific title"
description: "One sentence explaining why this note is worth a reader’s time."
author: "Rodi Düger"
date: 2026-07-10
categories: [software]
# bibliography: references.bib
---
```

The title, description, date, and calculated reading time are used automatically on the writing
page. Existing posts already follow this pattern.

## Math, citations, and visualizations

- Write LaTeX directly: `$f(x) = x^2$` or a `$$ ... $$` display block.
- Add a BibTeX file beside a post and set `bibliography: references.bib`; cite with `[@key]`.
- Use normal Markdown for images and Quarto’s figure syntax for captions and cross-references.
- For diagrams, Quarto supports Mermaid and Graphviz directly.

## Useful commands

```sh
quarto check    # verify the local Quarto installation
quarto preview  # rebuild and refresh while writing
quarto render   # build the production site into _site/
python3 -m http.server 8000 --directory _site  # serve the rendered site
```

`_site/` and Quarto’s local working files are intentionally ignored by Git. The source files,
images, and bibliography files are what you commit.

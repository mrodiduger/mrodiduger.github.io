```{=html}
<div class="post-list list">
<% if (items.length === 0) { %>
  <p class="blog-empty">No entries yet.</p>
<% } %>
<% for (const item of items) { %>
  <article class="post-row" <%= metadataAttrs(item) %>>
    <p class="post-date"><span class="listing-date"><%- item.date %></span></p>
    <a class="post-copy" href="<%- item.path %>">
      <strong class="listing-title"><%- item.title %></strong>
    </a>
    <span class="post-arrow" aria-hidden="true">↗</span>
  </article>
<% } %>
</div>
```

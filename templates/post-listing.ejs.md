```{=html}
<div class="post-list list">
<% for (const item of items) { %>
  <article class="post-row" <%= metadataAttrs(item) %>>
    <p class="post-date">
      <span class="listing-date"><%- item.date %></span>
      <% if (item['reading-time']) { %>
        <span aria-hidden="true"> / </span><span class="listing-reading-time"><%- item['reading-time'] %></span>
      <% } %>
    </p>
    <a class="post-copy" href="<%- item.path %>">
      <strong class="listing-title"><%- item.title %></strong>
      <% if (item.description) { %><span class="post-summary listing-description"><%- item.description %></span><% } %>
    </a>
    <span class="post-arrow" aria-hidden="true">↗</span>
  </article>
<% } %>
</div>
```

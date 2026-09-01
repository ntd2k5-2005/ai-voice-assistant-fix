from __future__ import annotations

from .base import DocumentParser, parsed_documents, read_text_file, source_path


class HTMLParser(DocumentParser):
    """Walk semantic HTML blocks in source order and discard executable noise."""

    def __init__(self, encoding_fallbacks=None):
        self.encoding_fallbacks = encoding_fallbacks

    def parse(self, source):
        path = source_path(source)
        text, encoding = read_text_file(path, self.encoding_fallbacks)
        return self.parse_html(source, path, text, encoding=encoding)

    def parse_html(self, source, path, text: str, *, encoding: str = "unicode"):
        from bs4 import BeautifulSoup, NavigableString, Tag

        soup = BeautifulSoup(text, "html.parser")
        for node in soup(["script", "style", "noscript", "template", "svg", "canvas"]):
            node.decompose()
        for node in soup.select("[hidden], [aria-hidden='true']"):
            node.decompose()

        title = soup.title.get_text(" ", strip=True) if soup.title else path.stem
        segments: list[dict] = []
        headings: dict[int, str] = {}
        table_index = 0

        def current_path() -> list[str]:
            return [headings[level] for level in sorted(headings)]

        def links_for(node: Tag) -> list[str]:
            return [
                value for value in (link.get("href") for link in node.find_all("a", href=True))
                if value and not value.lower().startswith("javascript:")
            ]

        def add(value: str, block_type: str, node: Tag, **extra):
            value = value.strip(" \r\n")
            if not value:
                return
            path_values = current_path()
            segments.append(
                {
                    "text": value,
                    "title": title,
                    "method": "html",
                    "heading": path_values[-1] if path_values else title,
                    "heading_path": path_values,
                    "heading_level": max(headings) if headings else None,
                    "section": " > ".join(path_values) if path_values else title,
                    "block_type": block_type,
                    "source_line": getattr(node, "sourceline", None),
                    "links": links_for(node),
                    "encoding": encoding,
                    **extra,
                }
            )

        def render_table(node: Tag) -> tuple[str, list[str]]:
            rows: list[list[str]] = []
            for row in node.find_all("tr"):
                cells = [cell.get_text(" ", strip=True) for cell in row.find_all(["th", "td"], recursive=False)]
                if cells:
                    rows.append(cells)
            if not rows:
                return "", []
            width = max(len(row) for row in rows)
            rows = [row + [""] * (width - len(row)) for row in rows]
            header = rows[0]
            rendered = ["| " + " | ".join(header) + " |", "| " + " | ".join(["---"] * width) + " |"]
            rendered.extend("| " + " | ".join(row) + " |" for row in rows[1:])
            return "\n".join(rendered), header

        def visit(node):
            nonlocal table_index
            if isinstance(node, NavigableString):
                if node.parent and node.parent.name in {"body", "main", "article", "section"}:
                    value = str(node).strip()
                    if value:
                        add(value, "text", node.parent)
                return
            if not isinstance(node, Tag):
                return
            name = (node.name or "").lower()
            if name in {"script", "style", "noscript", "template", "svg", "canvas", "head"}:
                return
            if name in {"h1", "h2", "h3", "h4", "h5", "h6"}:
                level = int(name[1])
                value = node.get_text(" ", strip=True)
                for stale in [candidate for candidate in headings if candidate >= level]:
                    del headings[stale]
                if value:
                    headings[level] = value
                return
            if name == "table":
                rendered, headers = render_table(node)
                if rendered:
                    table_index += 1
                    add(
                        rendered,
                        "table",
                        node,
                        table_id=f"html-table-{table_index}",
                        table_index=table_index,
                        column_names=headers,
                        row_start=1,
                        row_end=max(1, len(node.find_all("tr"))),
                    )
                return
            if name in {"ul", "ol"}:
                values = []
                for index, item in enumerate(node.find_all("li", recursive=False), 1):
                    prefix = f"{index}." if name == "ol" else "-"
                    values.append(f"{prefix} {item.get_text(' ', strip=True)}")
                add("\n".join(values), "list", node)
                return
            if name in {"p", "pre", "blockquote", "figcaption", "caption", "dt", "dd"}:
                separator = "\n" if name == "pre" else " "
                add(node.get_text(separator, strip=True), name, node)
                return
            if name == "img":
                alt = (node.get("alt") or "").strip()
                if alt:
                    add(alt, "image_caption", node, image_source=node.get("src"))
                return
            for child in list(node.children):
                visit(child)

        visit(soup.body or soup)
        if not segments:
            fallback = soup.get_text("\n", strip=True)
            if fallback:
                segments = [{
                    "text": fallback,
                    "title": title,
                    "method": "html",
                    "section": title,
                    "heading": title,
                    "block_type": "text",
                    "encoding": encoding,
                }]
        return parsed_documents(source, path, segments)

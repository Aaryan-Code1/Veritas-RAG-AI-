import re
from typing import List, Dict, Any
from bs4 import BeautifulSoup


class DocumentParser:
    """Parses raw document text, HTML, and Markdown into clean structural blocks."""

    @staticmethod
    def parse_html(html_content: str) -> str:
        """Strips HTML boilerplate while preserving semantic headers and table formatting."""
        soup = BeautifulSoup(html_content, "html.parser")

        # Strip navigation, script, and style tags
        for element in soup(["script", "style", "nav", "footer", "header", "aside"]):
            element.decompose()

        # Format tables cleanly
        for table in soup.find_all("table"):
            rows = []
            for tr in table.find_all("tr"):
                cells = [td.get_text(strip=True) for td in tr.find_all(["td", "th"])]
                if cells:
                    rows.append(" | ".join(cells))
            if rows:
                table_md = "\n" + "\n".join(rows) + "\n"
                table.replace_with(table_md)

        text = soup.get_text(separator="\n")
        # Collapse multiple empty lines
        text = re.sub(r"\n{3,}", "\n\n", text).strip()
        return text

    @staticmethod
    def extract_sections(markdown_text: str) -> List[Dict[str, Any]]:
        """Splits Markdown text into sections based on headers (# Header)."""
        lines = markdown_text.splitlines()
        sections = []
        current_header = "Introduction"
        current_content: List[str] = []

        header_regex = re.compile(r"^(#{1,6})\s+(.+)$")

        for line in lines:
            match = header_regex.match(line.strip())
            if match:
                if current_content:
                    sections.append({
                        "section_name": current_header,
                        "content": "\n".join(current_content).strip()
                    })
                    current_content = []
                current_header = match.group(2).strip()
            else:
                current_content.append(line)

        if current_content:
            sections.append({
                "section_name": current_header,
                "content": "\n".join(current_content).strip()
            })

        return sections

"""XLSX parser — extracts sheets as tables with row chunking for small tables."""
import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)


class XLSXParser:
    """Parse XLSX files into the standard document dict format."""

    def parse(self, path: str) -> dict:
        xlsx = pd.ExcelFile(path)
        pages = []
        tables = []

        for sheet_name in xlsx.sheet_names:
            df = xlsx.parse(sheet_name)
            if df.empty:
                continue

            # Store table data as list of lists (header + rows)
            header = [str(c) for c in df.columns.tolist()]
            rows = []
            for _, row in df.iterrows():
                rows.append([str(v) for v in row.tolist()])

            table_data = [header] + rows
            tables.append({
                "data": table_data,
                "index": len(tables),
                "sheet_name": sheet_name,
                "num_rows": len(rows),
                "dataframe": df,
            })

            # Also create a text representation for vectorstore
            text_repr = f"Sheet: {sheet_name}\n"
            text_repr += f"Columns: {', '.join(header)}\n"
            text_repr += f"Rows: {len(rows)}\n"
            # Include sample rows for context
            sample_rows = min(5, len(rows))
            for row_data in table_data[1:sample_rows + 1]:
                text_repr += " | ".join(row_data) + "\n"
            if len(rows) > sample_rows:
                text_repr += f"... ({len(rows) - sample_rows} more rows)\n"

            pages.append({"text": text_repr, "page": 1})

        return {
            "pages": pages,
            "source": Path(path).name,
            "title": Path(path).stem,
            "creation_date": "",
            "authors": "",
            "tables": tables,
        }

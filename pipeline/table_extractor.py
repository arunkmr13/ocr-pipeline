# pipeline/table_extractor.py
from transformers import TableTransformerForObjectDetection, DetrImageProcessor
import torch
from PIL import Image
from dataclasses import dataclass, field
from typing import List, Dict, Any
import json

@dataclass
class TableCell:
    row: int
    col: int
    row_span: int
    col_span: int
    bbox: tuple
    text: str = ""

@dataclass
class ExtractedTable:
    bbox: tuple
    cells: List[TableCell] = field(default_factory=list)
    num_rows: int = 0
    num_cols: int = 0

    def to_json(self) -> Dict[str, Any]:
        grid = [[None] * self.num_cols for _ in range(self.num_rows)]
        for cell in self.cells:
            if cell.row < self.num_rows and cell.col < self.num_cols:
                grid[cell.row][cell.col] = cell.text
        return {"rows": self.num_rows, "cols": self.num_cols, "data": grid}


class TableExtractor:
    DETECTION_MODEL = "microsoft/table-transformer-detection"
    STRUCTURE_MODEL = "microsoft/table-transformer-structure-recognition-v1.1-all"

    def __init__(self, device: str = "cpu"):
        self.device = device
        self.processor = DetrImageProcessor.from_pretrained(self.DETECTION_MODEL)
        self.detector = TableTransformerForObjectDetection.from_pretrained(
            self.DETECTION_MODEL
        ).to(device).eval()
        self.structure_processor = DetrImageProcessor.from_pretrained(
            self.STRUCTURE_MODEL
        )
        self.structure_model = TableTransformerForObjectDetection.from_pretrained(
            self.STRUCTURE_MODEL
        ).to(device).eval()

    @torch.inference_mode()
    def detect_tables(self, pil_image: Image.Image, threshold: float = 0.85) -> List[tuple]:
        """Returns list of table bounding boxes."""
        inputs = self.processor(images=pil_image, return_tensors="pt").to(self.device)
        outputs = self.detector(**inputs)
        target_sizes = torch.tensor([pil_image.size[::-1]]).to(self.device)
        results = self.processor.post_process_object_detection(
            outputs, threshold=threshold, target_sizes=target_sizes
        )[0]
        boxes = []
        for score, label, box in zip(
            results["scores"], results["labels"], results["boxes"]
        ):
            if label.item() == 0:  # label 0 = table
                x1, y1, x2, y2 = [int(v) for v in box.tolist()]
                boxes.append((x1, y1, x2, y2))
        return boxes

    @torch.inference_mode()
    def extract_structure(self, table_image: Image.Image,table_bbox: tuple) -> ExtractedTable:
        """
        Given a cropped table image, recover cell grid structure.
        Uses structure recognition model to detect rows/columns/cells.
        """
        inputs = self.structure_processor(
            images=table_image, return_tensors="pt"
        ).to(self.device)
        outputs = self.structure_model(**inputs)
        target_sizes = torch.tensor([table_image.size[::-1]]).to(self.device)
        results = self.structure_processor.post_process_object_detection(
            outputs, threshold=0.7, target_sizes=target_sizes
        )[0]

        rows, cols, cells = [], [], []
        label_names = self.structure_model.config.id2label

        for score, label, box in zip(
            results["scores"], results["labels"], results["boxes"]
        ):
            name = label_names[label.item()]
            bbox = tuple(int(v) for v in box.tolist())
            if "row" in name and "header" not in name:
                rows.append(bbox)
            elif "column" in name:
                cols.append(bbox)
            elif "spanning cell" in name or name == "table cell":
                cells.append(bbox)

        rows = sorted(rows, key=lambda b: b[1])
        cols = sorted(cols, key=lambda b: b[0])

        table = ExtractedTable(
            bbox=table_bbox,
            num_rows=len(rows),
            num_cols=len(cols)
        )

        # Assign each cell to a grid position
        for cell_bbox in cells:
            cx = (cell_bbox[0] + cell_bbox[2]) / 2
            cy = (cell_bbox[1] + cell_bbox[3]) / 2
            row_idx = self._find_index(cy, [(r[1] + r[3]) / 2 for r in rows])
            col_idx = self._find_index(cx, [(c[0] + c[2]) / 2 for c in cols])
            table.cells.append(TableCell(
                row=row_idx, col=col_idx,
                row_span=1, col_span=1,
                bbox=cell_bbox
            ))

        return table

    def _find_index(self, value: float, centers: List[float]) -> int:
        if not centers:
            return 0
        return min(range(len(centers)), key=lambda i: abs(centers[i] - value))
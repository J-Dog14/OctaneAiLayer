/*
 * Build a chain-breakdown analysis report (.docx, landscape).
 *
 * Usage:
 *   node build_chain_analysis.js <input.json> <output.docx>
 *
 * Input JSON structure:
 *   {
 *     "subject": { name, handed, age_str, size_str, velo_summary, test_date },
 *     "title_subline": "How a modest pelvis produces elite velocity ...",
 *     "observation": "Paragraph(s) of what the data shows",
 *     "rankings": [ { segment, value, rank, top, read, status }, ... ],   // 9 rows typical
 *     "amplification": [ { name, velo, p2t, t2h, h2hand, note, highlight }, ... ],
 *     "mechanism": "Paragraph(s) explaining WHY the chain works/breaks",
 *     "comparison_with": "Boyatt, Max" | null,
 *     "comparison_rows": [ { dim, a, b, read }, ... ] | null,
 *     "diagnosis": "Closing diagnosis paragraph",
 *     "headroom": [ { heading, body }, ... ] | null,
 *     "bottom_line": "Closing paragraph"
 *   }
 */

const fs = require('fs');
const {
  Document, Packer, Paragraph, TextRun, Table, TableRow, TableCell,
  HeadingLevel, AlignmentType, WidthType, BorderStyle, ShadingType,
  LevelFormat, PageOrientation, PageBreak
} = require('docx');

const border = { style: BorderStyle.SINGLE, size: 4, color: "BFBFBF" };
const borders = { top: border, bottom: border, left: border, right: border };
const cellMargins = { top: 100, bottom: 100, left: 140, right: 140 };
const TABLE_CELL_SIZE = 24;
const TABLE_HEADER_SIZE = 26;
const CW = 14400;

const STATUS_COLORS = { best: '548235', good: '70AD47', warn: 'ED7D31', bad: 'C00000', neutral: '7F7F7F' };

const P = (text) => new Paragraph({ spacing: { after: 120 }, children: Array.isArray(text) ? text : [new TextRun(text)] });
const bullet = (text) => new Paragraph({ numbering: { reference: "bullets", level: 0 }, spacing: { after: 80 }, children: Array.isArray(text) ? text : [new TextRun(text)] });
const h1 = (text) => new Paragraph({ heading: HeadingLevel.HEADING_1, spacing: { before: 280, after: 140 }, children: [new TextRun(text)] });
const h2 = (text) => new Paragraph({ heading: HeadingLevel.HEADING_2, spacing: { before: 220, after: 120 }, children: [new TextRun(text)] });
const h3 = (text) => new Paragraph({ heading: HeadingLevel.HEADING_3, spacing: { before: 160, after: 80 }, children: [new TextRun(text)] });

function cell(text, opts = {}) {
  const { bold = false, fill = null, width = 1872, align = AlignmentType.LEFT, color = null, size = TABLE_CELL_SIZE } = opts;
  return new TableCell({ borders, margins: cellMargins,
    width: { size: width, type: WidthType.DXA },
    shading: fill ? { fill, type: ShadingType.CLEAR } : undefined,
    children: [new Paragraph({ alignment: align, children: [new TextRun({ text: String(text), bold, color, size })] })] });
}
function headerCell(text, width) {
  return new TableCell({ borders, margins: cellMargins,
    width: { size: width, type: WidthType.DXA },
    shading: { fill: "1F4E78", type: ShadingType.CLEAR },
    children: [new Paragraph({ alignment: AlignmentType.CENTER, children: [new TextRun({ text, bold: true, color: "FFFFFF", size: TABLE_HEADER_SIZE })] })] });
}
function statusCell(text, status, width) {
  return new TableCell({ borders, margins: cellMargins,
    width: { size: width, type: WidthType.DXA },
    shading: { fill: STATUS_COLORS[status] || STATUS_COLORS.neutral, type: ShadingType.CLEAR },
    children: [new Paragraph({ alignment: AlignmentType.CENTER, children: [new TextRun({ text, bold: true, color: 'FFFFFF', size: TABLE_CELL_SIZE })] })] });
}

function buildRankings(rows) {
  const w = [3400, 1700, 1700, 2900, 4700];
  const out = [
    new TableRow({ tableHeader: true, children: [
      headerCell("Segment / Plane", w[0]),
      headerCell("Pitcher", w[1]),
      headerCell("Rank", w[2]),
      headerCell("Top of cohort", w[3]),
      headerCell("Read", w[4]),
    ]}),
  ];
  rows.forEach(r => {
    out.push(new TableRow({ children: [
      cell(r.segment, { bold: true, width: w[0], fill: "F2F2F2" }),
      cell(r.value, { width: w[1], align: AlignmentType.CENTER, bold: true }),
      statusCell(r.rank, r.status, w[2]),
      cell(r.top, { width: w[3], align: AlignmentType.CENTER }),
      cell(r.read, { width: w[4] }),
    ]}));
  });
  return out;
}

function buildAmplification(rows) {
  const w = [3000, 1100, 2200, 2200, 2200, 3700];
  const out = [
    new TableRow({ tableHeader: true, children: [
      headerCell("Pitcher", w[0]),
      headerCell("V", w[1]),
      headerCell("Pelvis → Thorax", w[2]),
      headerCell("Thorax → Humerus", w[3]),
      headerCell("Humerus → Hand", w[4]),
      headerCell("Note", w[5]),
    ]}),
  ];
  rows.forEach(r => {
    out.push(new TableRow({ children: [
      cell(r.name, { bold: true, width: w[0], fill: r.highlight ? "FFF2CC" : null }),
      cell(r.velo, { width: w[1], align: AlignmentType.CENTER, bold: true }),
      cell(r.p2t, { width: w[2], align: AlignmentType.CENTER, bold: r.highlight }),
      cell(r.t2h, { width: w[3], align: AlignmentType.CENTER }),
      cell(r.h2hand, { width: w[4], align: AlignmentType.CENTER }),
      cell(r.note, { width: w[5] }),
    ]}));
  });
  return out;
}

function buildComparison(rows, aName, bName) {
  const w = [2800, 2900, 2900, 4800];
  const out = [
    new TableRow({ tableHeader: true, children: [
      headerCell("Dimension", w[0]),
      headerCell(aName, w[1]),
      headerCell(bName, w[2]),
      headerCell("Read", w[3]),
    ]}),
  ];
  rows.forEach(r => {
    out.push(new TableRow({ children: [
      cell(r.dim, { bold: true, width: w[0], fill: "F2F2F2" }),
      cell(r.a, { width: w[1], align: AlignmentType.CENTER, bold: true }),
      cell(r.b, { width: w[2], align: AlignmentType.CENTER, bold: true }),
      cell(r.read, { width: w[3] }),
    ]}));
  });
  return out;
}

function buildDoc(input) {
  const s = input.subject;
  const children = [
    new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 60 },
      children: [new TextRun({ text: `${s.name} — Chain Breakdown Analysis`, bold: true, size: 40 })] }),
    new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 80 },
      children: [new TextRun({ text: input.title_subline || 'Where the velocity is — and where it\'s leaking', size: 24, color: "555555" })] }),
    new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 240 },
      children: [new TextRun({ text: `${s.handed} · ${s.size_str || ''} · ${s.age_str || ''} · ${s.velo_summary || ''} · ${s.test_date || ''}`, size: 20, color: "888888", italics: true })] }),

    h1("The Observation"),
    P(input.observation || ''),
  ];

  if (input.rankings && input.rankings.length) {
    children.push(h2("Where the pitcher sits across the cohort"));
    children.push(new Table({ width: { size: CW, type: WidthType.DXA },
      columnWidths: [3400, 1700, 1700, 2900, 4700], rows: buildRankings(input.rankings) }));
    children.push(P(" "));
  }

  if (input.amplification && input.amplification.length) {
    children.push(new Paragraph({ children: [new PageBreak()] }));
    children.push(h1("Amplification Ratios — Chain Transfer"));
    children.push(P("Each segment should be faster than the one before it. Elite Pelvis→Thorax amplification is 1.4–1.8×. Humerus→Hand of <1.0× is typical and healthy; >1.0× means the arm is rescuing the chain."));
    children.push(new Table({ width: { size: CW, type: WidthType.DXA },
      columnWidths: [3000, 1100, 2200, 2200, 2200, 3700], rows: buildAmplification(input.amplification) }));
    children.push(P(" "));
  }

  if (input.mechanism) {
    children.push(h1("The Mechanism"));
    if (Array.isArray(input.mechanism)) {
      input.mechanism.forEach(p => children.push(P(p)));
    } else {
      children.push(P(input.mechanism));
    }
  }

  if (input.comparison_rows && input.comparison_rows.length && input.comparison_a && input.comparison_b) {
    children.push(new Paragraph({ children: [new PageBreak()] }));
    children.push(h1(`Side-by-Side — ${input.comparison_a} vs ${input.comparison_b}`));
    if (input.comparison_intro) children.push(P(input.comparison_intro));
    children.push(new Table({ width: { size: CW, type: WidthType.DXA },
      columnWidths: [2800, 2900, 2900, 4800], rows: buildComparison(input.comparison_rows, input.comparison_a, input.comparison_b) }));
    children.push(P(" "));
  }

  if (input.headroom && input.headroom.length) {
    children.push(h1("Where the Velocity Headroom Is"));
    input.headroom.forEach(item => {
      children.push(h3(item.heading));
      children.push(P(item.body));
    });
  }

  if (input.diagnosis) {
    children.push(h1("Diagnosis"));
    children.push(P(input.diagnosis));
  }

  if (input.bottom_line) {
    children.push(h1("Bottom Line"));
    if (Array.isArray(input.bottom_line)) {
      input.bottom_line.forEach(p => children.push(P(p)));
    } else {
      children.push(P(input.bottom_line));
    }
  }

  children.push(P(" "));
  children.push(new Paragraph({ alignment: AlignmentType.CENTER, children: [new TextRun({ text: "— End of analysis —", italics: true, color: "888888" })] }));

  return new Document({
    creator: "8ctane Baseball",
    title: `${s.name} — Chain Breakdown Analysis`,
    styles: {
      default: { document: { run: { font: "Calibri", size: 22 } } },
      paragraphStyles: [
        { id: "Heading1", name: "Heading 1", basedOn: "Normal", next: "Normal", quickFormat: true,
          run: { size: 32, bold: true, color: "1F4E78", font: "Calibri" },
          paragraph: { spacing: { before: 280, after: 140 }, outlineLevel: 0 } },
        { id: "Heading2", name: "Heading 2", basedOn: "Normal", next: "Normal", quickFormat: true,
          run: { size: 28, bold: true, color: "2E75B6", font: "Calibri" },
          paragraph: { spacing: { before: 240, after: 120 }, outlineLevel: 1 } },
        { id: "Heading3", name: "Heading 3", basedOn: "Normal", next: "Normal", quickFormat: true,
          run: { size: 24, bold: true, color: "333333", font: "Calibri" },
          paragraph: { spacing: { before: 180, after: 100 }, outlineLevel: 2 } },
      ]
    },
    numbering: {
      config: [
        { reference: "bullets",
          levels: [{ level: 0, format: LevelFormat.BULLET, text: "•", alignment: AlignmentType.LEFT,
            style: { paragraph: { indent: { left: 720, hanging: 360 } } } }] }
      ]
    },
    sections: [{
      properties: {
        page: {
          size: { width: 12240, height: 15840, orientation: PageOrientation.LANDSCAPE },
          margin: { top: 720, right: 720, bottom: 720, left: 720 }
        }
      },
      children
    }]
  });
}

async function main() {
  const [, , inputPath, outputPath] = process.argv;
  if (!inputPath || !outputPath) {
    console.error("Usage: node build_chain_analysis.js <input.json> <output.docx>");
    process.exit(1);
  }
  const input = JSON.parse(fs.readFileSync(inputPath, 'utf8'));
  const doc = buildDoc(input);
  const buf = await Packer.toBuffer(doc);
  fs.writeFileSync(outputPath, buf);
  console.log(`Wrote ${outputPath} (${buf.length} bytes)`);
}

main().catch(e => { console.error(e); process.exit(1); });
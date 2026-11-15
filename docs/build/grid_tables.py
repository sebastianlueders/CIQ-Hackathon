"""Rewrite pandoc's longtables into fully bordered grids (vertical + horizontal rules)."""
import re, sys
tex = open(sys.argv[1]).read()

def grid(block):
    # column spec: {@{} >{..}p{..} >{..}p{..}@{}}  ->  {| >{..}p{..} | >{..}p{..} |}
    head, rest = block.split("\n", 1)
    spec_end = rest.index("@{}}")
    spec = head + "\n" + rest[:spec_end]
    body = rest[spec_end + len("@{}}"):]
    spec = spec.replace("{@{}", "{|", 1)
    spec = re.sub(r"(p\{[^\n]*\}\})\n", r"\1|\n", spec)  # bar after each column but the last
    spec = spec + "|}"
    body = re.sub(r"\\(toprule|midrule)\\noalign\{\}", r"\\hline", body)
    body = re.sub(r"\\bottomrule\\noalign\{\}", "", body)
    head_part, rows = body.split("\\endlastfoot", 1)
    rows = re.sub(r"\\\\\s*\n", r"\\\\ \\hline\n", rows)  # rule under every body row
    return spec + head_part + "\\endlastfoot" + rows

out, pos = [], 0
for m in re.finditer(r"\\begin\{longtable\}.*?\\end\{longtable\}", tex, re.S):
    out.append(tex[pos:m.start()]); out.append(grid(m.group(0))); pos = m.end()
out.append(tex[pos:])
open(sys.argv[1], "w").write("".join(out))

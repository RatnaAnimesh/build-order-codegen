"""Decompositions: parse the decomposer's stubs, parse implementations, build the dependency levels, assemble programs.

Shared by both decomposition arms, so they differ only in what the prompts and the scheduler (arms.py) do.

* A unit is a top-level function of the decomposition, or a method of its `class Solution`.
* The decomposition shown to implementers is DecompRL's "function headers and documentation strings": imports and
  stubbed units only. `render(bodies)` shows chosen implementations in place of some stubs (the idea arm).
* A program is the decomposition's own code block with each unit's lines replaced by one implementation, so its
  constants and entry code stay. Imports and extra top-level definitions an implementation brings go on top.
  Entry: a LeetCode problem without `class Solution` gets a wrapper that calls the top-level function of the same
  name; a stdin program that never calls anything at top level gets `main()` (or `solve()`).
* An implementation is the last fenced block that defines the unit's function, at top level or inside a class.
"""
import ast, builtins, itertools, re, textwrap
from dataclasses import dataclass, field

from lcb_exec import PRELUDE

_g = {}
exec(PRELUDE, _g)
KNOWN = set(dir(builtins)) | set(_g)
FENCE = re.compile(r"^\s*```")
USES = re.compile(r"^\s*uses\s*:\s*(.*)$", re.I)
MISSING = re.compile(r"^\s*MISSING\s*:\s*(.+)$", re.M)


def blocks(text):
    """Fenced code blocks in order, pairing ``` lines one after another."""
    lines = text.split("\n")
    idx = [i for i, l in enumerate(lines) if FENCE.match(l)]
    return ["\n".join(lines[a + 1:b]) for a, b in zip(idx[0::2], idx[1::2])]


def lcb_extract(text):
    """lcb_runner's extract_code for chat models: between the last two lines that contain ```."""
    lines = text.split("\n")
    idx = [i for i, l in enumerate(lines) if "```" in l]
    return "\n".join(lines[idx[-2] + 1:idx[-1]]) if len(idx) >= 2 else ""


def _parse(src):
    for s in (src, textwrap.dedent(src)):
        try:
            return ast.parse(s), s
        except (SyntaxError, ValueError):
            pass
    return None, None


def _seg(src_lines, node):
    """Source of a def/class including decorators, dedented."""
    start = min([d.lineno for d in node.decorator_list] + [node.lineno])
    return textwrap.dedent("\n".join(src_lines[start - 1:node.end_lineno])), start


def _stub(node):
    n = ast.FunctionDef(name=node.name, args=node.args, returns=node.returns, decorator_list=node.decorator_list,
                        type_params=getattr(node, "type_params", []), lineno=0, col_offset=0,
                        body=[*([node.body[0]] if ast.get_docstring(node) is not None else []), ast.Pass()])
    return ast.unparse(ast.fix_missing_locations(n))


def _header(node):
    return ast.unparse(ast.fix_missing_locations(ast.FunctionDef(
        name=node.name, args=node.args, returns=node.returns, decorator_list=[], type_params=getattr(node, "type_params", []),
        body=[ast.Pass()], lineno=0, col_offset=0))).split("\n")[0]


@dataclass
class Unit:
    name: str                 # "f" or "Solution.f"
    fname: str
    cls: str | None
    start: int
    end: int
    col: int
    header: str
    stub: str
    uses_raw: list | None     # names on the docstring's Uses: line, None if it has none


@dataclass
class Decomp:
    src: str
    units: list
    imports: list
    layout: list              # ("fn", unit name) | ("cls", class name, [unit names])
    top_names: set
    deps: dict = field(default_factory=dict)

    def unit(self, name):
        return next(u for u in self.units if u.name == name)

    def render(self, bodies=None):
        """The decomposition as an implementer sees it: imports + each unit as a stub, or as bodies[name]."""
        bodies, out = bodies or {}, list(self.imports)
        for item in self.layout:
            if item[0] == "fn":
                out.append(bodies.get(item[1]) or self.unit(item[1]).stub)
            else:
                parts = [textwrap.indent(bodies.get(n) or self.unit(n).stub, "    ") for n in item[2]]
                out.append(f"class {item[1]}:\n" + "\n\n".join(parts))
        return "\n\n".join(out)


def parse_decomposition(text):
    for src in reversed(blocks(text)):
        tree, src = _parse(src)
        if tree is None:
            continue
        lines, units, imports, layout, top = src.split("\n"), [], [], [], set()
        for node in tree.body:
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                imports.append(ast.get_source_segment(src, node))
                top |= {(a.asname or a.name).split(".")[0] for a in node.names}
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                units.append(_unit(node, None, lines))
                layout.append(("fn", node.name))
            elif isinstance(node, ast.ClassDef):
                top.add(node.name)
                if node.name == "Solution":
                    ms = [m for m in node.body if isinstance(m, (ast.FunctionDef, ast.AsyncFunctionDef))]
                    units += [_unit(m, node.name, lines) for m in ms]
                    layout.append(("cls", node.name, [f"{node.name}.{m.name}" for m in ms]))
            else:
                for t in ast.walk(node):
                    if isinstance(t, ast.Name) and isinstance(t.ctx, ast.Store):
                        top.add(t.id)
        if units:
            return Decomp(src=src, units=units, imports=imports, layout=layout, top_names=top)
    return None


def _unit(node, cls, lines):
    doc = ast.get_docstring(node) or ""
    uses = None
    for line in doc.split("\n"):
        m = USES.match(line)
        if m:
            raw = re.split(r"[,\s]+", m.group(1).strip())
            uses = [w.strip("`()").removeprefix("self.") for w in raw if w.strip("`()-.") and w.lower() not in ("none", "nothing")]
    start = min([d.lineno for d in node.decorator_list] + [node.lineno])
    return Unit(name=f"{cls}.{node.name}" if cls else node.name, fname=node.name, cls=cls, start=start,
                end=node.end_lineno, col=node.col_offset, header=_header(node), stub=_stub(node), uses_raw=uses)


def dependency_levels(d):
    """deps from the Uses: lines (names resolved to units; self-use dropped) -> list of levels, basics first.
    Mutually recursive units share a level (strongly connected components)."""
    by_fname = {}
    for u in d.units:
        by_fname.setdefault(u.fname, u.name)
    d.deps = {u.name: sorted({by_fname[x] for x in (u.uses_raw or []) if x in by_fname} - {u.name}) for u in d.units}
    index, low, stack, on, comps, counter = {}, {}, [], set(), [], [0]

    def strong(v):                                        # Tarjan
        index[v] = low[v] = counter[0]
        counter[0] += 1
        stack.append(v)
        on.add(v)
        for w in d.deps[v]:
            if w not in index:
                strong(w)
                low[v] = min(low[v], low[w])
            elif w in on:
                low[v] = min(low[v], index[w])
        if low[v] == index[v]:
            comp = set()
            while True:
                w = stack.pop()
                on.discard(w)
                comp.add(w)
                if w == v:
                    break
            comps.append(comp)

    for u in d.units:
        if u.name not in index:
            strong(u.name)
    comp_of = {v: i for i, c in enumerate(comps) for v in c}
    level = {}
    for i, c in enumerate(comps):                         # Tarjan emits components callees first
        level[i] = max([level[comp_of[w]] + 1 for v in c for w in d.deps[v] if comp_of[w] != i], default=0)
    out = [[] for _ in range(max(level.values(), default=-1) + 1)]
    for u in d.units:
        out[level[comp_of[u.name]]].append(u.name)
    return out


def roots(d):
    """Units nothing else uses; the entry (main / solve / the LeetCode method) first."""
    used = {w for v in d.deps.values() for w in v}
    rs = [u.name for u in d.units if u.name not in used] or [d.units[-1].name]
    pri = lambda n: (0 if n.split(".")[-1] in ("main", "solve") or n.startswith("Solution.") else 1)
    return sorted(rs, key=pri)


@dataclass
class Impl:
    def_src: str
    imports: list
    helpers: list
    tokens: int = 0
    seen: dict = field(default_factory=dict)        # idea arm: dependency -> index of the implementation shown
    built: list = field(default_factory=list)       # idea arm: helpers built for it (MISSING or checker)


def parse_impl(text, unit, plan_fnames):
    for src in reversed(blocks(text)):
        tree, src = _parse(src)
        if tree is None:
            continue
        lines, found = src.split("\n"), None
        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == unit.fname:
                found = node
            elif isinstance(node, ast.ClassDef) and not found:
                found = next((m for m in node.body if isinstance(m, (ast.FunctionDef, ast.AsyncFunctionDef))
                              and m.name == unit.fname), None)
        if not found:
            continue
        imports, helpers = [], []
        for node in tree.body:
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                imports.append(ast.get_source_segment(src, node))
            elif node is found:
                continue
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                if node.name not in plan_fnames and node.name != "Solution":
                    helpers.append(_seg(lines, node)[0])
            elif isinstance(node, (ast.Assign, ast.AnnAssign)):
                names = {t.id for t in ast.walk(node) if isinstance(t, ast.Name) and isinstance(t.ctx, ast.Store)}
                if names and not names & plan_fnames:
                    helpers.append(ast.get_source_segment(src, node))
        return Impl(def_src=_seg(lines, found)[0], imports=imports, helpers=helpers)
    return None


def parse_missing(text):
    """MISSING lines -> [(name, signature line)], only when the reply has no code block (it stopped instead)."""
    if blocks(text):
        return []
    out = []
    for m in MISSING.finditer(text):
        sig = m.group(1).strip().strip("`")
        nm = re.match(r"(?:async\s+)?def\s+([A-Za-z_]\w*)", sig) or re.match(r"([A-Za-z_]\w*)\s*\(", sig)
        if nm:
            out.append((nm.group(1), sig))
    return out


def parse_helper(text, name):
    for src in reversed(blocks(text)):
        tree, src = _parse(src)
        if tree is None:
            continue
        if any(isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name == name for n in tree.body):
            keep = [n for n in tree.body if not isinstance(n, (ast.Import, ast.ImportFrom))]
            imps = [ast.get_source_segment(src, n) for n in tree.body if isinstance(n, (ast.Import, ast.ImportFrom))]
            lines = src.split("\n")
            body = [(_seg(lines, n)[0] if hasattr(n, "decorator_list") else ast.get_source_segment(src, n)) for n in keep]
            return imps, body
    return None


def undefined_calls(im, d):
    """Names this implementation calls that nothing defines: not builtins or LCB's prelude, not its own imports,
    helpers, arguments or locals, not a unit or another top-level name of the decomposition."""
    tree, _ = _parse("\n".join(im.imports + im.helpers + [im.def_src]))
    if tree is None:
        return []
    bound = set()
    for n in ast.walk(tree):
        if isinstance(n, ast.Name) and isinstance(n.ctx, (ast.Store, ast.Del)):
            bound.add(n.id)
        elif isinstance(n, ast.arg):
            bound.add(n.arg)
        elif isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            bound.add(n.name)
        elif isinstance(n, (ast.Import, ast.ImportFrom)):
            bound |= {(a.asname or a.name).split(".")[0] for a in n.names}
        elif isinstance(n, ast.ExceptHandler) and n.name:
            bound.add(n.name)
        elif isinstance(n, (ast.Global, ast.Nonlocal)):
            bound |= set(n.names)
    called = {n.func.id for n in ast.walk(tree) if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)}
    return sorted(called - bound - KNOWN - {u.fname for u in d.units} - d.top_names)


def _dedupe(xs):
    seen, out = set(), []
    for x in xs:
        if x and x not in seen:
            seen.add(x)
            out.append(x)
    return out


def assemble(d, choice, testtype, fn_name):
    """choice: unit name -> Impl or None (None keeps the stub)."""
    lines = d.src.split("\n")
    for u in sorted(d.units, key=lambda u: -u.start):
        im = choice.get(u.name)
        text = im.def_src if im else u.stub
        lines[u.start - 1:u.end] = textwrap.indent(text, " " * u.col, lambda l: True).split("\n")
    ims = [im for im in choice.values() if im]
    head = _dedupe([x for im in ims for x in im.imports]) + _dedupe([x for im in ims for x in im.helpers])
    return entry_fix("\n\n".join(head + ["\n".join(lines)]) + "\n", testtype, fn_name)


def entry_fix(prog, testtype, fn_name):
    tree, _ = _parse(prog)
    if tree is None:
        return prog
    defs = {n.name: n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))}
    classes = {n.name for n in tree.body if isinstance(n, ast.ClassDef)}
    if testtype == "functional":
        if "Solution" not in classes and fn_name in defs:
            prog += (f"\n\nclass Solution:\n    def {fn_name}(self, *args, **kwargs):\n"
                     f"        return {fn_name}(*args, **kwargs)\n")
        return prog
    if any(isinstance(n, ast.If) or (isinstance(n, ast.Expr) and isinstance(n.value, ast.Call)) for n in tree.body):
        return prog
    if "main" in defs:
        return prog + "\n\nmain()\n"
    s = defs.get("solve")
    if s and len(s.args.args) == len(s.args.defaults):
        return prog + "\n\nsolve()\n"
    return prog


def combos(d, impls, m, rng, first=()):
    """Up to m distinct choices (unit -> index into impls[unit], or None for the stub when a unit has no parsed
    implementation): the `first` choices in order, then uniform random ones without replacement."""
    names = [u.name for u in d.units]
    opts = [[i for i, im in enumerate(impls[n]) if im is not None] or [None] for n in names]
    out, seen = [], set()
    for c in first:
        key = tuple(c.get(n) for n in names)
        if key not in seen and len(out) < m:
            seen.add(key)
            out.append(key)
    total = 1
    for o in opts:
        total *= len(o)
    if total <= 4 * m:                                    # small product: shuffle all of it
        allk = list(itertools.product(*opts))
        rng.shuffle(allk)
        for key in allk:
            if len(out) >= m:
                break
            if key not in seen:
                seen.add(key)
                out.append(key)
    else:
        tries = 0
        while len(out) < m and tries < 20 * m:
            tries += 1
            key = tuple(rng.choice(o) for o in opts)
            if key not in seen:
                seen.add(key)
                out.append(key)
    return [{n: k for n, k in zip(names, key)} for key in out]

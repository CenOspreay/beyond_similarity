import re
from tree_sitter import Parser
from tree_sitter_languages import get_language


###############################################
# 工具：把语言别名映射为 tree-sitter 名字
###############################################
LANG_MAP = {
    "c": "c",
    "cpp": "cpp",
    "java": "java",
    "cs": "c_sharp",
    "c#": "c_sharp",
    "csharp": "c_sharp",
    "python": "python",
    "py": "python",
}


###############################################
# 1) 遍历 AST，找所有 for/while/foreach 结构
###############################################
def extract_loops(root, lang):
    """
    递归遍历 AST，提取语言中的所有循环节点，返回列表
    """
    loops = []

    def dfs(node):
        if lang in ("c", "cpp", "java", "cs"):
            if node.type in ("for_statement", "while_statement", "enhanced_for_statement"):
                loops.append(node)
        elif lang == "python":
            if node.type == "for_statement":
                loops.append(node)

        for ch in node.children:
            dfs(ch)

    dfs(root)
    return loops


###############################################
# 2) 把所有循环统一为规范形式
###############################################
def normalize_loop(code, node, lang):
    """
    把不同语言循环统一为：

    for (i = 0; i < N; i++):
        elem = list[i]
        body
    """

    src = code

    # 获取循环代码区间
    start = node.start_byte
    end = node.end_byte
    loop_src = code[start:end]

    ###############################################
    # C / C++ / Java / C# for(i=0;i<n;i++)
    ###############################################
    if lang in ("c", "cpp", "java", "cs") and node.type == "for_statement":
        # 简单匹配头部 i=0; i<n; i++
        head_match = re.search(r'for\s*\((.*?)\)\s*', loop_src, re.S)
        if not head_match:
            return code

        head = head_match.group(1)
        parts = [p.strip() for p in head.split(";")]
        if len(parts) != 3:
            return code

        init, cond, iter_ = parts

        # 从 init 提取 i
        m = re.search(r'(\w+)\s*=', init)
        if not m:
            return code
        idx = m.group(1)

        # 统一 cond 为 i < N
        c = re.search(r'<\s*(\w+)', cond)
        N = c.group(1) if c else "N"

        # body
        body_match = re.search(r'\{(.*)\}', loop_src, re.S)
        body = body_match.group(1) if body_match else ""

        # 生成统一形式
        norm = f"""
for ({idx} = 0; {idx} < {N}; {idx}++ ) {{
    elem = list[{idx}];
    {body}
}}
"""
        return src[:start] + norm + src[end:]


    ###############################################
    # Python: for x in numbers:
    ###############################################
    if lang == "python" and node.type == "for_statement":
        txt = loop_src

        m = re.search(r'for\s+(\w+)\s+in\s+(\w+)\s*:', txt)
        if not m:
            return code

        var = m.group(1)
        seq = m.group(2)

        # body
        body = ""
        try:
            indent = re.match(r'[ \t]*', txt).group(0)
            body = "\n".join([indent + "    " + line for line in txt.split("\n")[1:]])
        except:
            body = ""

        norm = f"""
for i in range(len({seq})):
    {var} = {seq}[i]
{body}
"""
        return src[:start] + norm + src[end:]

    return code


###############################################
# 3) If 结构标准化
###############################################
def normalize_if_statements(code):
    """
    把所有可能的单行 if、三元表达式标准化成显式 if-else：
    """

    # 三元表达式 a if cond else b → if cond: a else: b
    code = re.sub(
        r'(\w+)\s*=\s*(.+?)\s*if\s+(.+?)\s+else\s+(.+)',
        r'if (\3):\n    \1 = \2\nelse:\n    \1 = \4',
        code
    )

    # C/Java 单行 if (cond) stmt; → if(cond){stmt;}
    code = re.sub(
        r'if\s*\((.*?)\)\s*([A-Za-z0-9_]+\s*\(?.*?;)',
        r'if (\1) {\n    \2\n}',
        code
    )

    return code


###############################################
# 4) 列表访问统一化
###############################################
def normalize_list_access(code):
    """
    numbers.get(i) → numbers[i]
    numbers.Add(x) → numbers.append(x)
    """
    code = re.sub(r'(\w+)\.get\((.*?)\)', r'\1[\2]', code)
    code = re.sub(r'(\w+)\.Add\((.*?)\)', r'\1.append(\2)', code)
    return code


###############################################
# 5) 总入口：normalize_code
###############################################
def normalize_code(code, lang):
    lang = lang.lower()
    if lang not in LANG_MAP:
        return code

    parser = Parser()
    parser.set_language(get_language(LANG_MAP[lang]))

    tree = parser.parse(bytes(code, "utf8"))
    root = tree.root_node

    loops = extract_loops(root, lang)

    # 1. 先统一循环
    for lp in reversed(loops):
        code = normalize_loop(code, lp, lang)

    # 2. 再统一 if
    code = normalize_if_statements(code)

    # 3. 再统一列表访问方式
    code = normalize_list_access(code)

    return code

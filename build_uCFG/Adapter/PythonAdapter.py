from tree_sitter import Node
# label 索引
ASSIGNMENT = 0
CONDITIONAL = 1
LOOP = 2
RETURN = 3
ADD = 4
SUB = 5
MUL = 6
DIV = 7
COMPARE = 8
LOGICAL = 9
BITWISE = 10
LABEL_LEN = 11

class PythonAdapter:
    """
    Python 适配器（tree-sitter-python）
    作用：
    - 判断语句类型
    - 提取 if/for/while/def/return
    - 检测赋值语句
    - 扫描表达式运算符（+ - * / < > == and or not 等）
    - 提取 body 子语句
    """

    # =====================
    # Python statement types
    # =====================
    STMT_TYPES = {
        "expression_statement",
        "assignment",
        "augmented_assignment",
        "if_statement",
        "for_statement",
        "while_statement",
        "return_statement",
        "import_statement",
        "import_from_statement",
        "break_statement",
        "continue_statement",
        "assert_statement",
        "pass_statement",
        "try_statement",
        "with_statement",
        "raise_statement",
        "match_statement",
        # suite 是 block，但里面包含 statement，需要递归展开
    }

    LOOP_TYPES = {"for_statement", "while_statement"}

    def __init__(self, source: bytes):
        self.source = source

    # ========== 工具：取文本 ==========
    def text_of(self, node: Node) -> str:
        return self.source[node.start_byte:node.end_byte].decode("utf-8", errors="ignore")

    # ========== 类型判断 ==========
    def is_statement(self, node: Node) -> bool:
        return node.type in self.STMT_TYPES

    def is_subroutine(self, node: Node) -> bool:
        return node.type in ("function_definition", "class_definition")

    def is_conditional(self, node: Node) -> bool:
        return node.type == "if_statement"

    def is_loop(self, node: Node) -> bool:
        return node.type in self.LOOP_TYPES

    def is_return(self, node: Node) -> bool:
        return node.type == "return_statement"

    def is_assignment_stmt(self, node: Node) -> bool:
        return node.type in ("assignment", "augmented_assignment")

    # ========== uIR category ==========
    def category_of(self, node: Node) -> str:
        if self.is_return(node):
            return "Return"
        if self.is_assignment_stmt(node):
            return "Assignment"
        if self.is_loop(node):
            return "Loop"
        if self.is_conditional(node):
            return "Conditional"
        if self.is_subroutine(node):
            return "Subroutine"
        return "Statement"

    # ========== 表达式扫描 ==========
    def scan_expression(self, node: Node, counts: list):
        if node is None:
            return

        t = node.type

        # assignment: a = b
        if t == "assignment":
            # operator is always "=" for python simple assignment
            counts[ASSIGNMENT] += 1
            self.scan_expression(node.child_by_field_name("right"), counts)
            return

        # augmented assignment: += -= *= ...
        if t == "augmented_assignment":
            op_node = node.child_by_field_name("operator")
            op = self.text_of(op_node)
            self._acc_aug_assign(op, counts)
            self.scan_expression(node.child_by_field_name("right"), counts)
            return

        # binary: a + b, a and b, a < b
        if t == "binary_operator":  # tree-sitter-python V0.20
            op = self.text_of(node.child_by_field_name("operator"))
            self._acc_binary_operator(op, counts)
            self.scan_expression(node.child_by_field_name("left"), counts)
            self.scan_expression(node.child_by_field_name("right"), counts)
            return

        # comparison: a < b < c
        if t == "comparison_operator":
            op = self.text_of(node)
            # <, >, <=, >=, ==, !=
            self._acc_compare_operator(op, counts)
            return

        # boolean operators: and / or
        if t == "boolean_operator":
            op = self.text_of(node)
            if op == "and" or op == "or":
                counts[LOGICAL] += 1
            return

        # unary: not x
        if t == "not_operator":
            counts[LOGICAL] += 1
            return

        # bitwise unary: ~x
        if t == "unary_operator":
            op = self.text_of(node)
            if op == "~":
                counts[BITWISE] += 1
            return

        # 递归扫描子节点
        for ch in node.children:
            if self.is_statement(ch):
                continue
            self.scan_expression(ch, counts)

    # ========== 运算符分类 ==========

    def _acc_aug_assign(self, op: str, counts: list):
        if op == "+=":
            counts[ADD] += 1
        elif op == "-=":
            counts[SUB] += 1
        elif op == "*=":
            counts[MUL] += 1
        elif op == "/=":
            counts[DIV] += 1
        elif op in ("&=", "|=", "^="):
            counts[BITWISE] += 1

    def _acc_binary_operator(self, op: str, counts: list):
        if op == "+":
            counts[ADD] += 1
        elif op == "-":
            counts[SUB] += 1
        elif op == "*":
            counts[MUL] += 1
        elif op == "/":
            counts[DIV] += 1
        elif op in ("<", ">", "<=", ">=", "==", "!="):
            counts[COMPARE] += 1
        elif op in ("&", "|", "^"):
            counts[BITWISE] += 1

    def _acc_compare_operator(self, op: str, counts: list):
        if op in ("<", ">", "<=", ">=", "==", "!="):
            counts[COMPARE] += 1

    # ========== 为某条语句提取表达式 ==========
    def collect_exprs_for_statement(self, stmt: Node):
        t = stmt.type
        exprs = []

        if t == "if_statement":
            cond = stmt.child_by_field_name("condition")
            if cond:
                exprs.append(cond)

        elif t == "while_statement":
            cond = stmt.child_by_field_name("condition")
            if cond:
                exprs.append(cond)

        elif t == "for_statement":
            # Python for x in y   -> y is expression
            right = stmt.child_by_field_name("right")
            if right:
                exprs.append(right)

        elif t == "assignment":
            exprs.append(stmt.child_by_field_name("right"))

        elif t == "augmented_assignment":
            exprs.append(stmt.child_by_field_name("right"))

        elif t == "expression_statement":
            exprs.append(stmt.child_by_field_name("expression"))

        elif t == "return_statement":
            val = stmt.child_by_field_name("value")
            if val:
                exprs.append(val)

        return exprs

    # ========== 提取子语句列表（block 展开） ==========
    def iter_statement_children(self, node: Node):
        """
        Python AST 根节点一般是 module
        module → statement_list
        """
        # 1. 根节点 module
        if node.type == "module":
            for ch in node.children:
                yield from self.iter_statement_children(ch)
            return

        # 2. 函数定义
        if node.type == "function_definition":
            yield node
            block = node.child_by_field_name("body")
            if block:
                for ch in block.children:
                    yield from self.iter_statement_children(ch)
            return

        # 3. 代码块 / suite
        if node.type in ("block", "suite"):
            for ch in node.children:
                if self.is_statement(ch):
                    yield ch
                else:
                    yield from self.iter_statement_children(ch)
            return

        # 4. 语句节点本身
        if self.is_statement(node):
            yield node

    # ========== 提取复合语句 body ==========
    def bodies_of(self, stmt: Node):
        t = stmt.type
        bodies = []

        if t in ("function_definition", "class_definition"):
            body = stmt.child_by_field_name("body")
            if body:
                bodies.append(body)

        elif t == "if_statement":
            cons = stmt.child_by_field_name("consequence")
            alt = stmt.child_by_field_name("alternative")
            if cons:
                bodies.append(cons)
            if alt:
                bodies.append(alt)

        elif t in ("for_statement", "while_statement"):
            body = stmt.child_by_field_name("body")
            if body:
                bodies.append(body)

        return bodies

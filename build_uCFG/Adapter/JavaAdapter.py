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

class JavaAdapter:
    """
    Java 的 AST Adapter，负责：
    - 识别语句类型
    - 提取 Loop / Conditional / Return / Assignment
    - 提取表达式运算符 (+ - * / < > && || 等)
    - 构建 block 的子语句列表
    - 获取 if/loop/method 的 body
    """

    # Java 的 statement 统一列表（Tree-sitter-java 定义）
    STMT_TYPES = {
        "local_variable_declaration",
        "expression_statement",
        "if_statement",
        "while_statement",
        "for_statement",
        "enhanced_for_statement",
        "return_statement",
        "block",
        "empty_statement",
        "assert_statement",
        "switch_statement",
        "labeled_statement",
        "throw_statement",
        "synchronized_statement",
        "try_statement",
        "break_statement",
        "continue_statement"
    }

    LOOP_TYPES = {
        "for_statement",
        "enhanced_for_statement",
        "while_statement",
        "do_statement",
    }

    def __init__(self, source: bytes):
        self.source = source

    def text_of(self, node: Node) -> str:
        return self.source[node.start_byte:node.end_byte].decode("utf-8", errors="ignore")

    # ------------------- 节点类型识别 -------------------

    def is_statement(self, node: Node) -> bool:
        return node.type in self.STMT_TYPES

    def is_subroutine(self, node: Node) -> bool:
        return node.type == "method_declaration"

    def is_conditional(self, node: Node) -> bool:
        return node.type == "if_statement"

    def is_loop(self, node: Node) -> bool:
        return node.type in self.LOOP_TYPES

    def is_return(self, node: Node) -> bool:
        return node.type == "return_statement"

    def is_assignment_stmt(self, node: Node) -> bool:
        """Java 的初始化和赋值都可能在 expression_statement 或 local_var_decl 中出现"""
        if node.type == "expression_statement":
            expr = node.child_by_field_name("expression")
            if expr and expr.type == "assignment_expression":
                return True

        if node.type == "local_variable_declaration":
            # int a = b;
            for ch in node.children:
                if ch.type == "variable_declarator":
                    init = ch.child_by_field_name("initializer")
                    if init is not None:
                        return True
        return False

    # ------------------- 分类标签 -------------------

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

    # ------------------- 表达式扫描 -------------------

    def scan_expression(self, node: Node, counts: list):
        if node is None:
            return

        t = node.type

        # assignment: a = b, a += b
        if t == "assignment_expression":
            op_node = node.child_by_field_name("operator")
            op = self.text_of(op_node)
            self._acc_assign_operator(op, counts)

            # 扫描左右表达式
            self.scan_expression(node.child_by_field_name("left"), counts)
            self.scan_expression(node.child_by_field_name("right"), counts)
            return

        # binary op (+ - * / < > == != && ||)
        if t == "binary_expression":
            op_node = node.child_by_field_name("operator")
            if op_node:
                op = self.text_of(op_node)
                self._acc_binary_operator(op, counts)

            self.scan_expression(node.child_by_field_name("left"), counts)
            self.scan_expression(node.child_by_field_name("right"), counts)
            return

        # unary: !a, ~a
        if t == "unary_expression":
            op = self.text_of(node.child_by_field_name("operator"))
            if op == "!":
                counts[LOGICAL] += 1
            elif op == "~":
                counts[BITWISE] += 1
            self.scan_expression(node.child_by_field_name("operand"), counts)
            return

        # 递归扫描子节点
        for ch in node.children:
            # 不要跨到 statement block 中
            if self.is_statement(ch):
                continue
            self.scan_expression(ch, counts)

    # ---- operator 分类 ----

    def _acc_assign_operator(self, op: str, counts: list):
        if op == "=":
            counts[ASSIGNMENT] += 1
        elif op == "+=":
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
        elif op in ("<", ">", "<=", ">=", "==", "!=", "==="):
            counts[COMPARE] += 1
        elif op in ("&&", "||"):
            counts[LOGICAL] += 1
        elif op in ("&", "|", "^"):
            counts[BITWISE] += 1

    # ------------------- 语句表达式提取 -------------------

    def collect_exprs_for_statement(self, stmt: Node):
        t = stmt.type
        exprs = []

        if t == "if_statement":
            cond = stmt.child_by_field_name("condition")
            if cond:
                exprs.append(cond)

        elif t in ("while_statement", "do_statement"):
            cond = stmt.child_by_field_name("condition")
            if cond:
                exprs.append(cond)

        elif t in ("for_statement", "enhanced_for_statement"):
            # Java for(a;b;c)
            for fld in ("init", "condition", "update"):
                nd = stmt.child_by_field_name(fld)
                if nd:
                    exprs.append(nd)

        elif t == "return_statement":
            val = stmt.child_by_field_name("value")
            if val:
                exprs.append(val)

        elif t == "expression_statement":
            expr = stmt.child_by_field_name("expression")
            if expr:
                exprs.append(expr)

        elif t == "local_variable_declaration":
            # int a = expr;
            for ch in stmt.children:
                if ch.type == "variable_declarator":
                    init = ch.child_by_field_name("initializer")
                    if init:
                        exprs.append(init)

        return exprs

    # ------------------- block 子语句 -------------------

    def iter_statement_children(self, node: Node):
        """
        安全遍历 Java AST，避免死循环：
        仅对允许包含语句的容器节点递归。
        """

        CONTAINERS = {
            "program",
            "compilation_unit",
            "class_declaration",
            "class_body",
            "method_declaration",
            "constructor_declaration",
            "block",
            "switch_block"
        }

        # -------- 1) compilation_unit / program --------
        if node.type in ("program", "compilation_unit"):
            for ch in node.children:
                if ch.type in ("class_declaration", "interface_declaration"):
                    yield from self.iter_statement_children(ch)
            return

        # -------- 2) class_declaration / class_body --------
        if node.type in ("class_declaration", "class_body", "interface_declaration"):
            for ch in node.children:
                # 只递归 method / constructor / class_body
                if ch.type in ("method_declaration", "constructor_declaration", "class_body"):
                    yield from self.iter_statement_children(ch)
            return

        # -------- 3) method_declaration / constructor --------
        if node.type in ("method_declaration", "constructor_declaration"):
            # 方法本身作为 Subroutine
            yield node
            body = node.child_by_field_name("body")
            if body is not None:
                yield from self.iter_statement_children(body)
            return

        # -------- 4) block --------
        if node.type == "block":
            for ch in node.children:
                if self.is_statement(ch):
                    yield ch
                elif ch.type in CONTAINERS:
                    yield from self.iter_statement_children(ch)
            return

        # -------- 5) switch_block --------
        if node.type == "switch_block":
            for ch in node.children:
                if self.is_statement(ch):
                    yield ch
                elif ch.type == "block":
                    yield from self.iter_statement_children(ch)
            return

        # -------- 6) base case：如果本身是 statement --------
        if self.is_statement(node):
            yield node

        # -------- 7) 否则不递归（避免死循环） --------
        return

    # ------------------- bodies_of: 获取子 block -------------------

    def bodies_of(self, stmt: Node):
        t = stmt.type
        bodies = []

        if t == "method_declaration":
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

        elif t in self.LOOP_TYPES:
            body = stmt.child_by_field_name("body")
            if body:
                bodies.append(body)

        elif t == "switch_statement":
            body = stmt.child_by_field_name("body")
            if body:
                bodies.append(body)

        return bodies

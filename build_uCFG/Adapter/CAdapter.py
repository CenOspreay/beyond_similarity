from tree_sitter import Node

# 请确保：
# from tree_sitter_languages import get_language
# parser.set_language(get_language("c"))

# label 索引（必须与全局一致）
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


class CAdapter:
    """
    Tree-sitter C 语言适配器
    对应文法：https://github.com/tree-sitter/tree-sitter-c
    """

    # ========= 各类语句节点 ==========
    STMT_TYPES = {
        "if_statement",
        "switch_statement",
        "for_statement",
        "while_statement",
        "do_statement",
        "return_statement",
        "expression_statement",
        "declaration",
        "declaration_list",
        "labeled_statement",
        "compound_statement",   # 花括号 block
        "break_statement",
        "continue_statement",
        "goto_statement",
        "case_statement",
        "default_statement",
    }

    LOOP_TYPES = {
        "for_statement",
        "while_statement",
        "do_statement",
    }

    def __init__(self, source: bytes):
        self.source = source

    # ========== 工具 ==========
    def text_of(self, node: Node) -> str:
        return self.source[node.start_byte:node.end_byte].decode("utf-8", errors="ignore")

    # ========== 判定 ==========
    def is_statement(self, node: Node) -> bool:
        return node.type in self.STMT_TYPES

    def is_subroutine(self, node: Node) -> bool:
        return node.type == "function_definition"

    def is_conditional(self, node: Node) -> bool:
        return node.type in {"if_statement", "switch_statement"}

    def is_loop(self, node: Node) -> bool:
        return node.type in self.LOOP_TYPES

    def is_return(self, node: Node) -> bool:
        return node.type == "return_statement"

    def is_assignment_stmt(self, node: Node) -> bool:
        # declaration: int x = 0;
        if node.type == "declaration":
            for child in node.children:
                if child.type == "init_declarator":  # x=0
                    return True

        # expression_statement: x = y + 1;
        if node.type == "expression_statement":
            expr = node.child_by_field_name("expression")
            if expr and expr.type == "assignment_expression":
                return True
        return False

    # 归类
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
    def scan_expression(self, node: Node, counts):
        if node is None:
            return
        t = node.type

        # assignment expression
        if t == "assignment_expression":
            counts[ASSIGNMENT] += 1
            op = self.text_of(node.child_by_field_name("operator"))
            self._acc_assign_op(op, counts)

            self.scan_expression(node.child_by_field_name("left"), counts)
            self.scan_expression(node.child_by_field_name("right"), counts)
            return

        # binary_expression
        if t == "binary_expression":
            op = self.text_of(node.child_by_field_name("operator"))
            self._acc_binary_op(op, counts)

            self.scan_expression(node.child_by_field_name("left"), counts)
            self.scan_expression(node.child_by_field_name("right"), counts)
            return

        # unary_expression
        if t == "unary_expression":
            op = self.text_of(node.child_by_field_name("operator"))
            if op == "!":
                counts[LOGICAL] += 1
            elif op == "~":
                counts[BITWISE] += 1

            self.scan_expression(node.child_by_field_name("argument"), counts)
            return

        # update_expression (++, --) 不计入算术
        if t == "update_expression":
            self.scan_expression(node.child_by_field_name("argument"), counts)
            return

        # conditional_expression (a ? b : c)
        if t == "conditional_expression":
            counts[CONDITIONAL] += 1
            for fld in ("condition", "consequence", "alternative"):
                x = node.child_by_field_name(fld)
                self.scan_expression(x, counts)
            return

        # 递归扫描所有子节点
        for child in node.children:
            if not self.is_statement(child):
                self.scan_expression(child, counts)

    # ========== 操作符分类 ==========
    def _acc_assign_op(self, op: str, counts):
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

    def _acc_binary_op(self, op: str, counts):
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
        elif op in ("&&", "||"):
            counts[LOGICAL] += 1
        elif op in ("&", "|", "^", "<<", ">>"):
            counts[BITWISE] += 1

    # ========== 提取条件表达式 ==========
    def collect_exprs_for_statement(self, stmt: Node):
        t = stmt.type
        exprs = []

        if t == "if_statement":
            cond = stmt.child_by_field_name("condition")
            if cond:
                exprs.append(cond)

        elif t == "switch_statement":
            cond = stmt.child_by_field_name("condition")
            if cond:
                exprs.append(cond)

        elif t == "while_statement":
            cond = stmt.child_by_field_name("condition")
            if cond:
                exprs.append(cond)

        elif t == "for_statement":
            for fld in ("initializer", "condition", "update"):
                x = stmt.child_by_field_name(fld)
                if x:
                    exprs.append(x)

        elif t == "do_statement":
            cond = stmt.child_by_field_name("condition")
            if cond:
                exprs.append(cond)

        elif t == "return_statement":
            expr = stmt.child_by_field_name("argument")
            if expr:
                exprs.append(expr)

        elif t == "expression_statement":
            expr = stmt.child_by_field_name("expression")
            if expr:
                exprs.append(expr)

        elif t == "declaration":
            # init-declarator → value
            for child in stmt.children:
                if child.type == "init_declarator":
                    val = child.child_by_field_name("value")
                    if val:
                        exprs.append(val)

        return exprs

    # ========== 遍历 block ==========
    def iter_statement_children(self, node: Node):
        """
        C / C++ 根节点通常是 translation_unit
        translation_unit → declaration_list
        """
        # 1. translation_unit
        if node.type == "translation_unit":
            for ch in node.children:
                yield from self.iter_statement_children(ch)
            return

        # 2. 函数定义
        if node.type == "function_definition":
            yield node
            body = node.child_by_field_name("body")
            if body:
                yield from self.iter_statement_children(body)
            return

        # 3. block
        if node.type == "compound_statement":
            for ch in node.children:
                if self.is_statement(ch):
                    yield ch
                else:
                    yield from self.iter_statement_children(ch)
            return

        # 4. 是 statement
        if self.is_statement(node):
            yield node

    # ========== 子 block ==========
    def bodies_of(self, stmt: Node):
        t = stmt.type
        bodies = []

        if t == "function_definition":
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

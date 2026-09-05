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


class CppAdapter:
    """
    Tree-sitter C++ 语言适配器
    grammar: https://github.com/tree-sitter/tree-sitter-cpp
    """

    # ========= 可视为语句的节点 =========
    STMT_TYPES = {
        "if_statement",
        "switch_statement",
        "for_statement",
        "range_for_statement",
        "while_statement",
        "do_statement",
        "return_statement",
        "expression_statement",
        "declaration",
        "declaration_list",
        "labeled_statement",
        "compound_statement",
        "break_statement",
        "continue_statement",
        "goto_statement",
        "case_statement",
        "default_statement",
        "try_statement",
        "catch_clause",
    }

    LOOP_TYPES = {
        "for_statement",
        "range_for_statement",
        "while_statement",
        "do_statement",
    }

    def __init__(self, source: bytes):
        self.source = source

    def text_of(self, node: Node) -> str:
        return self.source[node.start_byte:node.end_byte].decode("utf-8", errors="ignore")

    # ====== 类型判定 ======
    def is_statement(self, node: Node) -> bool:
        return node.type in self.STMT_TYPES

    def is_subroutine(self, node: Node) -> bool:
        """
        function_definition
        · Function with body: (function_definition declarator body)
        · Constructor / destructor also appear as function_definition
        """
        return node.type == "function_definition"

    def is_conditional(self, node: Node) -> bool:
        return node.type in {"if_statement", "switch_statement"}

    def is_loop(self, node: Node) -> bool:
        return node.type in self.LOOP_TYPES

    def is_return(self, node: Node) -> bool:
        return node.type == "return_statement"

    def is_assignment_stmt(self, node: Node) -> bool:
        # declaration with initializer:  int x = 0;
        if node.type == "declaration":
            for child in node.children:
                if child.type == "init_declarator":
                    # declarator = value
                    value = child.child_by_field_name("value")
                    if value is not None:
                        return True

        # expression_statement: x = y + 1;
        if node.type == "expression_statement":
            expr = node.child_by_field_name("expression")
            if expr and expr.type == "assignment_expression":
                return True

        return False

    def category_of(self, node: Node) -> str:
        if self.is_return(node): return "Return"
        if self.is_assignment_stmt(node): return "Assignment"
        if self.is_loop(node): return "Loop"
        if self.is_conditional(node): return "Conditional"
        if self.is_subroutine(node): return "Subroutine"
        return "Statement"

    # ========= 表达式扫描 ==========
    def scan_expression(self, node: Node, counts):
        if node is None:
            return
        t = node.type

        # assignment_expression
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

        # update_expression (++/--)
        if t == "update_expression":
            self.scan_expression(node.child_by_field_name("argument"), counts)
            return

        # conditional_expression (?:)
        if t == "conditional_expression":
            counts[CONDITIONAL] += 1
            for fld in ("condition", "consequence", "alternative"):
                self.scan_expression(node.child_by_field_name(fld), counts)
            return

        # 其他表达式递归
        for child in node.children:
            if not self.is_statement(child):
                self.scan_expression(child, counts)

    # ========= 运算符分类 ==========
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
        if op == "+": counts[ADD] += 1
        elif op == "-": counts[SUB] += 1
        elif op == "*": counts[MUL] += 1
        elif op == "/": counts[DIV] += 1
        elif op in ("<", ">", "<=", ">=", "==", "!="): counts[COMPARE] += 1
        elif op in ("&&", "||"): counts[LOGICAL] += 1
        elif op in ("&", "|", "^", "<<", ">>"): counts[BITWISE] += 1

    # ========= 提取条件表达式 ==========
    def collect_exprs_for_statement(self, stmt: Node):
        t = stmt.type
        exprs = []

        # if / switch
        if t in ("if_statement", "switch_statement"):
            exprs.append(stmt.child_by_field_name("condition"))

        # while / do-while
        elif t in ("while_statement", "do_statement"):
            exprs.append(stmt.child_by_field_name("condition"))

        # for_statement
        elif t == "for_statement":
            for fld in ("initializer", "condition", "update"):
                x = stmt.child_by_field_name(fld)
                if x: exprs.append(x)

        # range-for
        elif t == "range_for_statement":
            exprs.append(stmt.child_by_field_name("right"))
            # 左侧是声明，通常不作为表达式扫描

        # return
        elif t == "return_statement":
            expr = stmt.child_by_field_name("argument")
            if expr: exprs.append(expr)

        # expression_statement
        elif t == "expression_statement":
            exprs.append(stmt.child_by_field_name("expression"))

        # declaration
        elif t == "declaration":
            for child in stmt.children:
                if child.type == "init_declarator":
                    val = child.child_by_field_name("value")
                    if val: exprs.append(val)

        return [e for e in exprs if e is not None]

    # ========= block 遍历 ==========
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

    # ========= 子 block（重要） ==========
    def bodies_of(self, stmt: Node):
        t = stmt.type
        bodies = []

        # function definition
        if t == "function_definition":
            body = stmt.child_by_field_name("body")
            if body: bodies.append(body)

        # if / switch
        elif t == "if_statement":
            cons = stmt.child_by_field_name("consequence")
            alt = stmt.child_by_field_name("alternative")
            if cons: bodies.append(cons)
            if alt: bodies.append(alt)

        elif t == "switch_statement":
            body = stmt.child_by_field_name("body")
            if body: bodies.append(body)

        # loops
        elif t in self.LOOP_TYPES:
            body = stmt.child_by_field_name("body")
            if body: bodies.append(body)

        # try { } catch { }
        elif t == "try_statement":
            block = stmt.child_by_field_name("body")
            if block: bodies.append(block)

            # catch_clause 有自己的 body
            for c in stmt.children:
                if c.type == "catch_clause":
                    block = c.child_by_field_name("body")
                    if block:
                        bodies.append(block)

        return bodies

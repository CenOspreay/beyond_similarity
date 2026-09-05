from tree_sitter import Node
from tree_sitter import Language, Parser
# label indexes (must match your global definition)
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


class CSharpAdapter:
    """
    Tree-sitter C# adapter.
    Grammar: https://github.com/tree-sitter/tree-sitter-c-sharp
    """

    # ===== Statement types =====
    STMT_TYPES = {
        # selection
        "if_statement",
        "switch_statement",
        "switch_section",
        "switch_expression",

        # loop
        "for_statement",
        "foreach_statement",
        "while_statement",
        "do_statement",

        # jump
        "return_statement",
        "break_statement",
        "continue_statement",

        # blocks
        "block",
        "labeled_statement",

        # expressions / declarations
        "expression_statement",
        "local_declaration_statement",
        "declaration_expression",

        # try-catch-finally
        "try_statement",
        "catch_clause",
        "finally_clause",

        # others
        "using_statement",
        "throw_statement",
    }

    LOOP_TYPES = {
        "for_statement",
        "foreach_statement",
        "while_statement",
        "do_statement",
    }

    def __init__(self, src: bytes):
        self.source = src

    def text_of(self, node: Node) -> str:
        return self.source[node.start_byte:node.end_byte].decode("utf8", errors="ignore")

    # ===== Basic Classification =====
    def is_statement(self, node: Node) -> bool:
        return node.type in self.STMT_TYPES

    def is_subroutine(self, node: Node) -> bool:
        # function_declaration or method_declaration or constructor
        return node.type in {
            "method_declaration",
            "constructor_declaration",
            "destructor_declaration",
            "operator_declaration",
            "conversion_operator_declaration",
        }

    def is_conditional(self, node: Node) -> bool:
        return node.type in {
            "if_statement",
            "switch_statement",
            "switch_expression",
        }

    def is_loop(self, node: Node) -> bool:
        return node.type in self.LOOP_TYPES

    def is_return(self, node: Node) -> bool:
        return node.type == "return_statement"

    # detect assignments
    def is_assignment_stmt(self, node: Node) -> bool:
        if node.type == "expression_statement":
            expr = node.child_by_field_name("expression")
            if expr and expr.type == "assignment_expression":
                return True
        if node.type == "local_declaration_statement":
            # int x = 1;
            for c in node.children:
                if c.type == "variable_declarator":
                    value = c.child_by_field_name("initializer")
                    if value is not None:
                        return True
        return False

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

    # ===== Expression Scan =====
    def scan_expression(self, node: Node, counts):
        if node is None:
            return

        t = node.type

        # assignment (x = y)
        if t == "assignment_expression":
            counts[ASSIGNMENT] += 1
            op = self.text_of(node.child_by_field_name("operator"))
            self._acc_assign_op(op, counts)
            self.scan_expression(node.child_by_field_name("left"), counts)
            self.scan_expression(node.child_by_field_name("right"), counts)
            return

        # binary operator
        if t == "binary_expression":
            op = self.text_of(node.child_by_field_name("operator"))
            self._acc_binary_op(op, counts)
            self.scan_expression(node.child_by_field_name("left"), counts)
            self.scan_expression(node.child_by_field_name("right"), counts)
            return

        # unary
        if t == "unary_expression":
            op = self.text_of(node.child_by_field_name("operator"))
            if op == "!":
                counts[LOGICAL] += 1
            elif op == "~":
                counts[BITWISE] += 1
            self.scan_expression(node.child_by_field_name("operand"), counts)
            return

        # conditional expression (a ? b : c)
        if t == "conditional_expression":
            counts[CONDITIONAL] += 1
            for fld in ("condition", "consequence", "alternative"):
                self.scan_expression(node.child_by_field_name(fld), counts)
            return

        # lambda_expression - scan the body
        if t == "lambda_expression":
            body = node.child_by_field_name("body")
            self.scan_expression(body, counts)
            return

        # recursively scan children except statements
        for child in node.children:
            if not self.is_statement(child):
                self.scan_expression(child, counts)

    # ===== Operator Classification =====
    def _acc_assign_op(self, op, counts):
        if op == "+=": counts[ADD] += 1
        elif op == "-=": counts[SUB] += 1
        elif op == "*=": counts[MUL] += 1
        elif op == "/=": counts[DIV] += 1
        elif op in ("&=", "|=", "^="):
            counts[BITWISE] += 1

    def _acc_binary_op(self, op, counts):
        if op == "+": counts[ADD] += 1
        elif op == "-": counts[SUB] += 1
        elif op == "*": counts[MUL] += 1
        elif op == "/": counts[DIV] += 1
        elif op in ("<", ">", "<=", ">=", "==", "!="):
            counts[COMPARE] += 1
        elif op in ("&&", "||"):
            counts[LOGICAL] += 1
        elif op in ("&", "|", "^", "<<", ">>"):
            counts[BITWISE] += 1

    # ===== Extract Expressions for Each Statement =====
    def collect_exprs_for_statement(self, stmt: Node):
        t = stmt.type
        exprs = []

        if t == "if_statement":
            exprs.append(stmt.child_by_field_name("condition"))

        elif t == "switch_statement":
            exprs.append(stmt.child_by_field_name("condition"))

        elif t in ("while_statement", "do_statement"):
            exprs.append(stmt.child_by_field_name("condition"))

        elif t == "for_statement":
            # initializer condition increment
            for fld in ("initializer", "condition", "increment"):
                x = stmt.child_by_field_name(fld)
                if x: exprs.append(x)

        elif t == "foreach_statement":
            exprs.append(stmt.child_by_field_name("right"))

        elif t == "return_statement":
            expr = stmt.child_by_field_name("argument")
            if expr: exprs.append(expr)

        elif t == "expression_statement":
            expr = stmt.child_by_field_name("expression")
            if expr: exprs.append(expr)

        elif t == "local_declaration_statement":
            for c in stmt.children:
                if c.type == "variable_declarator":
                    init = c.child_by_field_name("initializer")
                    if init: exprs.append(init)

        return [e for e in exprs if e is not None]

    # ===== Block Traversal =====
    def iter_statement_children(self, node: Node):
        """
        安全遍历 C# AST，避免深度递归死循环。
        只对“允许包含语句的容器节点”递归。
        """

        STATEMENT_TYPES = self.STMT_TYPES  # 已在 Adapter 中定义

        CONTAINER_NODES = {
            "compilation_unit",
            "namespace_declaration",
            "class_declaration",
            "struct_declaration",
            "interface_declaration",
            "enum_declaration",
            "method_declaration",
            "constructor_declaration",
            "block",
            "switch_body",
            "switch_section",
        }

        # ========= 1) compilation_unit =========
        if node.type == "compilation_unit":
            for ch in node.children:
                # 只递归命名空间 / 类 / 接口等
                if ch.type in (
                        "namespace_declaration",
                        "class_declaration",
                        "struct_declaration",
                        "interface_declaration",
                        "enum_declaration",
                        "method_declaration",
                        "constructor_declaration",
                ):
                    yield from self.iter_statement_children(ch)
            return

        # ========= 2) namespace_declaration =========
        if node.type == "namespace_declaration":
            body = node.child_by_field_name("body")
            if body:
                yield from self.iter_statement_children(body)
            return

        # ========= 3) class / struct / interface / enum =========
        if node.type in (
                "class_declaration",
                "struct_declaration",
                "interface_declaration",
                "enum_declaration",
        ):
            for ch in node.children:
                if ch.type in ("class_body",):
                    yield from self.iter_statement_children(ch)
            return

        # ========= 4) method / constructor =========
        if node.type in ("method_declaration", "constructor_declaration"):
            yield node  # 作为 Subroutine
            body = node.child_by_field_name("body")
            if body:
                yield from self.iter_statement_children(body)
            return

        # ========= 5) block { ... } =========
        if node.type == "block":
            for ch in node.children:
                if ch.type in STATEMENT_TYPES:
                    yield ch
                elif ch.type in CONTAINER_NODES:
                    # 仅递归容器节点
                    yield from self.iter_statement_children(ch)
            return

        # ========= 6) switch_body / switch_section =========
        if node.type in ("switch_body", "switch_section"):
            for ch in node.children:
                if ch.type in STATEMENT_TYPES:
                    yield ch
                elif ch.type == "block":
                    yield from self.iter_statement_children(ch)
            return

        # ========= 7) 基础情况：单一语句 =========
        if node.type in STATEMENT_TYPES:
            yield node
            return

        # ========= 8) 默认：不递归 =========
        return

    # ===== Sub blocks =====
    def bodies_of(self, stmt: Node):
        t = stmt.type
        bodies = []

        if t in {
            "method_declaration",
            "constructor_declaration",
            "destructor_declaration",
        }:
            b = stmt.child_by_field_name("body")
            if b: bodies.append(b)

        if t == "if_statement":
            cons = stmt.child_by_field_name("consequence")
            alt = stmt.child_by_field_name("alternative")
            if cons: bodies.append(cons)
            if alt: bodies.append(alt)

        elif t == "switch_statement":
            body = stmt.child_by_field_name("body")
            if body: bodies.append(body)

        elif t in self.LOOP_TYPES:
            body = stmt.child_by_field_name("body")
            if body: bodies.append(body)

        elif t == "try_statement":
            block = stmt.child_by_field_name("body")
            if block: bodies.append(block)
            for c in stmt.children:
                if c.type == "catch_clause":
                    b = c.child_by_field_name("body")
                    if b: bodies.append(b)
                if c.type == "finally_clause":
                    b = c.child_by_field_name("body")
                    if b: bodies.append(b)

        return bodies

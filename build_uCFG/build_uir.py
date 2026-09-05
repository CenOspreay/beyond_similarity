from dataclasses import dataclass, asdict
from typing import List, Optional, Dict, Any
from tree_sitter import Parser, Node, Language
from tree_sitter_languages import get_language
from build_uCFG.Adapter.JavaAdapter import JavaAdapter
from build_uCFG.Adapter.CAdapter import CAdapter
from build_uCFG.Adapter.PythonAdapter import PythonAdapter
from build_uCFG.Adapter.CSharpAdapter import CSharpAdapter
from build_uCFG.Adapter.CppAdapter import CppAdapter

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

Language.build_library(
    'build_uCFG/build/my-languages.so',
    [
        'build_uCFG/build/tree-sitter-c-sharp'
    ]
)
CS_LANGUAGE = Language('build_uCFG/build/my-languages.so', 'c_sharp')


LANG_ALIASES = {
        "java": ["java"],
        "python": ["python", "py"],
        "c": ["c"],
        "cpp": ["cpp", "c++", "cc", "cp"],
        "cs": ["cs", "c#", "csharp"],
        "javascript": ["javascript", "js"]
    }

@dataclass
class UIR:
    id: int
    label: List[int]
    category: str
    NS: Optional[int]
    NCS: Optional[int]
    NLS: Optional[int]
    NSS: Optional[int]


class UIRBuilder:
    def __init__(self, label_mode: str = "binary"):
        assert label_mode in ("binary", "count")
        self.label_mode = label_mode
        self.uirs: List[UIR] = []
        self.next_id = 1

        # 栈：最近条件 / 最近循环
        self.cond_stack: List[int] = []   # NCS
        self.loop_stack: List[int] = []   # NLS

    # ========== 对外入口 ==========
    def normalize_lang(self, lang: str) -> str:
        lang = lang.lower()
        for std, alias_list in LANG_ALIASES.items():
            if lang in alias_list:
                return std
        raise NotImplementedError(f"Unknown language alias: {lang}")

    def build_uir_lang(self, code: str, language: str) -> List[Dict[str, Any]]:
        """
        根据输入的语言类型和源代码，选择相应的适配器并构建 uIR。

        :param code: 源代码字符串
        :param language: 编程语言类型（'javascript', 'python', 'java', 'c', 'cpp', 'csharp'）
        :return: 构建的 uIR 列表
        """

        # 1) 标准化，不允许出现 py/c#/c++
        language = self.normalize_lang(language)

        TREE_SITTER_NAME = {
            "java": "java",
            "python": "python",
            "c": "c",
            "cpp": "cpp",
            "cs": "c_sharp",
            "javascript": "javascript",
        }

        # 2) 根据标准语言选择 Adapter
        if language == "java":
            adapter = JavaAdapter(code.encode("utf-8"))
        elif language == "python":
            adapter = PythonAdapter(code.encode("utf-8"))
        elif language == "c":
            adapter = CAdapter(code.encode("utf-8"))
        elif language == "cpp":
            adapter = CppAdapter(code.encode("utf-8"))
        elif language == "cs":
            adapter = CSharpAdapter(code.encode("utf-8"))
        else:
            raise NotImplementedError(f"Language '{language}' not supported.")

        # 解析语法树，处理顶层语句
        parser = Parser()
        parser.set_language(get_language(TREE_SITTER_NAME[language]))
        tree = parser.parse(code.encode("utf-8"))

        root = tree.root_node
        top_stmts = list(adapter.iter_statement_children(root))
        self._process_block(top_stmts, adapter)

        # 二值化标签（binary）或计数标签（count）
        if self.label_mode == "binary":
            for u in self.uirs:
                u.label = [1 if v > 0 else 0 for v in u.label]

        return [asdict(u) for u in self.uirs]

    # ========== 工具函数 ==========
    def _new_uir(self, category: str) -> UIR:
        u = UIR(
            id=self.next_id,
            label=[0] * LABEL_LEN,
            category=category,
            NS=None,
            NCS=self.cond_stack[-1] if self.cond_stack else None,
            NLS=self.loop_stack[-1] if self.loop_stack else None,
            NSS=None,
        )
        self.next_id += 1
        self.uirs.append(u)
        return u

    def _find(self, uid: int) -> UIR:
        for u in self.uirs:
            if u.id == uid:
                return u
        raise KeyError(uid)

    # ========== 核心：按 Block 处理同层级语句 ==========
    def _process_block(self, stmt_nodes: List[Node], ad) -> Optional[int]:
        """
        处理同一层级的一组语句：
        - 为每条语句创建 UIR
        - 设置它们的 NS
        - 递归处理子 block
        返回：该 block 第一条语句对应的 UIR id（用于父节点 NSS）
        """
        first_uid: Optional[int] = None
        prev_uid: Optional[int] = None

        for stmt in stmt_nodes:
            uid = self._handle_single_stmt(stmt, ad)
            if uid is None:
                continue

            if first_uid is None:
                first_uid = uid

            if prev_uid is not None:
                prev_uir = self._find(prev_uid)
                prev_uir.NS = uid

            prev_uid = uid

        # 最后一条的 NS 默认就是 None
        return first_uid

    # ========== 处理一条语句 ==========
    def _handle_single_stmt(self, stmt: Node, ad) -> Optional[int]:
        cat = ad.category_of(stmt)
        u = self._new_uir(cat)

        # 主类型位
        if cat == "Assignment":
            u.label[ASSIGNMENT] += 1
        elif cat == "Conditional":
            u.label[CONDITIONAL] += 1
        elif cat == "Loop":
            u.label[LOOP] += 1
        elif cat == "Return":
            u.label[RETURN] += 1

        # 表达式运算符扫描
        exprs = ad.collect_exprs_for_statement(stmt)
        for e in exprs:
            ad.scan_expression(e, u.label)

        # ========== 复合语句：有子 block ==========
        # 1) 函数（可选，看你要不要当成一个 Subroutine 节点）
        if ad.is_subroutine(stmt):
            # 函数一般不算 NCS/NLS，这里不改栈
            bodies = ad.bodies_of(stmt)  # 对 JS 来说只有一个 body block
            for body in bodies:
                child_stmts = list(ad.iter_statement_children(body))
                first_child_uid = self._process_block(child_stmts, ad)
                if first_child_uid is not None and u.NSS is None:
                    u.NSS = first_child_uid
            return u.id

        # 2) 条件语句 if / switch
        if ad.is_conditional(stmt):
            self.cond_stack.append(u.id)
            bodies = ad.bodies_of(stmt)
            # 第一个 body 是 then 分支，NSS 只指向它的第一条语句
            for idx, body in enumerate(bodies):
                child_stmts = list(ad.iter_statement_children(body))
                first_child_uid = self._process_block(child_stmts, ad)
                if idx == 0 and first_child_uid is not None:
                    u.NSS = first_child_uid
            self.cond_stack.pop()
            return u.id

        # 3) 循环语句 for / while / ...
        if ad.is_loop(stmt):
            self.loop_stack.append(u.id)
            bodies = ad.bodies_of(stmt)   # 正常只会有一个 body
            for body in bodies:
                child_stmts = list(ad.iter_statement_children(body))
                first_child_uid = self._process_block(child_stmts, ad)
                if first_child_uid is not None and u.NSS is None:
                    u.NSS = first_child_uid
            self.loop_stack.pop()
            return u.id

        # 4) 其他简单语句
        return u.id


# 对外接口
# 对外接口
def build_uir(source_code: str, language: str, label_mode: str = "binary") -> List[Dict[str, Any]]:
    builder = UIRBuilder(label_mode=label_mode)
    return builder.build_uir_lang(source_code, language)



# tests = {
#     "java": """
# public int sumEven(int n) {
#     int s = 0;
#     for (int i = 0; i < n; i++) {
#         if (i % 2 == 0) {
#             s += i;
#         }
#     }
#     return s;
# }
# """,
#
#     "python": """
# def sum_even(n):
#     s = 0
#     for i in range(n):
#         if i % 2 == 0:
#             s += i
#     return s
# """,
#
#     "c": """
# int sum_even(int n) {
#     int s = 0;
#     for (int i = 0; i < n; i++) {
#         if (i % 2 == 0) {
#             s += i;
#         }
#     }
#     return s;
# }
# """,
#
#     "cpp": """
# int sum_even(int n) {
#     int s = 0;
#     for (int i = 0; i < n; i++) {
#         if (i % 2 == 0) {
#             s += i;
#         }
#     }
#     return s;
# }
# """,
#
#     "cs": """
# public int SumEven(int n) {
#     int s = 0;
#     for (int i = 0; i < n; i++) {
#         if (i % 2 == 0) {
#             s += i;
#         }
#     }
#     return s;
# }
# """
# }
#
#
# for lang, code in tests.items():
#     print("="*80)
#     print("Testing language:", lang)
#     uirs = build_uir(code, language=lang, label_mode="binary")
#     for u in uirs:
#         print(u)

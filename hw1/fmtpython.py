import os
import sys
from dataclasses import dataclass
from typing import Generator, Self, Callable

from tree_sitter import Language, Node, Parser, Tree

Language.build_library(
    "build/my-languages.so",
    [os.path.join(os.getcwd(), "tree-sitter-python")],
)

PY_LANGUAGE = Language("build/my-languages.so", "python")


def traverse_tree(tree: Tree) -> Generator[Node, None, None]:
    cursor = tree.walk()

    visited_children = False
    while True:
        if not visited_children:
            yield cursor.node
            if not cursor.goto_first_child():
                visited_children = True
        elif cursor.goto_next_sibling():
            visited_children = False
        elif not cursor.goto_parent():
            break


def delete_comments(program: bytes) -> bytes:
    changed_program = []
    parser = Parser()
    parser.set_language(PY_LANGUAGE)
    tree = parser.parse(program)
    prev = 0
    for node in traverse_tree(tree):
        if node.type == "comment":
            newline_index = program.rfind(b"\n", prev, node.start_byte)
            if newline_index == -1:
                newline_index = prev

            before_comment = program[newline_index + 1: node.start_byte]
            if before_comment.strip() == b"":
                changed_program.append(program[prev:newline_index])
                prev = newline_index + 1 + len(before_comment)

            changed_program.append(program[prev: node.start_byte])
            prev = node.end_byte

    changed_program.append(program[prev:])
    return b"".join(changed_program)


def replace_tabs_with_spaces(program: bytes) -> bytes:
    changed_program = []
    parser = Parser()
    parser.set_language(PY_LANGUAGE)
    tree = parser.parse(program)
    prev = 0
    for node in traverse_tree(tree):
        if node.type == "string":
            string_slice = bytes(
                program[prev: node.start_byte].decode(
                    "utf-8").expandtabs(4), "utf-8"
            )
            changed_program.append(string_slice)
            changed_program.append(program[node.start_byte: node.end_byte])
            prev = node.end_byte
    last_slice = bytes(program[prev:].decode("utf-8").expandtabs(4), "utf-8")
    changed_program.append(last_slice)
    return b"".join(changed_program)


def _delete_meaningless_spaces_one_line(line: str) -> str:
    res = []
    words = [word for word in line.split() if word]
    for i in range(len(words)):
        res.append(words[i])
        if (
            i + 1 < len(words)
            and (words[i][-1].isalnum() or words[i][-1] == "_")
            and (words[i + 1][0].isalnum() or words[i + 1][0] == "_")
        ):
            res.append(" ")
    return "".join(res)


def _indentation(line: str) -> str:
    length = 0
    for c in line:
        if c != " ":
            break
        length += 1
    return line[:length]


def _delete_meaningless_spaces(program: str) -> str:
    changed_program = []
    for line in program.split(os.linesep):
        line_without_spaces = _delete_meaningless_spaces_one_line(line)
        if line_without_spaces:
            changed_program.append(_indentation(line) + line_without_spaces)
        else:
            changed_program.append(line_without_spaces)
    return "\n".join(changed_program)


def delete_meaningless_spaces(program: bytes) -> bytes:
    changed_program = []
    parser = Parser()
    parser.set_language(PY_LANGUAGE)
    tree = parser.parse(program)

    prev = 0
    visited_string = False
    for node in traverse_tree(tree):
        if node.type == "string":
            visited_string = True
            not_string_part = _delete_meaningless_spaces(
                program[prev: node.start_byte].decode("utf-8").lstrip(" ")
            )
            changed_program.append(bytes(not_string_part, "utf-8"))

            newline_index = program.rfind(b"\n", prev, node.start_byte)
            if newline_index == -1:
                newline_index = prev
            before_string = program[newline_index + 1: node.start_byte]
            if before_string.strip() == b"":
                changed_program.append(before_string)

            string_part = program[node.start_byte: node.end_byte]

            last_char = changed_program[-1].decode("utf-8")[-1]
            new_char = string_part.decode("utf-8")[0]
            if (last_char.isalnum() or last_char == "_") and (
                new_char.isalnum() or new_char == "_"
            ):
                changed_program.append(b" ")

            changed_program.append(string_part)

            prev = node.end_byte

    end_part = program[prev:].decode("utf-8")
    if visited_string:
        end_part = end_part.lstrip()

    changed_program.append(
        bytes(_delete_meaningless_spaces(end_part), "utf-8"))
    return b"".join(changed_program)


class Scope:
    def __init__(self, parent: Self or None):
        self.parent = parent
        self.names = dict()
        self.namespaces = dict()

    def insert(self, old_name: str, new_name: str) -> None:
        self.names[old_name] = new_name

    def get(self, old_name: str) -> str or None:
        if old_name in self.names:
            return self.names[old_name]
        if self.parent is None:
            return None
        return self.parent.get(old_name)

    def insert_namespace(self, old_name: str, namespace: dict[str, str]):
        self.namespaces[old_name] = namespace

    def get_namespace(self, old_name: str) -> Self or None:
        if old_name in self.namespaces:
            return self.namespaces[old_name]
        if self.parent is None:
            return None
        return self.parent.get_namespace(old_name)


@dataclass
class Change:
    start_byte: int
    end_byte: int
    replacement: bytes


def ignore(*args, **kwargs):
    return


class NamesReplacer:
    def __init__(self):
        self.global_scope = Scope(None)
        self.changes = []
        self.last_id = 0

    def make_changes(self, tree: Tree) -> list[Change]:
        root = tree.root_node
        self._visit(root, self.global_scope)
        return sorted(self.changes, key=lambda change: change.start_byte)

    def _create_new_name(self) -> str:
        name = f"name_{self.last_id}"
        self.last_id += 1
        return name

    def _visit(
        self,
        node: Node or None,
        scope: Scope,
        visit_identifier: Callable[[Node, Scope], None] = None,
        visit_function_definition: Callable[[Node, Scope], None] = None,
        visit_keyword_argument: Callable[[Node, Scope], None] = None,
        visit_assignment: Callable[[Node, Scope], None] = None,
        visit_type: Callable[[Node, Scope], None] = None,
    ) -> Callable[[], None] or None:
        if visit_identifier is None:
            visit_identifier = self._rename_identifier
        if visit_function_definition is None:
            visit_function_definition = self._visit_function_definition
        if visit_assignment is None:
            visit_assignment = self._visit_assignment
        if visit_type is None:
            visit_type = self._visit_type

        if node is None:
            return None
        elif node.type == "class_definition":
            return self._visit_class_definition(node, scope)
        elif node.type == "function_definition":
            return visit_function_definition(node, scope)
        elif node.type == "for_statement":
            self._visit_for_statement(node, scope)
        elif node.type == "assignment":
            visit_assignment(node, scope)
        elif node.type == "attribute":
            self._visit_attribute(node, scope, visit_identifier)
        elif node.type == "call":
            self._visit_call(node, scope)
        elif node.type == "keyword_argument" and visit_keyword_argument:
            visit_keyword_argument(node, scope)
        elif node.type == "named_expression":
            self._visit_named_expression(node, scope)
        elif node.type == "type":
            visit_type(node, scope)
        elif node.type == "identifier":
            visit_identifier(node, scope)
        else:
            visit_class_or_function_bodies = []
            for child in node.children:
                visit_class_or_function_bodies.append(
                    self._visit(
                        child,
                        scope,
                        visit_identifier=visit_identifier,
                        visit_function_definition=visit_function_definition,
                        visit_keyword_argument=visit_keyword_argument,
                        visit_type=visit_type,
                        visit_assignment=visit_assignment,
                    )
                )
            for visit_class_or_function_body in visit_class_or_function_bodies:
                if visit_class_or_function_body:
                    visit_class_or_function_body()
        return None

    def _visit_function_definition(
        self, node: Node, scope: Scope
    ) -> Callable[[], None]:
        name = node.child_by_field_name("name")
        self._define_and_rename_identifier(name, scope)
        function_namespace = Scope(None)
        function_scope = Scope(scope)

        def visit_type(node: Node, ignored_scope: Scope):
            self._visit_type(node, scope, visit_type=visit_type)

        parameters = node.child_by_field_name("parameters")
        self._visit(
            parameters,
            function_namespace,
            visit_identifier=self._define_and_rename_identifier,
            visit_type=visit_type
        )
        function_scope.names.update(function_namespace.names)
        scope.insert_namespace(str(name.text), function_namespace)
        self._visit(node.child_by_field_name("return_type"), scope)
        return lambda: (self._visit(node.child_by_field_name("body"), function_scope))

    def _visit_call(self, node: Node, scope: Scope):
        function = node.child_by_field_name("function")
        if function.type == "attribute":
            function_namespace = self._visit_attribute(function, scope)
        elif function.type == "identifier":
            self._visit(function, scope)
            function_namespace = scope.get_namespace(str(function.text))
        else:
            raise RuntimeError(f"Wrong function type: `{function.text}`")

        def visit_keyword_argument(node: Node, scope: Scope):
            if function_namespace:
                name = node.child_by_field_name("name")
                self._visit(name, function_namespace)
            value = node.child_by_field_name("value")
            self._visit(value, scope)

        arguments = node.child_by_field_name("arguments")
        self._visit(arguments, scope,
                    visit_keyword_argument=visit_keyword_argument)

    def _visit_class_definition(self, node: Node, scope: Scope) -> Callable[[], None]:
        name = node.child_by_field_name("name")
        self._define_and_rename_identifier(name, scope)

        self._visit(node.child_by_field_name("superclasses"), scope)

        class_scope = Scope(scope)
        class_namespace = Scope(None)
        scope.insert_namespace(str(name.text), class_namespace)

        methods = [
            method
            for method in node.child_by_field_name("body").children
            if method.type == "function_definition"
        ]
        method_scopes = []
        for method in methods:
            if method.type != "function_definition":
                continue

            def get_first_param(node: Node) -> str:
                if node.type == "typed_parameter":
                    return str(node.child(0).text)
                return str(node.text)

            name = method.child_by_field_name("name")
            method_namespace = Scope(None)
            method_scope = Scope(class_scope)
            method_scopes.append(method_scope)

            parameters = method.child_by_field_name("parameters")
            first_param = get_first_param(parameters.child(1))
            scope.insert_namespace(first_param, class_namespace)

            def visit_type(node: Node, ignored_scope: scope):
                self._visit_type(node, class_scope, visit_type=visit_type)

            self._visit(
                parameters,
                method_namespace,
                visit_identifier=self._define_and_rename_identifier,
                visit_type=visit_type,
            )
            method_scope.names.update(method_namespace.names)
            scope.insert_namespace(str(name.text), method_namespace)
            self._visit(method.child_by_field_name("return_type"), scope)

        def visit_assignment(node: Node, scope: Scope):
            left = node.child_by_field_name("left")
            if left.type != "identifier" and left.type != "pattern_list":
                self._visit_assignment(node, scope)
                return
            self._visit(
                node.child_by_field_name("left"),
                scope,
                visit_identifier=ignore,
            )
            self._visit(node.child_by_field_name("type"), scope)
            self._visit(node.child_by_field_name("right"), scope)

        def visit_body():
            self._visit(
                node.child_by_field_name("body"),
                class_scope,
                visit_function_definition=ignore,
                visit_assignment=visit_assignment,
            )
            for method, method_scope in zip(methods, method_scopes):
                self._visit(method.child_by_field_name("body"), method_scope)

        return visit_body

    def _visit_assignment(self, node: Node, scope: Scope):
        self._visit(node.child_by_field_name("left"),
                    scope, visit_identifier=self._define_and_rename_identifier)
        self._visit(node.child_by_field_name("type"), scope)
        self._visit(node.child_by_field_name("right"), scope)

    def _visit_attribute(
        self,
        node: Node,
        scope: Scope,
        visit_identifier: Callable[[Node, Scope], None] = None,
    ) -> Scope or None:
        if visit_identifier is None:
            visit_identifier = self._rename_identifier
        if node.type == "identifier":
            if scope is None:
                return None
            visit_identifier(node, scope)
            return scope.get_namespace(str(node.text))
        elif node.type == "attribute":
            namespace = self._visit_attribute(
                node.child_by_field_name("object"), scope, visit_identifier
            )
            if namespace:
                attribute = node.child_by_field_name("attribute")
                self._rename_identifier(attribute, namespace)
                return scope.get_namespace(str(attribute.text))
        else:
            self._visit(node, scope)
        return None

    def _visit_for_statement(self, node: Node, scope: Scope):
        self._visit(
            node.child_by_field_name("left"),
            scope,
            visit_identifier=self._define_and_rename_identifier,
        )
        self._visit(node.child_by_field_name("right"), scope)
        inside_scope = Scope(scope)
        self._visit(node.child_by_field_name("body"), inside_scope)

    def _visit_named_expression(self, node: Node, scope: Scope):
        self._visit(
            node.child(0),
            scope,
            visit_identifier=self._define_and_rename_identifier,
        )
        self._visit(node.child_by_field_name("value"), scope)

    def _visit_type(
        self, node: Node, scope: Scope, visit_type: Callable[[Node, Scope], None] = None
    ):
        for child in node.children:
            self._visit(child, scope, visit_type=visit_type)

    def _rename_identifier(self, node: Node, scope: Scope):
        if new_name := scope.get(node.text):
            self.changes.append(
                Change(node.start_byte, node.end_byte, new_name))

    def _define_and_rename_identifier(self, node: Node, scope: Scope):
        new_name = scope.get(node.text) or self._create_new_name()
        scope.insert(node.text, new_name)
        self.changes.append(Change(node.start_byte, node.end_byte, new_name))


def replace_names(program: bytes) -> bytes:
    changed_program = []
    parser = Parser()
    parser.set_language(PY_LANGUAGE)
    tree = parser.parse(program)

    changes = NamesReplacer().make_changes(tree)
    prev = 0
    for change in changes:
        changed_program.append(program[prev: change.start_byte])
        changed_program.append(bytes(change.replacement, "utf-8"))
        prev = change.end_byte
    changed_program.append(program[prev:])

    return b"".join(changed_program)


program = bytes(sys.stdin.read(), "utf-8")
program = delete_comments(program)
program = replace_tabs_with_spaces(program)
program = delete_meaningless_spaces(program)
program = replace_names(program)
print(program.decode("utf-8"))

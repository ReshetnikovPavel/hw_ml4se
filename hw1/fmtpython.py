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
    print(tree.root_node.sexp())
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
            not_string_part = (
                program[prev: node.start_byte].decode("utf-8").lstrip(" ")
            )
            changed_program.append(
                bytes(_delete_meaningless_spaces(not_string_part), "utf-8")
            )

            newline_index = program.rfind(b"\n", prev, node.start_byte)
            if newline_index == -1:
                newline_index = prev
            before_string = program[newline_index + 1: node.start_byte]
            if before_string.strip() == "":
                changed_program.append(before_string)

            string_part = program[node.start_byte: node.end_byte]
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
    ):
        if visit_identifier is None:
            visit_identifier = self._rename_identifier
        if visit_function_definition is None:
            visit_function_definition = self._visit_function_definition

        if node is None:
            return
        elif node.type == "class_definition":
            self._visit_class_definition(node, scope)
        elif node.type == "function_definition":
            visit_function_definition(node, scope)
        elif node.type == "for_statement":
            self._visit_for_statement(node, scope)
        elif node.type == "assignment":
            self._visit_assignment(node, scope)
        elif node.type == "attribute":
            self._visit_attribute(node, scope, visit_identifier)
        elif node.type == "call":
            self._visit_call(node, scope)
        elif node.type == "keyword_argument" and visit_keyword_argument:
            visit_keyword_argument(node, scope)
        elif node.type == "typed_parameter":
            self._visit_typed_parameter(node, scope, visit_identifier)
        elif node.type == "type":
            for child in node.children:
                self._visit(child, scope)
        elif node.type == "identifier":
            visit_identifier(node, scope)
        else:
            for child in node.children:
                self._visit(
                    child,
                    scope,
                    visit_identifier=visit_identifier,
                    visit_function_definition=visit_function_definition,
                    visit_keyword_argument=visit_keyword_argument,
                )

    def _visit_function_definition(self, node: Node, scope: Scope):
        name = node.child_by_field_name("name")
        self._define_and_rename_identifier(name, scope)
        parameters_scope = Scope(scope)
        inside_scope = Scope(parameters_scope)
        parameters = node.child_by_field_name("parameters")
        self._visit(parameters, parameters_scope,
                    self._define_and_rename_identifier)
        scope.insert_namespace(str(name.text), parameters_scope)
        self._visit(node.child_by_field_name("return_type"), scope)
        self._visit(node.child_by_field_name("body"), inside_scope)

    def _visit_typed_parameter(
        self,
        node: Node,
        scope: Scope,
        visit_identifier: Callable[[Node, Scope], None] = None,
    ):
        type_node = node.child_by_field_name("type")
        if type_node.type == "identifier":
            class_namespace = scope.get_namespace(str(type_node.text))
            scope.insert_namespace(str(node.child(0).text), class_namespace)

        for child in node.children:
            self._visit(child, scope, visit_identifier=visit_identifier)

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

    def _visit_class_definition(self, node: Node, scope: Scope):
        print(node.child_by_field_name("name").text, node.sexp())
        name = node.child_by_field_name("name")
        self._define_and_rename_identifier(name, scope)

        self._visit(node.child_by_field_name("superclasses"), scope)

        class_scope = Scope(scope)
        scope.insert_namespace(str(name.text), class_scope)

        methods = node.child_by_field_name("body").children_by_field_name(
            "function_definition"
        )
        method_scopes = []
        for method in methods:
            if method.type != "function_definition":
                continue

            def get_first_param(node: Node) -> str:
                if node.type == "typed_parameter":
                    return str(node.child(0).text)
                return str(node.text)

            name = method.child_by_field_name("name")
            parameters_scope = Scope(scope)
            method_scope = Scope(parameters_scope)
            method_scopes.append(method_scope)

            parameters = method.child_by_field_name("parameters")
            first_param = get_first_param(parameters.child(1))
            scope.insert_namespace(first_param, class_scope)

            self._visit(
                parameters, parameters_scope, self._define_and_rename_identifier
            )
            scope.insert_namespace(str(name.text), parameters_scope)
            self._visit(method.child_by_field_name("return_type"), scope)

        for method, method_scope in zip(methods, method_scopes):
            self._visit(method.child_by_field_name("body"), method_scope)

        def ignore(*args, **kwargs):
            return

        self._visit(
            node.child_by_field_name("body"),
            class_scope,
            visit_function_definition=ignore,
        )

    def _visit_assignment(self, node: Node, scope: Scope):
        self._visit(
            node.child_by_field_name("left"),
            scope,
            visit_identifier=self._define_and_rename_identifier,
        )
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
                visit_identifier(attribute, namespace)
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

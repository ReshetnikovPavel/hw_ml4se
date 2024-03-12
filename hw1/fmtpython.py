import os
import sys
from dataclasses import dataclass
from typing import Generator, Self

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


def delete_comments(python_program: str) -> str:
    changed_program = []
    parser = Parser()
    parser.set_language(PY_LANGUAGE)
    tree = parser.parse(bytes(python_program, "utf-8"))
    print(tree.root_node.sexp())
    prev = 0
    for node in traverse_tree(tree):
        if node.type == "comment":
            changed_program.append(python_program[prev: node.start_byte])
            prev = node.end_byte
    changed_program.append(python_program[prev:])
    return "".join(changed_program)


def replace_tabs_with_spaces(python_program: str) -> str:
    changed_program = []
    parser = Parser()
    parser.set_language(PY_LANGUAGE)
    tree = parser.parse(bytes(python_program, "utf-8"))

    prev = 0
    for node in traverse_tree(tree):
        if node.type == "string":
            changed_program.append(
                python_program[prev: node.start_byte].expandtabs(4))
            changed_program.append(
                python_program[node.start_byte: node.end_byte])
            prev = node.end_byte
    changed_program.append(python_program[prev:].expandtabs(4))
    return "".join(changed_program)


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


def _delete_meaningless_spaces(python_program: str) -> str:
    changed_program = []
    for line in python_program.split(os.linesep):
        line_without_spaces = _delete_meaningless_spaces_one_line(line)
        if line_without_spaces:
            changed_program.append(_indentation(line) + line_without_spaces)
        else:
            changed_program.append(line_without_spaces)
    return os.linesep.join(changed_program)


def delete_meaningless_spaces(python_program: str) -> str:
    changed_program = []
    parser = Parser()
    parser.set_language(PY_LANGUAGE)
    tree = parser.parse(bytes(python_program, "utf-8"))

    prev = 0
    visited_string = False
    for node in traverse_tree(tree):
        if node.type == "string":
            visited_string = True
            not_string_part = python_program[prev: node.start_byte].lstrip(" ")
            changed_program.append(_delete_meaningless_spaces(not_string_part))

            string_part = python_program[node.start_byte: node.end_byte]
            changed_program.append(string_part)

            prev = node.end_byte

    end_part = python_program[prev:]
    if visited_string:
        end_part = end_part.lstrip()

    changed_program.append(_delete_meaningless_spaces(end_part))
    return "".join(changed_program)


class Scope:
    def __init__(self, parent: Self or None):
        self.parent = parent
        self.names = dict()

    def insert(self, old_name: str, new_name: str) -> None:
        if old_name in self.names:
            raise KeyError(
                f'Old name "{old_name}" was already added with'
                f' new name "{self.names[old_name]}"'
            )
        self.names[old_name] = new_name

    def get(self, old_name: str) -> str or None:
        if old_name in self.names:
            return self.names[old_name]
        if self.parent is None:
            return None
        return self.parent.get(old_name)


@dataclass
class Change:
    start_byte: int
    end_byte: int
    replacement: str


class NamesReplacer:
    def __init__(self):
        self.global_scope = Scope(None)
        self.changes = []
        self.last_id = 0

    def make_changes(self, tree: Tree) -> list[Change]:
        root = tree.root_node
        # print(root.sexp())
        self._visit(root, self.global_scope)
        return self.changes

    def _create_new_name(self) -> str:
        name = f"name_{self.last_id}"
        self.last_id += 1
        return name

    def _visit(self, node: Node or None, scope: Scope):
        if node is None:
            return
        elif node.type == "class_definition":
            self._visit_class_definition(node, scope)
        elif node.type == "identifier":
            self._visit_identifier(node, scope)
        else:
            for child in node.children:
                self._visit(child, scope)

    def _visit_class_definition(self, node: Node, scope: Scope):
        name = node.child_by_field_name("name")
        scope.insert(name.text, self._create_new_name())
        self._visit(name, scope)

        self._visit(node.child_by_field_name("superclasses"), scope)

        inside_class_scope = Scope(scope)
        self._visit(node.child_by_field_name("body"), inside_class_scope)

    def _visit_identifier(self, node: Node, scope: Scope):
        if new_name := scope.get(node.text):
            self.changes.append(
                Change(node.start_byte, node.end_byte, new_name))


def replace_names(python_program: str) -> str:
    changed_program = []
    parser = Parser()
    parser.set_language(PY_LANGUAGE)
    tree = parser.parse(bytes(python_program, "utf-8"))

    changes = NamesReplacer().make_changes(tree)
    prev = 0
    for change in changes:
        changed_program.append(python_program[prev:change.start_byte])
        changed_program.append(change.replacement)
        prev = change.end_byte
    changed_program.append(python_program[prev:])

    return "".join(changed_program)


program = sys.stdin.read()
program = delete_comments(program)
program = replace_tabs_with_spaces(program)
program = delete_meaningless_spaces(program)
program = replace_names(program)
print(program)

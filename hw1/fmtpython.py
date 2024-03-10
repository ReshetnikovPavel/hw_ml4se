import os
import sys
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
    # print(tree.root_node.sexp())
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
        if i + 1 < len(words) and words[i][-1].isalnum() and words[i + 1][0].isalnum():
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

    changed_program.append(_delete_meaningless_spaces(python_program[prev:]))
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

    def get(self, old_name: str) -> str:
        if old_name in self.names:
            return self.names[old_name]
        if self.parent is None:
            raise KeyError(f'Variable named "{old_name}" was not found')
        return self.parent.get(old_name)


def replace_names(python_program: str) -> str:
    changed_program = []
    parser = Parser()
    parser.set_language(PY_LANGUAGE)
    tree = parser.parse(bytes(python_program, "utf-8"))
    return "".join(changed_program)


python_program = sys.stdin.read()
without_comments = delete_comments(python_program)
replaced_tabs = replace_tabs_with_spaces(without_comments)
without_meaningless_spaces = delete_meaningless_spaces(replaced_tabs)
print(without_meaningless_spaces)

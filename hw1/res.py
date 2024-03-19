import os
import sys
from dataclasses import dataclass
from typing import Generator,Self,Callable

from tree_sitter import Language,Node,Parser,Tree

Language.build_library(
    "build/my-languages.so",
    [os.path.join(os.getcwd(),"tree-sitter-python")],
)

name_0=Language("build/my-languages.so", "python")


def name_1(name_2:Tree)->Generator[Node,None,None]:
    name_80=name_2.walk()

    name_81=False
    while True:
        if not name_81:
            yield name_80.node
            if not name_80.goto_first_child():
                name_81=True
        elif name_80.goto_next_sibling():
            name_81=False
        elif not name_80.goto_parent():
            break


def name_3(name_4:bytes)->bytes:
    name_82=[]
    name_83=Parser()
    name_83.set_language(name_0)
    name_84=name_83.parse(name_4)
    name_85=0
    for name_86 in name_1(name_84):
        if name_86.type=="comment":
            name_87=name_4.rfind(b"\n",name_85,name_86.start_byte)
            if name_87==-1:
                name_87=name_85

            name_88=name_4[name_87+1:name_86.start_byte]
            if name_88.strip()==b"":
                name_82.append(name_4[name_85:name_87])
                name_85=name_87+1+len(name_88)

            name_82.append(name_4[name_85:name_86.start_byte])
            name_85=name_86.end_byte

    name_82.append(name_4[name_85:])
    return b"".join(name_82)


def name_5(name_6:bytes)->bytes:
    name_89=[]
    name_90=Parser()
    name_90.set_language(name_0)
    name_91=name_90.parse(name_6)
    name_92=0
    for name_93 in name_1(name_91):
        if name_93.type=="string":
            name_94=bytes(
                name_6[name_92:name_93.start_byte].decode(
                    "utf-8").expandtabs(4),"utf-8"
            )
            name_89.append(name_94)
            name_89.append(name_6[name_93.start_byte:name_93.end_byte])
            name_92=name_93.end_byte
    name_95=bytes(name_6[name_92:].decode("utf-8").expandtabs(4),"utf-8")
    name_89.append(name_95)
    return b"".join(name_89)


def name_7(name_8:str)->str:
    name_96=[]
    name_97=[word for word in name_8.split()if word]
    for name_98 in range(len(name_97)):
        name_96.append(name_97[name_98])
        if(
            name_98+1<len(name_97)
            and(name_97[name_98][-1].isalnum()or name_97[name_98][-1]=="_")
            and(name_97[name_98+1][0].isalnum()or name_97[name_98+1][0]=="_")
        ):
            name_96.append(" ")
    return"".join(name_96)


def name_9(name_10:str)->str:
    name_99=0
    for name_100 in name_10:
        if name_100!=" ":
            break
        name_99+=1
    return name_10[:name_99]


def name_11(name_12:str)->str:
    name_101=[]
    for name_102 in name_12.split(os.linesep):
        name_103=name_7(name_102)
        if name_103:
            name_101.append(name_9(name_102)+name_103)
        else:
            name_101.append(name_103)
    return"\n".join(name_101)


def name_13(name_14:bytes)->bytes:
    name_104=[]
    name_105=Parser()
    name_105.set_language(name_0)
    name_106=name_105.parse(name_14)

    name_107=0
    name_108=False
    for name_109 in name_1(name_106):
        if name_109.type=="string":
            name_108=True
            name_110=name_11(
                name_14[name_107:name_109.start_byte].decode("utf-8").lstrip(" ")
            )
            name_104.append(bytes(name_110,"utf-8"))

            name_111=name_14.rfind(b"\n",name_107,name_109.start_byte)
            if name_111==-1:
                name_111=name_107
            name_112=name_14[name_111+1:name_109.start_byte]
            if name_112.strip()==b"":
                name_104.append(name_112)

            name_113=name_14[name_109.start_byte:name_109.end_byte]

            name_114=name_104[-1].decode("utf-8")[-1]
            name_115=name_113.decode("utf-8")[0]
            if(name_114.isalnum()or name_114=="_")and(
                name_115.isalnum()or name_115=="_"
            ):
                name_104.append(b" ")

            name_104.append(name_113)

            name_107=name_109.end_byte

    name_116=name_14[name_107:].decode("utf-8")
    if name_108:
        name_116=name_116.lstrip()

    name_104.append(
        bytes(name_11(name_116),"utf-8"))
    return b"".join(name_104)


class name_15:
    def __init__(name_16,name_17:Self or None):
        name_16.parent=name_17
        name_16.names=dict()
        name_16.namespaces=dict()

    def insert(name_18,name_19:str,name_20:str)->None:
        name_18.names[name_19]=name_20

    def get(name_21,name_22:str)->str or None:
        if name_22 in name_21.names:
            return name_21.names[name_22]
        if name_21.parent is None:
            return None
        return name_21.parent.get(name_22)

    def insert_namespace(name_23,name_24:str,name_25:dict[str,str]):
        name_23.namespaces[name_24]=name_25

    def get_namespace(name_26,name_27:str)->Self or None:
        if name_27 in name_26.namespaces:
            return name_26.namespaces[name_27]
        if name_26.parent is None:
            return None
        return name_26.parent.get_namespace(name_27)


@dataclass
class name_28:
    start_byte:int
    end_byte:int
    replacement:bytes


def name_29(*name_30,**name_31):
    return


class name_32:
    def __init__(name_33):
        name_33.global_scope=name_15(None)
        name_33.changes=[]
        name_33.last_id=0

    def make_changes(name_34,name_35:Tree)->list[name_28]:
        name_117=name_35.root_node
        name_34._visit(name_117,name_34.global_scope)
        return sorted(name_34.changes,key=lambda change:change.start_byte)

    def _create_new_name(name_36)->str:
        name_118=f"name_{name_36.last_id}"
        name_36.last_id+=1
        return name_118

    def _visit(
        name_37,
        name_38:Node or None,
        name_39:name_15,
        name_40:Callable[[Node,name_15],None]=None,
        name_41:Callable[[Node,name_15],None]=None,
        name_42:Callable[[Node,name_15],None]=None,
        name_43:Callable[[Node,name_15],None]=None,
        name_44:Callable[[Node,name_15],None]=None,
    )->Callable[[],None]or None:
        if name_40 is None:
            name_40=name_37._rename_identifier
        if name_41 is None:
            name_41=name_37._visit_function_definition
        if name_43 is None:
            name_43=name_37._visit_assignment
        if name_44 is None:
            name_44=name_37._visit_type

        if name_38 is None:
            return None
        elif name_38.type=="class_definition":
            return name_37._visit_class_definition(name_38,name_39)
        elif name_38.type=="function_definition":
            return name_41(name_38,name_39)
        elif name_38.type=="for_statement":
            name_37._visit_for_statement(name_38,name_39)
        elif name_38.type=="assignment":
            name_43(name_38,name_39)
        elif name_38.type=="attribute":
            name_37._visit_attribute(name_38,name_39,name_40)
        elif name_38.type=="call":
            name_37._visit_call(name_38,name_39)
        elif name_38.type=="keyword_argument"and name_42:
            name_42(name_38,name_39)
        elif name_38.type=="named_expression":
            name_37._visit_named_expression(name_38,name_39)
        elif name_38.type=="type":
            name_44(name_38,name_39)
        elif name_38.type=="identifier":
            name_40(name_38,name_39)
        else:
            name_119=[]
            for name_120 in name_38.children:
                name_119.append(
                    name_37._visit(
                        name_120,
                        name_39,
                        name_40=name_40,
                        name_41=name_41,
                        name_42=name_42,
                        name_44=name_44,
                        name_43=name_43,
                    )
                )
            for name_121 in name_119:
                if name_121:
                    name_121()
        return None

    def _visit_function_definition(
        name_45,name_46:Node,name_47:name_15
    )->Callable[[],None]:
        name_122=name_46.child_by_field_name("name")
        name_45._define_and_rename_identifier(name_122,name_47)
        name_123=name_15(None)
        name_124=name_15(name_47)

        def name_125(name_126:Node,name_127:name_15):
            name_45._visit_type(name_126,name_47,name_70=name_125)

        name_128=name_46.child_by_field_name("parameters")
        name_45._visit(
            name_128,
            name_123,
            name_40=name_45._define_and_rename_identifier,
            name_44=name_125
        )
        name_124.names.update(name_123.names)
        name_47.insert_namespace(str(name_122.text),name_123)
        name_45._visit(name_46.child_by_field_name("return_type"),name_47)
        return lambda:(name_45._visit(name_46.child_by_field_name("body"),name_124))

    def _visit_call(name_48,name_49:Node,name_50:name_15):
        name_129=name_49.child_by_field_name("function")
        if name_129.type=="attribute":
            name_130=name_48._visit_attribute(name_129,name_50)
        elif name_129.type=="identifier":
            name_48._visit(name_129,name_50)
            name_130=name_50.get_namespace(str(name_129.text))
        else:
            raise RuntimeError(f"Wrong function type: `{name_129.text}`")

        def name_131(name_132:Node,name_133:name_15):
            if name_130:
                name_135=name_132.child_by_field_name("name")
                name_48._visit(name_135,name_130)
            name_136=name_132.child_by_field_name("value")
            name_48._visit(name_136,name_133)

        name_134=name_49.child_by_field_name("arguments")
        name_48._visit(name_134,name_50,
                    name_42=name_131)

    def _visit_class_definition(name_51,name_52:Node,name_53:name_15)->Callable[[],None]:
        name_137=name_52.child_by_field_name("name")
        name_51._define_and_rename_identifier(name_137,name_53)

        name_51._visit(name_52.child_by_field_name("superclasses"),name_53)

        name_138=name_15(name_53)
        name_139=name_15(None)
        name_53.insert_namespace(str(name_137.text),name_139)

        name_140=[
            method
            for method in name_52.child_by_field_name("body").children
            if method.type=="function_definition"
        ]
        name_141=[]
        for name_142 in name_140:
            if name_142.type!="function_definition":
                continue

            def name_143(name_144:Node)->str:
                if name_144.type=="typed_parameter":
                    return str(name_144.child(0).text)
                return str(name_144.text)

            name_137=name_142.child_by_field_name("name")
            name_145=name_15(None)
            name_146=name_15(name_138)
            name_141.append(name_146)

            name_147=name_142.child_by_field_name("parameters")
            name_148=name_143(name_147.child(1))
            name_53.insert_namespace(name_148,name_139)

            def name_149(name_150:Node,name_151:name_53):
                name_51._visit_type(name_150,name_138,name_70=name_149)

            name_51._visit(
                name_147,
                name_145,
                name_40=name_51._define_and_rename_identifier,
                name_44=name_149,
            )
            name_146.names.update(name_145.names)
            name_53.insert_namespace(str(name_137.text),name_145)
            name_51._visit(name_142.child_by_field_name("return_type"),name_53)

        def name_152(name_153:Node,name_154:name_15):
            name_156=name_153.child_by_field_name("left")
            if name_156.type!="identifier"and name_156.type!="pattern_list":
                name_51._visit_assignment(name_153,name_154)
                return
            name_51._visit(
                name_153.child_by_field_name("left"),
                name_154,
                name_40=name_29,
            )
            name_51._visit(name_153.child_by_field_name("type"),name_154)
            name_51._visit(name_153.child_by_field_name("right"),name_154)

        def name_155():
            name_51._visit(
                name_52.child_by_field_name("body"),
                name_138,
                name_41=name_29,
                name_43=name_152,
            )
            for name_142,name_157 in zip(name_140,name_141):
                name_51._visit(name_142.child_by_field_name("body"),name_157)

        return name_155

    def _visit_assignment(name_54,name_55:Node,name_56:name_15):
        name_54._visit(name_55.child_by_field_name("left"),
                    name_56,name_40=name_54._define_and_rename_identifier)
        name_54._visit(name_55.child_by_field_name("type"),name_56)
        name_54._visit(name_55.child_by_field_name("right"),name_56)

    def _visit_attribute(
        name_57,
        name_58:Node,
        name_59:name_15,
        name_60:Callable[[Node,name_15],None]=None,
    )->name_15 or None:
        if name_60 is None:
            name_60=name_57._rename_identifier
        if name_58.type=="identifier":
            if name_59 is None:
                return None
            name_60(name_58,name_59)
            return name_59.get_namespace(str(name_58.text))
        elif name_58.type=="attribute":
            name_158=name_57._visit_attribute(
                name_58.child_by_field_name("object"),name_59,name_60
            )
            if name_158:
                name_159=name_58.child_by_field_name("attribute")
                name_57._rename_identifier(name_159,name_158)
                return name_59.get_namespace(str(name_159.text))
        else:
            name_57._visit(name_58,name_59)
        return None

    def _visit_for_statement(name_61,name_62:Node,name_63:name_15):
        name_61._visit(
            name_62.child_by_field_name("left"),
            name_63,
            name_40=name_61._define_and_rename_identifier,
        )
        name_61._visit(name_62.child_by_field_name("right"),name_63)
        name_160=name_15(name_63)
        name_61._visit(name_62.child_by_field_name("body"),name_160)

    def _visit_named_expression(name_64,name_65:Node,name_66:name_15):
        name_64._visit(
            name_65.child(0),
            name_66,
            name_40=name_64._define_and_rename_identifier,
        )
        name_64._visit(name_65.child_by_field_name("value"),name_66)

    def _visit_type(
        name_67,name_68:Node,name_69:name_15,name_70:Callable[[Node,name_15],None]=None
    ):
        for name_161 in name_68.children:
            name_67._visit(name_161,name_69,name_44=name_70)

    def _rename_identifier(name_71,name_72:Node,name_73:name_15):
        if name_162:=name_73.get(name_72.text):
            name_71.changes.append(
                name_28(name_72.start_byte,name_72.end_byte,name_162))

    def _define_and_rename_identifier(name_74,name_75:Node,name_76:name_15):
        name_163=name_76.get(name_75.text)or name_74._create_new_name()
        name_76.insert(name_75.text,name_163)
        name_74.changes.append(name_28(name_75.start_byte,name_75.end_byte,name_163))


def name_77(name_78:bytes)->bytes:
    name_164=[]
    name_165=Parser()
    name_165.set_language(name_0)
    name_166=name_165.parse(name_78)

    name_167=name_32().make_changes(name_166)
    name_168=0
    for name_169 in name_167:
        name_164.append(name_78[name_168:name_169.start_byte])
        name_164.append(bytes(name_169.replacement,"utf-8"))
        name_168=name_169.end_byte
    name_164.append(name_78[name_168:])

    return b"".join(name_164)


name_79=bytes(sys.stdin.read(),"utf-8")
name_79=name_3(name_79)
name_79=name_5(name_79)
name_79=name_13(name_79)
name_79=name_77(name_79)
print(name_79.decode("utf-8"))


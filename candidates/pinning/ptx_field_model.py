#!/usr/bin/env python3
"""Execute the straight-line integer PTX subset used by the field primitives.

This is a CPU semantic model, not a CUDA assembler, emulator or GPU test.
Unknown opcodes, uninitialized registers and malformed statements fail closed.
Semantics: https://docs.nvidia.com/cuda/parallel-thread-execution/index.html
"""
import ast
import re


def function(source, signature):
    start = source.index(signature)
    brace = source.index('{', start)
    depth = 0
    for end in range(brace, len(source)):
        depth += (source[end] == '{') - (source[end] == '}')
        if depth == 0:
            return source[start:end+1]
    raise ValueError('Unclosed function')


def extract_ptx(body):
    part = body[body.index('asm(')+4:body.index(': "=l"')]
    return ''.join(ast.literal_eval(s) for s in re.findall(r'"(?:[^"\\]|\\.)*"', part))


class Program:
    def __init__(self, text):
        self.widths = {}
        self.ops = []
        text = text.strip()
        assert text[0] == '{' and text[-1] == '}'
        for statement in text[1:-1].split(';'):
            statement = statement.strip()
            if not statement:
                continue
            if statement.startswith('.reg'):
                _, kind, names = statement.split(None, 2)
                bits = int(kind[2:])
                assert kind in ('.u32', '.u64')
                for name in names.split(','):
                    name = name.strip()
                    assert name not in self.widths
                    self.widths[name] = bits
                continue
            opcode, rest = statement.split(None, 1)
            args = tuple(s.strip() for s in re.findall(r'\{[^}]+\}|[^,\s]+', rest))
            assert opcode in {'mov.b64', 'mov.u32', 'mul.wide.u32', 'mul.lo.u32',
                'add.cc.u32', 'add.cc.u64', 'addc.cc.u32', 'addc.cc.u64',
                'addc.u32', 'addc.u64', 'cvt.u64.u32', 'mad.lo.u32', 'shf.l.wrap.b32'}, opcode
            self.ops.append((opcode, args))

    def run(self, inputs, input_base=4, outputs=4):
        regs = {'%'+str(i+input_base): value for i, value in enumerate(inputs)}
        carry = 0

        def read(name):
            if name.startswith('{'):
                a, b = (v.strip() for v in name[1:-1].split(','))
                return read(a) | (read(b) << 32)
            if re.fullmatch(r'(0x[0-9a-fA-F]+|[0-9]+)', name):
                return int(name, 0)
            return regs[name]

        def write(name, value):
            if name.startswith('{'):
                a, b = (v.strip() for v in name[1:-1].split(','))
                write(a, value & 0xffffffff)
                write(b, value >> 32)
            else:
                bits = 64 if name.startswith('%') else self.widths[name]
                regs[name] = value & ((1 << bits)-1)

        for op, args in self.ops:
            dest, *sources = args
            values = [read(s) for s in sources]
            if op.startswith(('mov.', 'cvt.')):
                assert len(values) == 1
                value = values[0]
            elif op.startswith('mul.'):
                assert len(values) == 2
                value = values[0]*values[1]
            elif op == 'mad.lo.u32':
                assert len(values) == 3
                value = values[0]*values[1]+values[2]
            elif op.startswith('add'):
                assert len(values) == 2
                bits = int(op.rsplit('u', 1)[1])
                value = values[0]+values[1]+(carry if op.startswith('addc.') else 0)
                if '.cc.' in op:
                    carry = value >> bits
                    assert carry in (0, 1)
            elif op == 'shf.l.wrap.b32':
                assert len(values) == 3
                a, b, shift = values
                value = (((b << 32) | a) << (shift & 31)) >> 32
            else:
                raise AssertionError(op)
            write(dest, value)
        return [regs['%'+str(i)] for i in range(outputs)]


def check_semantics():
    # Carry-in survives instructions without .cc, including addc itself.
    p = Program('''{.reg .u32 a,b,c,d;
      add.cc.u32 a, 4294967295, 1;
      addc.u32 b, 0, 0;
      addc.u32 c, 0, 0;
      addc.cc.u32 d, 0, 0;
      mov.b64 %0, {b,c}; mov.b64 %1, {d,a};
      mov.b64 %2, 0; mov.b64 %3, 0;}''')
    assert p.run([]) == [1+(1 << 32), 1, 0, 0]
    p = Program('''{.reg .u32 a,b;
      shf.l.wrap.b32 a, 2147483648, 1, 1;
      shf.l.wrap.b32 b, 7, 11, 32;
      mov.b64 %0, {a,b}; mov.b64 %1, 0;
      mov.b64 %2, 0; mov.b64 %3, 0;}''')
    assert p.run([]) == [3+(11 << 32), 0, 0, 0]


if __name__ == '__main__':
    check_semantics()
    print('PASS: model carry preservation, carry overwrite, packing and funnel shift')

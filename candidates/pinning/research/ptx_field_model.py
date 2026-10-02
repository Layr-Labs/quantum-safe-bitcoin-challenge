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
            assert opcode in {'mov.b64', 'mov.u32', 'mov.u64', 'mul.wide.u32', 'mul.lo.u32',
                'add.cc.u32', 'add.cc.u64', 'addc.cc.u32', 'addc.cc.u64',
                'addc.u32', 'addc.u64', 'cvt.u64.u32', 'mad.lo.u32', 'shf.l.wrap.b32',
                'sub.cc.u32', 'subc.cc.u32', 'subc.u32', 'xor.b32', 'and.b32', 'not.b32',
                'mad.wide.u32', 'cvt.u32.u64', 'shr.u64', 'shr.u32', 'shl.b32',
                'or.b32', 'shf.r.wrap.b32', 'mad.lo.cc.u32', 'madc.lo.cc.u32',
                'madc.hi.cc.u32', 'madc.hi.u32'}, opcode
            self.ops.append((opcode, args))

    def run(self, inputs, output_count=4):
        regs = {'%'+str(i+4): value for i, value in enumerate(inputs)}
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
            elif op.startswith(('mad.', 'madc.')):
                assert len(values) == 3
                product = values[0]*values[1]
                if '.hi.' in op:
                    product >>= 32
                elif '.lo.' in op:
                    product &= 0xffffffff
                value = product+values[2]+(carry if op.startswith('madc.') else 0)
                if '.cc.' in op:
                    carry = value >> 32
                    assert carry in (0, 1)
            elif op.startswith('add'):
                assert len(values) == 2
                bits = int(op.rsplit('u', 1)[1])
                value = values[0]+values[1]+(carry if op.startswith('addc.') else 0)
                if '.cc.' in op:
                    carry = value >> bits
                    assert carry in (0, 1)
            elif op.startswith('sub'):
                assert len(values) == 2
                value = values[0]-values[1]-(carry if op.startswith('subc.') else 0)
                if '.cc.' in op:
                    carry = int(value < 0)
            elif op == 'xor.b32':
                value = values[0] ^ values[1]
            elif op == 'and.b32':
                value = values[0] & values[1]
            elif op == 'not.b32':
                value = ~values[0]
            elif op == 'or.b32':
                value = values[0] | values[1]
            elif op.startswith(('shr.', 'shl.')):
                bits = int(op.rsplit('.', 1)[1][1:])
                shift = min(values[1], bits)
                value = values[0] >> shift if op.startswith('shr.') else values[0] << shift
            elif op == 'shf.l.wrap.b32':
                assert len(values) == 3
                a, b, shift = values
                value = (((b << 32) | a) << (shift & 31)) >> 32
            elif op == 'shf.r.wrap.b32':
                assert len(values) == 3
                a, b, shift = values
                value = ((b << 32) | a) >> (shift & 31)
            else:
                raise AssertionError(op)
            write(dest, value)
        return [regs['%'+str(i)] for i in range(output_count)]


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
    # subc consumes borrow and preserves it unless .cc writes a new borrow.
    p = Program('''{.reg .u32 a,b,c,d,e,f;
      sub.cc.u32 a, 0, 1; subc.u32 b, 0, 0; subc.u32 c, 1, 0;
      subc.cc.u32 d, 2, 0; subc.u32 e, 0, 0;
      xor.b32 f, a, 4294967295;
      mov.b64 %0, {a,b}; mov.b64 %1, {c,d};
      mov.b64 %2, {e,f}; mov.b64 %3, 0;}''')
    assert p.run([]) == [(1 << 64)-1, 1 << 32, 0, 0]
    # Wide MAD wraps its 64-bit output without changing CC. Shift amounts on
    # shr/shl clamp to width, while the .wrap funnel uses the low five bits.
    p = Program('''{.reg .u32 a,b,c,d,e; .reg .u64 t;
      add.cc.u32 a, 4294967295, 1;
      mad.wide.u32 t, 4294967295, 4294967295, 18446744073709551615;
      addc.u32 b, 0, 0;
      cvt.u32.u64 c, t; shr.u64 t, t, 32; cvt.u32.u64 d, t;
      shf.r.wrap.b32 e, c, d, 32;
      mov.b64 %0, {c,d}; mov.b64 %1, {e,b};
      shr.u32 c, 4294967295, 32; shl.b32 d, 1, 32;
      mov.b64 %2, {c,d}; mov.b64 %3, 0;}''')
    assert p.run([]) == [(1 << 64) - (1 << 33), 1 << 32, 0, 0]
    # High-part MAD extracts high(product) BEFORE adding c/carry, per PTX.
    # This is explicitly not (product+c+carry)>>32.
    p = Program('''{.reg .u32 lo,hi,cf,dummy;
      add.cc.u32 dummy, 4294967295, 1;
      madc.lo.cc.u32 lo, 4294967295, 2, 3;
      madc.hi.cc.u32 hi, 4294967295, 2, 7;
      addc.u32 cf, 0, 0;
      mov.b64 %0, {lo,hi}; mov.b64 %1, {cf,dummy};
      mov.b64 %2, 0; mov.b64 %3, 0;}''')
    assert p.run([]) == [(9 << 32)+2, 0, 0, 0]


if __name__ == '__main__':
    check_semantics()
    print('PASS: model carry preservation, carry overwrite, packing and funnel shift')

import sys
src = open(sys.argv[1], encoding='utf-8').read()
start = src.index(sys.argv[2])
brace = src.index('{', start)

# naive
depth = 1; i = brace + 1
while depth and i < len(src):
    depth += (src[i] == '{') - (src[i] == '}')
    i += 1
print('naive end line:', src[:i].count('\n') + 1)

# string/comment aware
depth = 1; i = brace + 1
instr = esc = incmt = inlcm = False
while depth and i < len(src):
    c = src[i]; n = src[i+1] if i + 1 < len(src) else ''
    if inlcm:
        if c == '\n': inlcm = False
    elif incmt:
        if c == '*' and n == '/': incmt = False; i += 1
    elif instr:
        if esc: esc = False
        elif c == '\\': esc = True
        elif c == '"': instr = False
    else:
        if c == '/' and n == '/': inlcm = True; i += 1
        elif c == '/' and n == '*': incmt = True; i += 1
        elif c == '"': instr = True
        else: depth += (c == '{') - (c == '}')
    i += 1
print('real end line:', src[:i].count('\n') + 1)

#!/usr/bin/env python3
"""Rename an ONNX model input tensor name (and replace occurrences).

Usage: rename_onnx_input.py <input.onnx> <old_name> <new_name> <output.onnx>
If <old_name> is '-' the script will use the first graph input name.
"""
import sys
import onnx

def replace_all(model, old, new):
    # graph inputs
    for vi in model.graph.input:
        if vi.name == old:
            vi.name = new
    # graph outputs
    for vi in model.graph.output:
        if vi.name == old:
            vi.name = new
    # initializers
    for init in model.graph.initializer:
        if init.name == old:
            init.name = new
    # value_info
    for vi in model.graph.value_info:
        if vi.name == old:
            vi.name = new
    # nodes
    for node in model.graph.node:
        for i, name in enumerate(node.input):
            if name == old:
                node.input[i] = new
        for i, name in enumerate(node.output):
            if name == old:
                node.output[i] = new

def main():
    if len(sys.argv) != 5:
        print(__doc__)
        sys.exit(2)
    inp, old, new, out = sys.argv[1:5]
    model = onnx.load(inp)
    if old == '-':
        if len(model.graph.input) == 0:
            print('Model has no inputs')
            sys.exit(1)
        old = model.graph.input[0].name
        print('Detected first input name:', old)
    print('Replacing', old, '->', new)
    replace_all(model, old, new)
    onnx.save(model, out)
    print('Saved', out)
    print('Graph inputs after change:', [i.name for i in model.graph.input])
    print('Graph outputs after change:', [o.name for o in model.graph.output])

if __name__ == '__main__':
    main()

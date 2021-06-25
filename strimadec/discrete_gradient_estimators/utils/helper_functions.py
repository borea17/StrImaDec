def set_requires_grad(module, val):
    for p in module.parameters():
        p.requires_grad = val
    return
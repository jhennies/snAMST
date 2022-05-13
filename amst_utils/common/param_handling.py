
REPLACE_CHARS = [
    ['=', '\{eq}']
]


def replace_special(inp, back=False):

    for rpl in REPLACE_CHARS:
        if not back:
            inp = inp.replace(rpl[0], rpl[1])
        else:
            inp = inp.replace(rpl[1], rpl[0])
    return inp

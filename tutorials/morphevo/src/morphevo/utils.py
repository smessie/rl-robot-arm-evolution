
def alternate(what, times):
    alternations = []
    for alternation, _ in zip(alternate_infinite(what), range(times)):
        alternations.append(alternation)

    return alternations

def alternate_infinite(what):
    current_index = 0
    while True:
        yield what[current_index]
        current_index = (current_index + 1) % len(what)

def normalize(raw):
    sum_raw = sum(raw)
    return [ i/sum_raw for i in raw ]

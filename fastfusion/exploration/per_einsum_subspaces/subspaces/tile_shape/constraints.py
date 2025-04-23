from operator import le, lt, ge, gt, eq
import re


CONSTRAINT_RE_PARSER = re.compile('^(>=|>|<|<=|==)\s*(\d+)')


STRING_TO_OPERATOR = {
    '==': eq,
    '>=': ge,
    '>':  gt,
    '<=': le,
    '<':  lt
}


def get_propagated_constraint(comparison: str, limit):
    if comparison == '>=':
        return lambda x: x >= limit
    elif comparison == '>':
        return lambda x: x > limit
    elif comparison == '<=':
        return None
    elif comparison == '<':
        return None
    elif comparison == '==':
        return lambda x: x >= limit


def parse_tile_constraint(constraint_str: str):
    match = CONSTRAINT_RE_PARSER.match(constraint_str)
    if match is None:
        raise RuntimeError(f'Cannot parse constraint {constraint_str}')

    comparison, limit = match.groups()

    limit = int(limit)
    if limit == 128:
        comparison = "=="

    if comparison in STRING_TO_OPERATOR:
        operator = STRING_TO_OPERATOR[comparison]
        main_constraint = lambda x: operator(x, limit)
    else:
        raise RuntimeError(f'Unknown comparison operator {comparison}')

    propagated_constraint = get_propagated_constraint(comparison, limit)

    return main_constraint, propagated_constraint


def parse_factor_constraint(constraint_str: str):
    match = CONSTRAINT_RE_PARSER.match(constraint_str)
    if match is None:
        raise RuntimeError(f'Cannot parse constraint {constraint_str}')

    comparison, limit = match.groups()

    limit = int(limit)
    if limit == 128:
        comparison = "=="

    if comparison in STRING_TO_OPERATOR:
        operator = STRING_TO_OPERATOR[comparison]
        main_constraint = lambda x: operator(x, limit)
    else:
        raise RuntimeError(f'Unknown comparison operator {comparison}')

    return main_constraint, None

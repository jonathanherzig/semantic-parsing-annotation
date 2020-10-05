import argparse
from collections import defaultdict
from itertools import permutations
import os
import re
import subprocess
import tempfile

import pyparsing


class Grammar(object):

    def __init__(self):
        self._rules = defaultdict(list)
        self._sums = defaultdict(float)
        self.parser = pyparsing.nestedExpr('(', ')', ignoreExpr=pyparsing.dblQuotedString)
        self.rules_counter = 0
        self.parses_counter = 0

    def add_rule(self, lhs, rhs, semantic_type, weight, type, rhs_typed_ind):
        """
        Adds a rule to internal grammar representation.
        """
        assert(isinstance(lhs, str))
        assert(isinstance(rhs, list))
        assert (isinstance(semantic_type, list))
        assert (isinstance(type, list))
        assert (isinstance(rhs_typed_ind, int))
        self._rules[lhs].append((rhs, semantic_type, weight, type, rhs_typed_ind))
        self._sums[lhs] += weight

    def parse_rule(self, rule):
        """
        Parses a grammar rule from file.
        """
        assert rule[0] == 'rule'
        weight = 1
        lhs = rule[1]
        rhs = rule[2].asList()
        semantic_type = rule[3].asList()
        assert(len(rule)>4)
        if len(rule) > 4:
            types = rule[4].asList()  # determine the input and output types of the rule.
        # if rhs is binary, this is the index for the product that should output the desired type.
        if len(rule) > 5:
            rhs_typed_ind = int(rule[5].asList()[0])
        else:
            rhs_typed_ind = -1
        self.add_rule(lhs, rhs, semantic_type, weight, types, rhs_typed_ind)

    def from_file(self, filename):
        """
        Reads grammar from file.
        :param filename: The input grammar file.
        """
        text = ''
        re_comment = re.compile(r'^ *#')
        events = []

        with open(filename, 'r') as f:
            for row in f:
                row_uncommented = row
                if re_comment.match(row): continue  # ignore incase the row is a comment.
                if '#' in row:   # incase there is a comment before row ends.
                    row_uncommented = row_uncommented.split('#')[0].rstrip()

                text += row_uncommented
                try:
                    code_parse = self.parser.parseString(text)
                    events.append(code_parse)
                    text = ''
                except:
                    continue

        for event in events:
            type = event[0][0]
            if type == 'def': continue
            elif type == 'rule':
                self.parse_rule(event[0])
            elif type == 'when':
                when_args = event[0][1]
                parameters = set(['generate'])
                if isinstance(when_args, str):
                    when_args = [when_args]

                is_pass = True
                if 'geo880' in set(when_args):
                    print(when_args)
                if set(when_args) == parameters or set(when_args) == set(['generate', 'and', 'general']):
                    is_pass = False
                if is_pass: continue

                for i, rule in enumerate(event[0]):
                    if i < 2:
                        continue
                    self.parse_rule(rule)

    def is_terminal(self, symbol): return symbol not in self._rules

    def gen_exhaustive(self, symbol, all_derives_path, domain, max_depth=6):
        """
        Generates all derivations exhaustively up to max_depth.
        :param symbol: The symbol to initialize the generation from.
        :param max_depth: Max depth of rules.
        :return:
        """
        all_derivs = []
        open(all_derives_path, 'a').close()
        rules = []
        semantic_types = []
        counter = [0]
        if domain == 'geo880':
            all_types = ['_number_', '_city_', '_state_', '_river_', '_lake_', '_mountain_', '_place_', '_country_']
        else:
            all_types = ['_number_', '_year_', '_author_', '_paper_', '_keyphrase_', '_dataset_', '_venue_', '_journal_', '_title_']
        # Generate for each type
        for type in all_types:
            print(type)
            self.gen_exhaustive_recurse_typed([(symbol, -1, 0, type, True)], rules, semantic_types,
                                              max_depth, all_derivs, all_derives_path, counter,
                                              all_types, domain)
        print(len(all_derivs))
        print(len(semantic_types))
        return all_derivs

    def gen_exhaustive_recurse_typed(self, part_sent, rules, semantic_types, max_depth, all_derivs,
                                     all_derives_path, counter, types, domain):
        sent = ' '.join([w[0] for w in part_sent])  # current partial generated sentence.
        if sent.count('and ') > 2:  # No more than 2 conjunctions
            return
        if sent.count('whose ') + sent.count('that ') + sent.count('than ') > 3:
            return
        if self.violates_number_binding(sent, domain):
            return
        all_terminal = all(self.is_terminal(symbol[0]) for symbol in part_sent)
        if any((self.is_terminal(symbol[0]) and symbol[0].startswith('$')) for symbol in part_sent):
            return
        # stop if the derived sentence is only terminals
        elif all_terminal:
            all_derivs.append((part_sent, rules, semantic_types))
            print(sent)
            print(counter[0])
            counter[0] += 1
            if len(all_derivs) % 1000000 == 0:
                write_all_derivs(all_derivs, all_derives_path, domain)
                print('cleared {}'.format(len(all_derivs)))
                del all_derivs[:]
            if counter[0] % 10000 == 0:
                print('generated {}'.format(counter[0]))
            return
        # stop if reached depth larger than max_depth
        elif any((not self.is_terminal(symbol[0]) and symbol[2] >= max_depth
                  ) for symbol in part_sent) or all_terminal:
            return
        else:
            # find first non-terminal. index is for the rule that derived the non-terminal.
            i, symbol, index, depth, type, is_right_child = self.get_first_non_terminal_typed(part_sent, max_depth)
            # RHS where the relation should be reversesd.
            reverse_expansions = [['that', '$NP', '$VP/NP'], ['that', '$NP', 'not', '$VP/NP'],
                                  ['whose', '$RelNP', 'is', '$NP'],
                                  ['whose', '$RelNP', 'is', 'not', '$NP'],
                                  ['that', 'the', 'most', 'number', 'of', '$NP', '$VP/NP'],
                                  ['that', 'the', 'least', 'number', 'of', '$NP', '$VP/NP']]
            for expansion in self._rules[symbol]:  # Expand all RHS for the LHS
                output_type = expansion[3][1]
                child_type_output = expansion[4]
                if not (output_type == '_any_' or output_type == type):
                    continue
                rhs = expansion[0]
                is_reverse = rhs in reverse_expansions
                semantic_type = expansion[1]

                # This is a special case, since any set is countable
                if rhs == ['number', 'of', '$NP']:
                    for t in types:
                        if t != '_number_':
                            deriv_sent, deriv_rules, deriv_semantic_types, current_ind = self.add_rule_prepare(
                                symbol, rhs, semantic_type, i, index, depth, t, is_right_child,
                                rules, semantic_types, part_sent)
                            self.gen_exhaustive_recurse_typed(deriv_sent, deriv_rules,
                                                              deriv_semantic_types, max_depth,
                                                              all_derivs, all_derives_path,
                                                              counter, types, domain)
                    continue
                # in this special case $RelNP should be a relation between type and _number_
                if rhs == ['that', 'has', 'the', 'largest', '$RelNP'] or rhs == [
                    'that', 'has', 'the', 'smallest', '$RelNP']:
                    type_next = '_number_'
                else:
                    type_next = type
                deriv_sent, deriv_rules, deriv_semantic_types, current_ind = self.add_rule_prepare(
                    symbol, rhs, semantic_type, i, index, depth, type_next, is_right_child, rules,
                    semantic_types, part_sent)

                # in this case the rhs is binary, so one product depends on the type of the
                # other product.
                if child_type_output != -1:
                    # whether the rule to expand is right_child
                    is_right_child_inner = child_type_output == 1
                    first_encountered = True  # whether the encountered non terminal is the first
                    for k, symbol_ in enumerate(rhs):
                        if not self.is_terminal(symbol_):
                            if first_encountered:  # is this the left non-terminal
                                if is_right_child_inner:
                                    rel_ind_not_expended = k
                                else:
                                    rel_ind_expended = k
                            else:
                                if is_right_child_inner:
                                    rel_ind_expended = k
                                else:
                                    rel_ind_not_expended = k
                            first_encountered = False
                    expended_ind = i+rel_ind_expended
                    symbol_to_iterate = deriv_sent[expended_ind][0]
                    for expansion_inner in self._rules[symbol_to_iterate]:
                        output_type_inner = expansion_inner[3][1]
                        input_type_inner = expansion_inner[3][0]
                        if is_reverse:  # switch if should be reversed
                            temp = output_type_inner
                            output_type_inner = input_type_inner
                            input_type_inner = temp
                        # special logic for this rule
                        if rhs == ['whose', '$RelNP', 'is', 'larger', 'than', '$NP'] or rhs == [
                                   'whose', '$RelNP', 'is', 'smaller', 'than', '$NP']:
                            if not (input_type_inner == type_next and output_type_inner == '_number_'):
                                continue
                            other_prod_type = '_number_'
                        else:
                            if not (output_type_inner == type_next):
                                continue
                            other_prod_type = input_type_inner
                        rhs_inner = expansion_inner[0]
                        semantic_type_inner = expansion_inner[1]

                        tuple_to_switch = deriv_sent[i + rel_ind_not_expended]
                        tuple_to_switch = list(tuple_to_switch)
                        tuple_to_switch[3] = other_prod_type
                        tuple_to_switch = tuple(tuple_to_switch)
                        deriv_sent[i + rel_ind_not_expended] = tuple_to_switch
                        deriv_sent_, deriv_rules_, deriv_semantic_types_, _ = self.add_rule_prepare(
                            symbol_to_iterate, rhs_inner, semantic_type_inner, expended_ind,
                            father_ind=current_ind, depth=depth+1, type=type_next,
                            is_right_child=is_right_child_inner, rules=deriv_rules,
                            semantic_types=deriv_semantic_types, part_sent=deriv_sent)
                        self.gen_exhaustive_recurse_typed(deriv_sent_, deriv_rules_,
                                                          deriv_semantic_types_, max_depth,
                                                          all_derivs, all_derives_path, counter,
                                                          types, domain)
                else:
                    self.gen_exhaustive_recurse_typed(deriv_sent, deriv_rules, deriv_semantic_types,
                                                      max_depth, all_derivs, all_derives_path,
                                                      counter, types, domain)

    def violates_number_binding(self, sent, domain):
        """Check if generated number is nonsensical in terms of number bindings."""
        if domain == 'geo880':
            rels = ['density', 'area', 'length', 'elevation', 'population', 'density']
            ents = ['sacramento', 'california', 'colorado river', 'lake tahoe', 'mount whitney',
                    'death valley', 'usa']
        else:
            rels = ['citation count', 'publication year']
            ents = ['noah smith', 'richard anderson', 'semantic parsing', 'deep learning', 'nature',
                    'acl', 'cell', 'neural attention', 'reviews', 'blogs']

        # e.g., "number of california"
        for e in ents:
            if 'number of {}'.format(e) in sent:
                return True

        is_check = False
        for r in rels:
            if 'whose {}'.format(r) in sent:
                is_check = True
                break
        if not is_check:
            return False
        for i, r_1 in enumerate(rels):
            bad_templates = [
                'whose {} is {}'.format(r_1, 'number'),
                'whose {} is not {}'.format(r_1, 'number'),
                'whose {} is larger than {}'.format(r_1, 'number'),
                'whose {} is smaller than {}'.format(r_1, 'number'),
                'whose {} is {}'.format(r_1, 'total'),
                'whose {} is not {}'.format(r_1, 'total'),
                'whose {} is larger than {}'.format(r_1, 'total'),
                'whose {} is smaller than {}'.format(r_1, 'total'),
                'whose {} is {}'.format(r_1, '0'),
                'whose {} is not {}'.format(r_1, '0'),
                'whose {} is larger than {}'.format(r_1, '0'),
                'whose {} is smaller than {}'.format(r_1, '0'),
            ]
            for template in bad_templates:
                if template in sent:
                    return True
            for j, r_2 in enumerate(rels):
                bad_templates = [
                    'whose {} is {}'.format(r_1, r_2),
                    'whose {} is not {}'.format(r_1, r_2),
                    'whose {} is larger than {}'.format(r_1, r_2),
                    'whose {} is smaller than {}'.format(r_1, r_2),
                ]
                if j!=i:
                    for template in bad_templates:
                        if template in sent:
                            return True
        return False

    def add_rule_prepare(self, symbol, rhs, semantic_type, ind_in_sent, father_ind, depth, type,
                         is_right_child, rules, semantic_types, part_sent):
        current_ind = self.rules_counter  # id for the rule
        self.rules_counter += 1
        deriv_rules = list(rules)
        deriv_rules.append((symbol, tuple(rhs)))
        deriv_semantic_types = list(semantic_types)
        deriv_semantic_types.append((tuple(semantic_type), current_ind, father_ind, is_right_child))
        deriv_sent = part_sent
        next_depth = depth if semantic_type == ['IdentityFn'] else depth+1

        number_non_terminals = sum([not self.is_terminal(symbol) for symbol in rhs])
        replace = []
        first_encountered = True
        for k, symbol_ in enumerate(rhs):
            is_right_child_next = True
            if not self.is_terminal(symbol_):
                if first_encountered and number_non_terminals>1:
                    is_right_child_next = False
                first_encountered = False
            replace.append((symbol_, current_ind, next_depth, type, is_right_child_next))
        deriv_sent = deriv_sent[:ind_in_sent] + replace + deriv_sent[ind_in_sent + 1:]
        return deriv_sent, deriv_rules, deriv_semantic_types, current_ind

    def get_first_non_terminal_typed(self, part_sent, max_depth):
        """Get first non-terminal to derive next."""
        for i, (symbol, index, depth, type, right_child) in enumerate(part_sent):
            if not self.is_terminal(symbol) and depth < max_depth:
                return i, symbol, index, depth, type, right_child

    def apply_semantic_functions(self, all_sf):
        """Produce an executable logical form recursively, given derived semnatic functions."""
        sf_init = all_sf[0][0]
        index = all_sf[0][1]
        args = self.find_args(all_sf, index)
        lf = self.apply_sf(sf_init, args, all_sf)
        lf = self.lf_replace(lf)
        return lf

    def lf_replace(self, lf):
        """
        Transform to an executable form.
        """
        lf_rep = str(tupleit(lf)).replace("'", "").replace(',', '').replace('(', ' ( ').replace(
            ')', ' ) ').replace('!type', '! type').replace('!=', '! =')
        lf_rep = lf_rep.replace('   ', ' ').replace('  ', ' ').strip()
        lf_rep = lf_rep.replace('@', 'SW.')
        return lf_rep

    def apply_sf(self, sf, args, all_sf):
        """Apply semantic functions recursively."""
        if sf[0] == 'ConstantFn':
            return sf[1]
        elif sf[0] == 'lambda':
            sf_curr = sf
            for arg in args:
                var = sf_curr[1]
                expr = sf_curr[2]
                child_sf = arg[0]
                child_ind = arg[1]
                expr = self.beta_replace(expr, var, child_sf, child_ind, all_sf)
                sf_curr = expr
            return expr
        elif sf[0] == 'JoinFn':
            for arg in args:
                is_right_child = arg[3]
                if is_right_child:
                    right_child_sf = arg[0]
                    right_child_ind = arg[1]
                else:
                    left_child_sf = arg[0]
                    left_child_ind = arg[1]
            if sf[1] == 'backward' or sf[2] == 'backward':
                right = self.apply_sf(right_child_sf, self.find_args(all_sf, right_child_ind), all_sf)
                return self.beta_replace(right[2], right[1], left_child_sf, left_child_ind, all_sf)
        elif sf[0] == 'IdentityFn':
            return self.apply_sf(args[0][0], self.find_args(all_sf, args[0][1]), all_sf)

    def find_args(self, all_sf, index):
        """Find the arguments from some predicate."""
        next_args = []
        for i, sf in enumerate(all_sf):
            if sf[2] == index:
                next_args.append(all_sf[i])
        # if there are 2 args, sort from left child to right
        if len(next_args) == 2 and next_args[0][3]:
            next_args = next_args[::-1]
        return next_args

    def beta_replace(self, beta_expr, var, target, target_ind, all_sf):
        expr = ['var', var]
        rep = self.apply_sf(target, self.find_args(all_sf, target_ind), all_sf)
        return self.replace_nested(expr, rep, beta_expr)

    def replace_nested(self, expr, rep, nested_list):
        res = []
        for item in nested_list:
            if not isinstance(item, list):
                res.append(item)
            elif item == expr:
                res.append(rep)
            else:
                res.append(self.replace_nested(expr, rep, item))
        return res


def tupleit(t):
    return tuple(map(tupleit, t)) if isinstance(t, (list, tuple)) else t


def execute_lfs(lfs, subdomain):
    all_lfs = ([format_lf(lf) for lf in lfs])
    tf_lines = all_lfs
    tf = tempfile.NamedTemporaryFile(mode='w', suffix='.examples')
    for line in tf_lines:
      print(line, file=tf)
    tf.flush()
    FNULL = open(os.devnull, 'w')
    eva_path = 'evaluator/scholar_gen' if subdomain == 'external' else 'evaluator/overnight'
    msg = subprocess.check_output([eva_path, subdomain, tf.name], stderr=FNULL)
    tf.close()

    denotations = [line.split('\t')[1] for line in msg.decode("utf-8").split('\n') if line.startswith('targetValue\t')]
    return denotations


def format_lf(lf):
    """Transform to KB naming before execution."""
    replacements = [
        ('! ', '!'),
        ('SW', 'edu.stanford.nlp.sempre.overnight.SimpleWorld'),
    ]
    for a, b in replacements:
      lf = lf.replace(a, b)
    # Balance parentheses
    num_left_paren = sum(1 for c in lf if c == '(')
    num_right_paren = sum(1 for c in lf if c == ')')
    diff = num_left_paren - num_right_paren
    if diff > 0:
      while len(lf) > 0 and lf[-1] == '(' and diff > 0:
        lf = lf[:-1]
        diff -= 1
      if len(lf) == 0: return ''
      lf = lf + ' )' * diff
    return lf


def is_error(d):
    return 'BADJAVA' in d or 'ERROR' in d or d == 'null'


def find_valid_lf(nls, lfs, rules, depths, domain):
    jump = 200000
    counter = 0
    nls_valid = []
    lfs_valid = []
    rules_valid = []
    denotations_valid = []
    depths_valid = []

    while counter <= len(lfs):
      print(counter)
      upper_bound = min(len(lfs), counter+jump)
      lfs_local = lfs[counter:upper_bound]
      nls_local = nls[counter:upper_bound]
      rules_local = rules[counter:upper_bound]
      depths_local = depths[counter:upper_bound]
      denotations = execute_lfs(lfs_local, domain)
      for i, den in enumerate(denotations):
          if not is_error(den) or rep_to_empty_list(den) == '(list)':
            nls_valid.append(nls_local[i])
            lfs_valid.append(lfs_local[i])
            rules_valid.append(rules_local[i])
            depths_valid.append(depths_local[i])
            denotations_valid.append(denotations[i])
      counter += jump
    return nls_valid, lfs_valid, rules_valid, denotations_valid, depths_valid


def rep_to_empty_list(pred_den):
    return pred_den


def write_all_derivs(all_derives, out_path, domain):
    cans = []
    lfs = []
    rules = []
    depths = []

    for i, sent in enumerate(all_derives):
        can = ' '.join([item[0] for item in sent[0]])
        cans.append(can)
        lf = grammar.apply_semantic_functions(sent[2])
        lfs.append(lf)
        rule = ' '.join([str(t) for t in sent[1]])
        rules.append(rule)
        depth = max([symbol[2] for symbol in sent[0]])
        depths.append(depth)
    nls_valid, lfs_valid, rules_valid, denotations_valid, depths_valid = find_valid_lf(
        cans, lfs, rules, depths, domain)

    num_found = len(nls_valid)
    print('found {} valid examples'.format(num_found))
    print(cans[:20])

    if num_found == 0:
        return

    with open(out_path, 'a') as f:
        for (nl, lf, r, d) in zip(nls_valid, lfs_valid, rules_valid, depths_valid):
            f.write('{}\t{}\t{}\t{}\n'.format(nl, lf, r, d))


def prune_generated(input_path, output_path):
    """Prune nonsensical derivations, e.g. 'A and not A'."""
    patterns = [
    re.compile('.* that (.*)( and .*)? and that \\1 whose .*( and .*)?$'),
    re.compile('.* that (.*)( and .*)? and that not \\1 whose .*( and .*)?$'),
    re.compile('.* that not (.*)( and .*)? and that \\1 whose .*( and .*)?$'),

    re.compile('.* that (.*) (.*)( and .*)? and that \\1 \\2( and .*)?$'),

    re.compile('.* that (.*) (.*)( and .*)? and that \\1 \\2( and .*)?$'),
    re.compile('.* that (.*) (.*)( and .*)? and that \\1 not \\2( and .*)?$'),
    re.compile('.* that (.*) not (.*)( and .*)? and that \\1 not \\2( and .*)?$'),
    re.compile('.* that (.*) not (.*)( and .*)? and that \\1 \\2( and .*)?$'),

    re.compile('.* whose (.*) and whose \\1( and .*)?$'),
    re.compile('.* whose (.*) is (.*)( and .*)? and whose \\1 is \\2( and .*)?$'),
    re.compile('.* whose (.*) is not (.*)( and .*)? and whose \\1 is \\2( and .*)?$'),
    re.compile('.* whose (.*) is (.*)( and .*)? and whose \\1 is not \\2( and .*)?$'),

    re.compile('.* that (not )?(.*)( and .*)? and that (not )?\\1( and .*)?$'),

    re.compile('.* that (.*)( and .*)? and that not \\1( and .*)?$'),

    re.compile('.* that (.*) and that \\1$'),
    re.compile('.* that (.*) and that not \\1$'),
    re.compile('.* that not (.*) and that not \\1$'),
    re.compile('.* that not (.*) and that \\1$'),

    re.compile('.* that has the largest .* and that .*$'),
    re.compile('.* that has the smallest .* and that .*$'),
    re.compile('.* that the most number of .* and that .*$'),
    re.compile('.* that the least number of .* and that .*$'),
    re.compile('.* that .* the most number of .* and that .*$'),
    re.compile('.* that .* the least number of .* and that .*$'),
    re.compile('.* that is .* of the most number of .* and that .*$'),
    re.compile('.* that is .* of the least number of .* and that .*$'),

    re.compile('.* that has the largest .* and whose .*$'),
    re.compile('.* that has the smallest .* and whose .*$'),
    re.compile('.* that the most number of .* and whose .*$'),
    re.compile('.* that the least number of .* and whose .*$'),
    re.compile('.* that .* the most number of .* and whose .*$'),
    re.compile('.* that .* the least number of .* and whose .*$'),
    re.compile('.* that is .* of the most number of .* and whose .*$'),
    re.compile('.* that is .* of the least number of .* and whose .*$'),
    ]

    patters_complex = [
        re.compile('(.*) (that .*) and (.*) and (.*)$'),
        re.compile('(.*) (whose .*) and (.*) and (.*)$'),
        re.compile('(.*) (that .*) and (.*)$'),
        re.compile('(.*) (whose .*) and (.*)$'),
    ]

    counter_pruned = 0
    permut_to_prune = []
    with open(input_path, 'r') as f:
        with open(output_path, 'w') as f_write:
            for row in f:
                found = False
                can = row.split('\t')[0]
                lf = row.split('\t')[1]

                if can in permut_to_prune:
                    print('found permute!')
                    continue

                for p in patterns:
                    m = p.match(can)
                    if m is not None:
                        print(can+'\t'+str(p))
                        counter_pruned += 1
                        found = True
                        break
                if not found:
                    f_write.write(can+'\t'+lf+'\n')

                    for p in patters_complex:
                        m = p.match(can)
                        if m is not None:
                            head = can[m.regs[1][0]:m.regs[1][1]]
                            to_cyc = [can[r[0]:r[1]] for r in m.regs[2:]]
                            permuts = permutations(to_cyc)
                            for permut in permuts:
                                after = ' and '.join(list(permut))
                                after = head+' '+after
                                permut_to_prune.append(after)
                            break

    print(counter_pruned)


def _parse_args():
    parser = argparse.ArgumentParser(
      description='experiment parser.',
      formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument('--domain', '-d', default='scholar',
                        choices=['geo', 'scholar'])
    parser.add_argument('--name', '-o', default='all_derives_scholar_4')
    parser.add_argument('--max_depth', '-m', default=4)
    return parser.parse_args()


if __name__ == "__main__":
    DIR = os.path.dirname(__file__)
    args = _parse_args()
    domain = args.domain
    name = args.name
    max_depth = args.max_depth
    general_grammar_path = os.path.join(DIR, '../grammars/general.grammar')
    if domain == 'geo':
        domain_grammar_path = os.path.join(DIR, '../grammars/geo880.grammar')
    else:
        domain_grammar_path = os.path.join(DIR, '../grammars/scholar.grammar')

    grammar = Grammar()
    grammar.from_file(general_grammar_path)
    grammar.from_file(domain_grammar_path)

    print('start')
    domain_for_executor = 'geo880' if domain == 'geo' else 'external'
    all_derives_path = os.path.join(DIR, name+'.txt')
    sents = grammar.gen_exhaustive('$ROOT', all_derives_path, domain_for_executor,
                                   max_depth=max_depth)
    write_all_derivs(sents, all_derives_path, domain_for_executor)

    generated_pruned_path = os.path.join(DIR, name+'_pruned_lf.txt')
    prune_generated(all_derives_path, generated_pruned_path)




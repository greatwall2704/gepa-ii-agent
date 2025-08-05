def is_dominated(y, programs, program_at_pareto_front_valset):
    y_fronts = [front for front in program_at_pareto_front_valset if y in front]
    for front in y_fronts:
        found_dominator_in_front = False
        for other_prog in front:
            if other_prog in programs:
                found_dominator_in_front = True
                break
        if not found_dominator_in_front:
            return False
    
    return True

def remove_dominated_programs(program_at_pareto_front_valset, scores=None):
    freq = {}
    for front in program_at_pareto_front_valset:
        for p in front:
            freq[p] = freq.get(p, 0) + 1

    dominated = set()
    programs = list(freq.keys())

    if scores is None:
        scores = {p:1 for p in programs}
    
    programs = sorted(programs, key=lambda x: scores[x], reverse=False)

    found_to_remove = True
    while found_to_remove:
        found_to_remove = False
        for y in programs:
            if y in dominated:
                continue
            if is_dominated(y, set(programs).difference({y}).difference(dominated), program_at_pareto_front_valset):
                dominated.add(y)
                found_to_remove = True
                break
    
    dominators = [p for p in programs if p not in dominated]
    for front in program_at_pareto_front_valset:
        assert any(p in front for p in dominators)
    
    new_program_at_pareto_front_valset = [{prog_idx for prog_idx in front if prog_idx in dominators} for front in program_at_pareto_front_valset]
    assert len(new_program_at_pareto_front_valset) == len(program_at_pareto_front_valset)
    for front_old, front_new in zip(program_at_pareto_front_valset, new_program_at_pareto_front_valset):
        assert front_new.issubset(front_old)

    return new_program_at_pareto_front_valset

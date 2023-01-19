from tabulate import tabulate


def _get_elapsed_time(func_name, arg, lib, repeat, **options):
    elapsed = []
    mem_info = []
    for i in range(repeat):
        e, m = getattr(lib, func_name)(arg, **options)
        elapsed.append(e)
        mem_info.append(m)
    elapsed = sum(elapsed) / len(elapsed)
    mem_info = {"min": min([m["min"] for m in mem_info]),
                "max": max([m["max"] for m in mem_info]),
                "avg": sum([m["avg"] for m in mem_info]) / len(mem_info)}
    return elapsed, mem_info


def _print_table(data):
    lib_names = list(sorted(list(data.keys())))
    kinds = [k.split("=")[0]
             for lib_name in lib_names
             for k, _ in data[lib_name].items()]
    kinds = list(sorted(list(set(kinds))))
    for f in kinds:
        table = []
        for lib_name in lib_names:
            raws = sorted([(k, v) for k, v in data[lib_name].items()
                           if k.startswith(f)],
                          key=lambda x: (len(x[0]), x[0]))
            rows = [v for _, v in raws]
            headers = ["method"] + [k for k, _ in raws]
            table.append([lib_name] + rows)
        if table:
            print(tabulate(table, headers=headers, tablefmt="github"))
            print("")
